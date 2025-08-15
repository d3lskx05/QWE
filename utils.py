# utils.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
import json
import tempfile
import os
import shutil
import zipfile
import time
import tarfile
import re
from typing import List, Tuple, Dict, Any

from sentence_transformers import SentenceTransformer, util

# ============== Утилиты ==============

def preprocess_text(t: Any) -> str:
    if pd.isna(t):
        return ""
    return " ".join(str(t).lower().strip().split())

def file_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def _try_read_json(raw: bytes) -> pd.DataFrame:
    """
    Пытаемся прочитать JSON/NDJSON в таблицу.
    Поддержка форматов:
      - [{"phrase_1": "...", "phrase_2": "...", ...}, ...]
      - NDJSON (по строке на объект)
      - {"phrase_1": [...], "phrase_2":[...], ...} (ориентация columns)
    """
    # 1) список объектов
    try:
        obj = json.loads(raw.decode("utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            # columns-orient
            return pd.DataFrame(obj)
    except Exception:
        pass
    # 2) NDJSON
    try:
        return pd.read_json(io.BytesIO(raw), lines=True)
    except Exception:
        pass
    raise ValueError("Не удалось распознать JSON/NDJSON")

def read_uploaded_file_bytes(uploaded) -> Tuple[pd.DataFrame, str]:
    raw = uploaded.read()
    h = file_md5(raw)
    # Пытаемся по расширению
    name = (uploaded.name or "").lower()
    if name.endswith(".json") or name.endswith(".ndjson"):
        df = _try_read_json(raw)
        return df, h
    # CSV
    try:
        df = pd.read_csv(io.BytesIO(raw))
        return df, h
    except Exception:
        pass
    # Excel
    try:
        df = pd.read_excel(io.BytesIO(raw))
        return df, h
    except Exception as e:
        raise ValueError("Файл должен быть CSV, Excel или JSON. Ошибка: " + str(e))

def parse_topics_field(val) -> List[str]:
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in [";", "|", ","]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s] if s else []

def jaccard_tokens(a: str, b: str) -> float:
    sa = set([t for t in a.split() if t])
    sb = set([t for t in b.split() if t])
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0

def style_suspicious_and_low(df, sem_thresh: float, lex_thresh: float, low_score_thresh: float):
    def highlight(row):
        out = []
        try:
            score = float(row.get('score', 0))
        except Exception:
            score = 0.0
        try:
            lex = float(row.get('lexical_score', 0))
        except Exception:
            lex = 0.0
        is_low_score = (score < low_score_thresh)
        is_suspicious = (score >= sem_thresh and lex <= lex_thresh)
        for _ in row:
            if is_suspicious:
                out.append('background-color: #fff2b8')  # жёлтый
            elif is_low_score:
                out.append('background-color: #ffcccc')  # розовый
            else:
                out.append('')
        return out
    return df.style.apply(highlight, axis=1)

# ======== Простые признаки для аналитики (без тяжёлых зависимостей) ========

NEG_PAT = re.compile(r"\bне\b|\bни\b|\bнет\b", flags=re.IGNORECASE)
NUM_PAT = re.compile(r"\b\d+\b")
DATE_PAT = re.compile(r"\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b")

def simple_flags(text: str) -> Dict[str, bool]:
    t = text or ""
    return {
        "has_neg": bool(NEG_PAT.search(t)),
        "has_num": bool(NUM_PAT.search(t)),
        "has_date": bool(DATE_PAT.search(t)),
        "len_char": len(t),
        "len_tok": len([x for x in t.split() if x]),
    }

# Морфология (опционально)
try:
    import pymorphy2   # type: ignore
    _MORPH = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH = None

def pos_first_token(text: str) -> str:
    """Очень лёгкая POS-метка по первому токену (если pymorphy2 доступен)."""
    if _MORPH is None:
        return "NA"
    toks = [t for t in text.split() if t]
    if not toks:
        return "NA"
    p = _MORPH.parse(toks[0])[0]
    return str(p.tag.POS) if p and p.tag and p.tag.POS else "NA"

# ======== Бутстрэп CI для A/B ========
def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 42, ci: float = 0.95):
    """Возвращает (mean_diff, low, high)."""
    rng = np.random.default_rng(seed)
    diffs = []
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0
    a = np.asarray(a)
    b = np.asarray(b)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx] - b[idx]))
    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    low = float(np.quantile(diffs, (1-ci)/2))
    high = float(np.quantile(diffs, 1-(1-ci)/2))
    return mean_diff, low, high

# ============== Загрузка модели ==============

def download_file_from_gdrive(file_id: str) -> str:
    import gdown
    tmp_dir = tempfile.gettempdir()
    archive_path = os.path.join(tmp_dir, f"model_gdrive_{file_id}")
    model_dir = os.path.join(tmp_dir, f"model_gdrive_extracted_{file_id}")
    if not os.path.exists(archive_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, archive_path, quiet=True)
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        return model_dir
    os.makedirs(model_dir, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(model_dir)
    else:
        try:
            shutil.copy(archive_path, model_dir)
        except Exception:
            pass
    return model_dir

@st.cache_resource(show_spinner=False)
def load_model_from_source(source: str, identifier: str) -> SentenceTransformer:
    if source == "huggingface":
        model_path = identifier
    elif source == "google_drive":
        model_path = download_file_from_gdrive(identifier)
    else:
        raise ValueError("Unknown model source")
    model = SentenceTransformer(model_path)
    return model

def encode_texts_in_batches(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embs)
