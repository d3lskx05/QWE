import pandas as pd
import numpy as np
import hashlib
import io
import json
import re
from typing import Any, List, Tuple, Dict

def preprocess_text(t: Any) -> str:
    if pd.isna(t):
        return ""
    return " ".join(str(t).lower().strip().split())

def file_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def _try_read_json(raw: bytes) -> pd.DataFrame:
    try:
        obj = json.loads(raw.decode("utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
    except Exception:
        pass
    try:
        return pd.read_json(io.BytesIO(raw), lines=True)
    except Exception:
        pass
    raise ValueError("Не удалось распознать JSON/NDJSON")

def read_uploaded_file_bytes(uploaded) -> Tuple[pd.DataFrame, str]:
    raw = uploaded.read()
    h = file_md5(raw)
    name = (uploaded.name or "").lower()
    if name.endswith(".json") or name.endswith(".ndjson"):
        df = _try_read_json(raw)
        return df, h
    try:
        df = pd.read_csv(io.BytesIO(raw))
        return df, h
    except Exception:
        pass
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
                return [str(x).strip() for x in parsed if x.strip()]
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
        score = float(row.get('score', 0) or 0)
        lex = float(row.get('lexical_score', 0) or 0)
        is_low_score = (score < low_score_thresh)
        is_suspicious = (score >= sem_thresh and lex <= lex_thresh)
        for _ in row:
            if is_suspicious:
                out.append('background-color: #fff2b8')
            elif is_low_score:
                out.append('background-color: #ffcccc')
            else:
                out.append('')
        return out
    return df.style.apply(highlight, axis=1)

NEG_PAT = re.compile(r"\\bне\\b|\\bни\\b|\\bнет\\b", flags=re.IGNORECASE)
NUM_PAT = re.compile(r"\\b\\d+\\b")
DATE_PAT = re.compile(r"\\b\\d{1,2}[./-]\\d{1,2}([./-]\\d{2,4})?\\b")

def simple_flags(text: str) -> Dict[str, bool]:
    t = text or ""
    return {
        "has_neg": bool(NEG_PAT.search(t)),
        "has_num": bool(NUM_PAT.search(t)),
        "has_date": bool(DATE_PAT.search(t)),
        "len_char": len(t),
        "len_tok": len([x for x in t.split() if x]),
    }

try:
    import pymorphy2
    _MORPH = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH = None

def pos_first_token(text: str) -> str:
    if _MORPH is None:
        return "NA"
    toks = [t for t in text.split() if t]
    if not toks:
        return "NA"
    p = _MORPH.parse(toks[0])[0]
    return str(p.tag.POS) if p and p.tag and p.tag.POS else "NA"

def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 42, ci: float = 0.95):
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
