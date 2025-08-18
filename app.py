import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import time
import gdown
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Local utilities
from utils import (
    read_uploaded_file_bytes,
    preprocess_text,
    parse_topics_field,
    style_suspicious_and_low,
    load_model_from_source,
    encode_texts_in_batches,
)

# Optional multimodal helpers (we'll gracefully fallback to HF if not present)
try:
    from multimodal import (
        load_clip_model as _mm_load_clip_model,
        load_blip_model as _mm_load_blip_model,
        clip_image_text_score as _mm_clip_image_text_score,
        blip_generate_caption as _mm_blip_generate_caption,
    )
    _HAS_MM_HELPERS = True
except Exception:
    _HAS_MM_HELPERS = False

from PIL import Image

# Transformers fallback loaders (only if multimodal.py not available)
@st.cache_resource(show_spinner=False)
def _hf_load_clip(model_id: str = "openai/clip-vit-base-patch32"):
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained(model_id)
    proc = CLIPProcessor.from_pretrained(model_id)
    return model, proc

@st.cache_resource(show_spinner=False)
def _hf_load_blip(model_id: str = "Salesforce/blip-image-captioning-base"):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    proc = BlipProcessor.from_pretrained(model_id)
    return model, proc

# ==================== Session State ====================
_DEF_TEXT_STATE = {
    "model": None,
    "model_id": None,
    "source": None,
    "data_df": None,
    "data_hash": None,
    "embeddings": None,
    "history": [],  # list of dicts
}

_DEF_MM_STATE = {
    "model_kind": None,  # "CLIP" or "BLIP"
    "source": None,
    "model_id": None,
    "model": None,
    "processor": None,
    "compare": False,
    "model2_id": None,
    "model2": None,
    "processor2": None,
    "history": [],  # list of dicts
}

for key, val in {
    "text": _DEF_TEXT_STATE.copy(),
    "mm": _DEF_MM_STATE.copy(),
}.items():
    st.session_state.setdefault(key, val)

# ==================== Helpers ====================

def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

@st.cache_resource(show_spinner=False)
def _load_clip_any(source: str, identifier: str):
    if _HAS_MM_HELPERS:
        return _mm_load_clip_model(source, identifier)
    # Fallback: only HF is supported here
    if source != "huggingface":
        raise ValueError("CLIP fallback поддерживает только загрузку из HuggingFace")
    return _hf_load_clip(identifier)

@st.cache_resource(show_spinner=False)
def _load_blip_any(source: str, identifier: str):
    if _HAS_MM_HELPERS:
        return _mm_load_blip_model(source, identifier)
    if source != "huggingface":
        raise ValueError("BLIP fallback поддерживает только загрузку из HuggingFace")
    return _hf_load_blip(identifier)

# ==================== UI: Sidebar Switch ====================
st.set_page_config(page_title="Model Analysis — Text & Multimodal", layout="wide")
st.sidebar.title("Настройка модели")
mode = st.sidebar.radio(
    "Выберите режим работы:",
    ["Работа с текстовыми моделями", "Работа с мультимодальными моделями"],
    index=0,
)

# ==================== TEXT MODE ====================

def render_text_mode():
    st.sidebar.subheader("Текстовые модели")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        source = st.selectbox("Источник модели", ["huggingface", "google_drive"], key="txt_source")
    with col_b:
        identifier = st.text_input("ID/путь модели", key="txt_identifier", placeholder="sentence-transformers/all-MiniLM-L6-v2")

    if st.sidebar.button("Загрузить текстовую модель", use_container_width=True):
        with st.spinner("Загрузка модели..."):
            try:
                model = load_model_from_source(source, identifier)
                st.session_state["text"].update({
                    "model": model,
                    "model_id": identifier,
                    "source": source,
                    "embeddings": None,
                })
                st.success(f"Модель загружена: {identifier}")
            except Exception as e:
                st.error(f"Не удалось загрузить модель: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("История (текст)")
    col_h1, col_h2 = st.sidebar.columns(2)
    with col_h1:
        if st.button("Скачать историю", key="txt_hist_dl", use_container_width=True):
            hist = st.session_state["text"]["history"]
            st.download_button("Скачать JSON", data=json.dumps(hist, ensure_ascii=False, indent=2).encode("utf-8"), file_name="text_history.json", mime="application/json")
    with col_h2:
        if st.button("Удалить историю", key="txt_hist_rm", use_container_width=True):
            st.session_state["text"]["history"] = []
            st.success("История очищена")

    st.markdown("## Работа с текстовыми моделями")
    tcols = st.columns([2, 3])
    with tcols[0]:
        st.markdown("### Данные для анализа")
        uploaded = st.file_uploader("Загрузите CSV/Excel/JSON с колонкой текстов", type=["csv", "xlsx", "json", "ndjson"], key="txt_file")
        text_col = st.text_input("Имя колонки с текстом", value="text", key="txt_col")
        if uploaded is not None:
            try:
                df, h = read_uploaded_file_bytes(uploaded)
                st.session_state["text"]["data_df"] = df
                st.session_state["text"]["data_hash"] = h
                st.success(f"Файл загружен, строк: {len(df)}")
                st.dataframe(df.head(10))
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
    with tcols[1]:
        st.markdown("### Поиск по схожести (Semantic Search)")
        query = st.text_input("Запрос")
        topk = st.number_input("Сколько результатов показать", 1, 100, 5)
        if st.button("Искать", use_container_width=True):
            model = st.session_state["text"]["model"]
            df = st.session_state["text"].get("data_df")
            if model is None:
                st.warning("Сначала загрузите текстовую модель в сайдбаре.")
            elif df is None or text_col not in df.columns:
                st.warning("Загрузите данные и укажите корректную колонку текста.")
            else:
                with st.spinner("Вычисление эмбеддингов..."):
                    # Cache embeddings per data hash + model id
                    if st.session_state["text"]["embeddings"] is None:
                        texts = [preprocess_text(x) for x in df[text_col].fillna("")]
                        embs = encode_texts_in_batches(model, texts, batch_size=64)
                        st.session_state["text"]["embeddings"] = embs
                    else:
                        embs = st.session_state["text"]["embeddings"]

                    q_emb = encode_texts_in_batches(model, [preprocess_text(query)], batch_size=1)
                    sims = np.dot(embs, q_emb[0]) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb[0]) + 1e-9)
                    idx = np.argsort(-sims)[: int(topk)]
                    out_df = df.iloc[idx].copy()
                    out_df.insert(0, "similarity", sims[idx])
                    st.dataframe(out_df)
                    st.session_state["text"]["history"].append({
                        "ts": _timestamp(),
                        "action": "semantic_search",
                        "query": query,
                        "topk": int(topk),
                        "model": st.session_state["text"]["model_id"],
                    })

# ==================== MULTIMODAL MODE ====================

def _download_history_button(hist: List[Dict[str, Any]], label: str, fname: str):
    payload = json.dumps(hist, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(label, data=payload, file_name=fname, mime="application/json", use_container_width=True)


def render_multimodal_mode():
    st.sidebar.subheader("Мультимодальные модели")

    model_kind = st.sidebar.selectbox("Тип модели", ["CLIP (сходство изображение↔текст)", "BLIP (подписи к изображению)"])
    kind_key = "CLIP" if model_kind.startswith("CLIP") else "BLIP"

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        source = st.selectbox("Источник модели", ["huggingface", "google_drive"], key="mm_source")
    with col_b:
        default_id = "openai/clip-vit-base-patch32" if kind_key == "CLIP" else "Salesforce/blip-image-captioning-base"
        identifier = st.text_input("ID/путь модели", key="mm_identifier", value=default_id)

    compare = st.sidebar.checkbox("Сравнение двух моделей")
    identifier2 = None
    source2 = None
    if compare:
        col_c, col_d = st.sidebar.columns(2)
        with col_c:
            source2 = st.selectbox("Источник 2", ["huggingface", "google_drive"], key="mm_source2")
        with col_d:
            identifier2 = st.text_input("ID/путь модели 2", key="mm_identifier2", value=default_id)

    if st.sidebar.button("Загрузить мультимодальную модель(и)", use_container_width=True):
        with st.spinner("Загрузка модели(ей)..."):
            try:
                if kind_key == "CLIP":
                    m1, p1 = _load_clip_any(source, identifier)
                    st.session_state["mm"].update({
                        "model_kind": kind_key,
                        "source": source,
                        "model_id": identifier,
                        "model": m1,
                        "processor": p1,
                    })
                    if compare and identifier2:
                        m2, p2 = _load_clip_any(source2, identifier2)
                        st.session_state["mm"].update({
                            "compare": True,
                            "model2_id": identifier2,
                            "model2": m2,
                            "processor2": p2,
                        })
                    else:
                        st.session_state["mm"].update({"compare": False, "model2": None, "processor2": None, "model2_id": None})
                else:  # BLIP
                    m1, p1 = _load_blip_any(source, identifier)
                    st.session_state["mm"].update({
                        "model_kind": kind_key,
                        "source": source,
                        "model_id": identifier,
                        "model": m1,
                        "processor": p1,
                    })
                    if compare and identifier2:
                        m2, p2 = _load_blip_any(source2, identifier2)
                        st.session_state["mm"].update({
                            "compare": True,
                            "model2_id": identifier2,
                            "model2": m2,
                            "processor2": p2,
                        })
                    else:
                        st.session_state["mm"].update({"compare": False, "model2": None, "processor2": None, "model2_id": None})
                st.success("Модель(и) загружены")
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("История (мультимодальные)")
    col_h1, col_h2 = st.sidebar.columns(2)
    with col_h1:
        _download_history_button(st.session_state["mm"]["history"], "Скачать историю", "multimodal_history.json")
    with col_h2:
        if st.button("Удалить историю", use_container_width=True):
            st.session_state["mm"]["history"] = []
            st.success("История очищена")

    # MAIN AREA
    st.markdown("## Работа с мультимодальными моделями")
    if st.session_state["mm"]["model"] is None:
        st.info("Сначала выберите и загрузите модель(и) в сайдбаре")
        return

    if st.session_state["mm"]["model_kind"] == "CLIP":
        st.markdown("### Режим: CLIP — сходство изображение↔текст")
        img = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"], key="clip_img")
        prompt = st.text_input("Текстовый запрос")
        run = st.button("Вычислить сходство", use_container_width=True)
        if run:
            if not img or not prompt:
                st.warning("Нужно загрузить изображение и ввести текст")
            else:
                image = Image.open(io.BytesIO(img.read())).convert("RGB")
                m1, p1 = st.session_state["mm"]["model"], st.session_state["mm"]["processor"]
                m2, p2 = st.session_state["mm"].get("model2"), st.session_state["mm"].get("processor2")
                with st.spinner("Считаем..."):
                    try:
                        if _HAS_MM_HELPERS and "clip_image_text_score" in globals():
                            score1 = _mm_clip_image_text_score(m1, p1, image, prompt)
                            score2 = _mm_clip_image_text_score(m2, p2, image, prompt) if m2 else None
                        else:
                            # HF fallback
                            from transformers import CLIPProcessor
                            assert isinstance(p1, CLIPProcessor)
                            inputs = p1(text=[prompt], images=image, return_tensors="pt", padding=True)
                            import torch
                            with torch.no_grad():
                                logits = m1(**inputs).logits_per_image
                                score1 = float(torch.softmax(logits, dim=1)[0, 0].cpu().item())
                            if m2:
                                inputs2 = p2(text=[prompt], images=image, return_tensors="pt", padding=True)
                                with torch.no_grad():
                                    logits2 = m2(**inputs2).logits_per_image
                                    score2 = float(torch.softmax(logits2, dim=1)[0, 0].cpu().item())
                            else:
                                score2 = None
                    except Exception as e:
                        st.error(f"Ошибка вычисления: {e}")
                        return
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image, caption="Входное изображение", use_column_width=True)
                with c2:
                    st.metric(label=f"Сходство (модель 1: {st.session_state['mm']['model_id']})", value=round(score1, 4))
                    if score2 is not None:
                        st.metric(label=f"Сходство (модель 2: {st.session_state['mm']['model2_id']})", value=round(score2, 4))
                st.session_state["mm"]["history"].append({
                    "ts": _timestamp(),
                    "kind": "CLIP",
                    "image_name": getattr(img, "name", "uploaded"),
                    "text": prompt,
                    "model1": st.session_state["mm"]["model_id"],
                    "model2": st.session_state["mm"].get("model2_id"),
                    "score1": float(score1),
                    "score2": float(score2) if score2 is not None else None,
                })

    else:  # BLIP
        st.markdown("### Режим: BLIP — генерация подписи к изображению")
        img = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"], key="blip_img")
        run = st.button("Сгенерировать подпись", use_container_width=True)
        if run:
            if not img:
                st.warning("Нужно загрузить изображение")
            else:
                image = Image.open(io.BytesIO(img.read())).convert("RGB")
                m1, p1 = st.session_state["mm"]["model"], st.session_state["mm"]["processor"]
                m2, p2 = st.session_state["mm"].get("model2"), st.session_state["mm"].get("processor2")
                with st.spinner("Генерируем..."):
                    try:
                        if _HAS_MM_HELPERS and "blip_generate_caption" in globals():
                            cap1 = _mm_blip_generate_caption(m1, p1, image)
                            cap2 = _mm_blip_generate_caption(m2, p2, image) if m2 else None
                        else:
                            # HF fallback
                            import torch
                            from transformers import BlipProcessor
                            assert p1 is not None
                            assert isinstance(p1, BlipProcessor)
                            inputs = p1(image, return_tensors="pt")
                            out = m1.generate(**inputs, max_new_tokens=30)
                            cap1 = p1.decode(out[0], skip_special_tokens=True)
                            if m2:
                                inputs2 = p2(image, return_tensors="pt")
                                out2 = m2.generate(**inputs2, max_new_tokens=30)
                                cap2 = p2.decode(out2[0], skip_special_tokens=True)
                            else:
                                cap2 = None
                    except Exception as e:
                        st.error(f"Ошибка генерации: {e}")
                        return
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image, caption="Входное изображение", use_column_width=True)
                with c2:
                    st.write(f"**Модель 1 ({st.session_state['mm']['model_id']}):**\n{cap1}")
                    if cap2 is not None:
                        st.write(f"**Модель 2 ({st.session_state['mm']['model2_id']}):**\n{cap2}")
                st.session_state["mm"]["history"].append({
                    "ts": _timestamp(),
                    "kind": "BLIP",
                    "image_name": getattr(img, "name", "uploaded"),
                    "model1": st.session_state["mm"]["model_id"],
                    "model2": st.session_state["mm"].get("model2_id"),
                    "caption1": cap1,
                    "caption2": cap2,
                })

# ==================== MAIN ROUTER ====================
if mode == "Работа с текстовыми моделями":
    render_text_mode()
else:
    render_multimodal_mode()
