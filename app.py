import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import io

from utils import (
    preprocess_text, parse_topics_field, read_uploaded_file_bytes,
    simple_flags, pos_first_token, style_suspicious_and_low,
    jaccard_tokens, bootstrap_diff_ci
)
from models import load_model_from_source, encode_texts_in_batches

st.set_page_config(page_title="Synonym Checker", layout="wide")
st.title("🔎 Synonym Checker")

with st.sidebar:
    st.header("Настройки модели")
    model_source = st.radio("Источник модели", ["huggingface", "google_drive"])
    model_identifier = st.text_input("ID модели", value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if st.button("Загрузить модель"):
        with st.spinner("Загружаем модель..."):
            model = load_model_from_source(model_source, model_identifier)
        st.success("Модель загружена!")

uploaded = st.file_uploader("Загрузите файл (CSV, Excel, JSON)", type=["csv", "xlsx", "json", "ndjson"])

if uploaded:
    try:
        df, file_hash = read_uploaded_file_bytes(uploaded)
        st.write(f"Файл загружен: {uploaded.name}, {len(df)} строк")
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "text" not in df.columns or "topics" not in df.columns:
        st.error("Файл должен содержать колонки 'text' и 'topics'")
        st.stop()

    df["text_clean"] = df["text"].apply(preprocess_text)
    df["topics_list"] = df["topics"].apply(parse_topics_field)

    if st.button("Рассчитать сходство"):
        if "model" not in locals():
            st.error("Сначала загрузите модель!")
            st.stop()
        with st.spinner("Вычисляем эмбеддинги..."):
            texts = df["text_clean"].tolist()
            embeddings = encode_texts_in_batches(model, texts)
            df["embedding"] = list(embeddings)

        st.success("Эмбеддинги рассчитаны")

        st.subheader("📊 Статистика")
        st.write(df.head())

        st.download_button(
            "Скачать результат (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="result.csv",
            mime="text/csv"
        )
