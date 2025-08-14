# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from sentence_transformers import util

from utils import (
    preprocess_text, read_uploaded_file_bytes, parse_topics_field,
    jaccard_tokens, style_suspicious_and_low, simple_flags, pos_first_token,
    bootstrap_diff_ci
)
from model_utils import load_model_from_source, encode_texts_in_batches

# ==============================
# Настройки страницы
# ==============================
st.set_page_config(page_title="Семантический поиск", layout="wide")
st.title("🔍 Семантический поиск по фразам")

# ==============================
# Сайдбар — загрузка модели
# ==============================
st.sidebar.header("⚙️ Настройки модели")
source = st.sidebar.selectbox("Источник модели", ["huggingface", "google_drive"])
identifier = st.sidebar.text_input("ID модели / путь", value="paraphrase-multilingual-MiniLM-L12-v2")

if st.sidebar.button("Загрузить модель"):
    with st.spinner("Загрузка модели..."):
        model = load_model_from_source(source, identifier)
        st.session_state["model"] = model
        st.success("Модель успешно загружена")

# ==============================
# Загрузка файлов
# ==============================
uploaded_files = st.file_uploader(
    "Загрузите файлы (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json", "ndjson"],
    accept_multiple_files=True
)

dataframes = []
hashes = []
if uploaded_files:
    for f in uploaded_files:
        try:
            df, h = read_uploaded_file_bytes(f)
            dataframes.append(df)
            hashes.append(h)
        except Exception as e:
            st.error(f"Ошибка в файле {f.name}: {e}")

# ==============================
# Параметры поиска
# ==============================
st.sidebar.header("🔍 Параметры поиска")
query = st.sidebar.text_input("Поисковый запрос")
semantic_threshold = st.sidebar.slider("Порог семантики", 0.0, 1.0, 0.7, 0.01)
lexical_threshold = st.sidebar.slider("Порог лексики", 0.0, 1.0, 0.3, 0.01)
low_score_threshold = st.sidebar.slider("Порог низкого сходства", 0.0, 1.0, 0.5, 0.01)
top_k = st.sidebar.number_input("Сколько результатов показывать", 1, 100, 10)

# ==============================
# Выполнение поиска
# ==============================
if st.button("Запустить поиск"):
    if "model" not in st.session_state:
        st.error("Сначала загрузите модель")
    elif not dataframes:
        st.error("Загрузите хотя бы один файл")
    elif not query.strip():
        st.error("Введите поисковый запрос")
    else:
        model = st.session_state["model"]

        # Объединяем все фразы
        combined_df = pd.concat(dataframes, ignore_index=True)
        if "phrase" not in combined_df.columns:
            st.error("В данных нет колонки 'phrase'")
        else:
            combined_df["phrase_proc"] = combined_df["phrase"].map(preprocess_text)

            # Кодируем
            with st.spinner("Вычисление эмбеддингов..."):
                query_emb = encode_texts_in_batches(model, [query])[0]
                corpus_embs = encode_texts_in_batches(model, combined_df["phrase_proc"].tolist())

            # Семантическое сходство
            scores = util.cos_sim(query_emb, corpus_embs)[0].cpu().numpy()

            # Лексическое сходство
            lexical_scores = [jaccard_tokens(query.lower(), p) for p in combined_df["phrase_proc"]]

            # Формируем таблицу
            results_df = combined_df.copy()
            results_df["score"] = scores
            results_df["lexical_score"] = lexical_scores

            # Подсветка подозрительных и низких
            styled = style_suspicious_and_low(results_df, semantic_threshold, lexical_threshold, low_score_threshold)
            st.subheader("📋 Результаты")
            st.dataframe(styled, use_container_width=True)

            # Метрики
            st.subheader("📊 Метрики")
            st.write(f"Среднее семантическое сходство: {np.mean(scores):.3f}")
            st.write(f"Среднее лексическое сходство: {np.mean(lexical_scores):.3f}")

            # График распределения
            chart_data = pd.DataFrame({
                "Семантическое": scores,
                "Лексическое": lexical_scores
            })
            chart = alt.Chart(chart_data.reset_index()).mark_circle(size=60).encode(
                x="Семантическое", y="Лексическое"
            )
            st.altair_chart(chart, use_container_width=True)
