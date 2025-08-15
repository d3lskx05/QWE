import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sentence_transformers import util
from utils import (
    preprocess_text, read_uploaded_file_bytes, parse_topics_field, simple_flags,
    pos_first_token, encode_texts_in_batches, jaccard_tokens, style_suspicious_and_low,
    bootstrap_diff_ci, load_model_from_source
)

# ============== Настройки страницы ==============
st.set_page_config(page_title="Анализ ответов участников", layout="wide")
st.title("📊 Анализ ответов участников")

# ============== Боковая панель: загрузка данных ==============
st.sidebar.header("Загрузка данных")
uploaded = st.sidebar.file_uploader("Загрузите CSV/Excel/JSON", type=["csv", "xlsx", "json", "ndjson"])

# ============== Основная логика ==============
if uploaded:
    try:
        df, file_hash = read_uploaded_file_bytes(uploaded)
        st.success(f"Загружено {len(df)} строк, {len(df.columns)} колонок.")
        st.write("Предпросмотр данных:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

    # ===== Выбор колонок =====
    st.sidebar.subheader("Настройка колонок")
    col_answer = st.sidebar.selectbox("Колонка с ответами", df.columns)
    col_topics = st.sidebar.selectbox("Колонка с темами", df.columns)

    # ===== Настройка модели =====
    st.sidebar.subheader("Модель")
    model_source = st.sidebar.radio("Источник модели", ["huggingface", "google_drive"])
    model_id = st.sidebar.text_input("ID или имя модели", value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # ===== Кнопка запуска =====
    if st.sidebar.button("Выполнить анализ"):
        with st.spinner("Загрузка модели..."):
            model = load_model_from_source(model_source, model_id)

        st.subheader("📥 Предобработка данных")
        df["_answer_clean"] = df[col_answer].apply(preprocess_text)
        df["_topics_list"] = df[col_topics].apply(parse_topics_field)
        df["_topic_main"] = df["_topics_list"].apply(lambda lst: lst[0] if lst else "")

        # ===== Семантические признаки =====
        st.info("Вычисление эмбеддингов...")
        embeddings = encode_texts_in_batches(model, df["_answer_clean"].tolist())

        # ===== Лексические и простые признаки =====
        st.info("Подсчёт лексических признаков...")
        df["_neg"] = df["_answer_clean"].apply(lambda t: simple_flags(t)["has_neg"])
        df["_num"] = df["_answer_clean"].apply(lambda t: simple_flags(t)["has_num"])
        df["_len_char"] = df["_answer_clean"].apply(lambda t: simple_flags(t)["len_char"])
        df["_len_tok"] = df["_answer_clean"].apply(lambda t: simple_flags(t)["len_tok"])
        df["_pos_first"] = df["_answer_clean"].apply(pos_first_token)

        # ===== Сравнение с эталоном (если есть) =====
        st.subheader("🔍 Анализ схожести")
        topic_groups = df.groupby("_topic_main")
        result_rows = []
        for topic, group in topic_groups:
            if len(group) < 2:
                continue
            idxs = group.index.tolist()
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    a_idx, b_idx = idxs[i], idxs[j]
                    emb_a, emb_b = embeddings[a_idx], embeddings[b_idx]
                    sem_score = float(util.cos_sim(emb_a, emb_b))
                    lex_score = jaccard_tokens(df["_answer_clean"].iloc[a_idx], df["_answer_clean"].iloc[b_idx])
                    result_rows.append({
                        "topic": topic,
                        "id_a": a_idx,
                        "id_b": b_idx,
                        "answer_a": df[col_answer].iloc[a_idx],
                        "answer_b": df[col_answer].iloc[b_idx],
                        "score": sem_score,
                        "lexical_score": lex_score
                    })
        if not result_rows:
            st.warning("Недостаточно данных для сравнения.")
            st.stop()
        sim_df = pd.DataFrame(result_rows)

        st.write("Таблица парных сравнений:")
        st.dataframe(sim_df.head(20))

        # ===== Статистика по темам =====
        st.subheader("📈 Статистика по темам")
        agg = sim_df.groupby("topic")["score"].mean().reset_index().rename(columns={"score": "mean_score"})
        chart = alt.Chart(agg).mark_bar().encode(
            x="topic",
            y="mean_score",
            tooltip=["topic", "mean_score"]
        ).properties(width=700, height=400)
        st.altair_chart(chart)

        # ===== Подсветка подозрительных пар =====
        st.subheader("⚠️ Подозрительные пары")
        sem_thresh = st.slider("Порог семантической близости", 0.7, 1.0, 0.85, 0.01)
        lex_thresh = st.slider("Макс. лексическая схожесть", 0.0, 0.5, 0.2, 0.01)
        low_score_thresh = st.slider("Порог низкой схожести", 0.0, 1.0, 0.4, 0.01)

        st.write("Подсветка строк:")
        styled = style_suspicious_and_low(sim_df, sem_thresh, lex_thresh, low_score_thresh)
        st.dataframe(styled, use_container_width=True)

        # ===== Бутстрэп-анализ =====
        st.subheader("📊 Bootstrap-анализ различий по темам")
        topics = sim_df["topic"].unique().tolist()
        if len(topics) >= 2:
            t1 = st.selectbox("Тема 1", topics, index=0)
            t2 = st.selectbox("Тема 2", topics, index=1)
            arr1 = sim_df.loc[sim_df["topic"] == t1, "score"].values
            arr2 = sim_df.loc[sim_df["topic"] == t2, "score"].values
            diff, low, high = bootstrap_diff_ci(arr1, arr2)
            st.write(f"Средняя разница: {diff:.4f}, 95% CI: [{low:.4f}, {high:.4f}]")
        else:
            st.info("Недостаточно тем для бутстрэп-анализа.")
