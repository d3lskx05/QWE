import streamlit as st

# ===================== Страница =====================
st.set_page_config(page_title="AI Analyzer", layout="wide")
st.title("🧠 AI Analyzer")

# ===================== Переключатель режимов =====================
mode = st.sidebar.radio(
    "Выберите режим работы:",
    ["Работа с текстовыми моделями", "Работа с мультимодальными моделями"],
    index=0
)

# ===================== РЕЖИМ: ТЕКСТОВЫЕ МОДЕЛИ =====================
if mode == "Работа с текстовыми моделями":
    import altair as alt
    import pandas as pd
    import numpy as np
    import json
    from typing import List

    # импортируем ровно то, что используем
    from utils import (
        preprocess_text,
        parse_topics_field,
        jaccard_tokens,
        style_suspicious_and_low,
        simple_flags,
        pos_first_token,
        load_model_from_source,
        encode_texts_in_batches,
        bootstrap_diff_ci,
        _MORPH,
    )

    from sentence_transformers import util  # нужен для cos_sim

    # ===================== Константы/настройки =====================
    DEFAULT_HF = "sentence-transformers/all-MiniLM-L6-v2"
    HISTORY_MAX = 500  # лимит на длину истории, чтобы не разрасталась

    st.header("🔎 Synonym Checker")

    # ===================== Сайдбар: Модели =====================
    st.sidebar.header("Настройки модели")
    model_source = st.sidebar.selectbox("Источник модели", ["huggingface", "google_drive"], index=0)
    if model_source == "huggingface":
        model_id = st.sidebar.text_input("Hugging Face Model ID", value=DEFAULT_HF)
    else:
        model_id = st.sidebar.text_input("Google Drive File ID", value="1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf")

    enable_ab_test = st.sidebar.checkbox("Включить A/B тест двух моделей", value=False)
    if enable_ab_test:
        ab_model_source = st.sidebar.selectbox("Источник второй модели", ["huggingface", "google_drive"], index=0, key="ab_source")
        if ab_model_source == "huggingface":
            ab_model_id = st.sidebar.text_input("Hugging Face Model ID (B)", value="sentence-transformers/all-mpnet-base-v2", key="ab_id")
        else:
            ab_model_id = st.sidebar.text_input("Google Drive File ID (B)", value="", key="ab_id")
    else:
        ab_model_id = ""

    batch_size = st.sidebar.number_input("Batch size для энкодинга", min_value=8, max_value=1024, value=64, step=8)

    # ===================== Сайдбар: Детектор =====================
    st.sidebar.header("Детектор неочевидных совпадений")
    enable_detector = st.sidebar.checkbox("Включить детектор (high sem, low lex)", value=True)
    semantic_threshold = st.sidebar.slider("Порог семантической схожести (>=)", 0.0, 1.0, 0.80, 0.01)
    lexical_threshold = st.sidebar.slider("Порог лексической похожести (<=)", 0.0, 1.0, 0.30, 0.01)
    low_score_threshold = st.sidebar.slider("Порог низкой семантической схожести", 0.0, 1.0, 0.75, 0.01)

    # ===================== Загрузка моделей =====================
    try:
        with st.spinner("Загружаю основную модель..."):
            model_a = load_model_from_source(model_source, model_id)
        st.sidebar.success("Основная модель загружена")
    except Exception as e:
        st.sidebar.error(f"Не удалось загрузить основную модель: {e}")
        st.stop()

    model_b = None
    if enable_ab_test:
        if ab_model_id.strip() == "":
            st.sidebar.warning("Введите ID второй модели")
        else:
            try:
                with st.spinner("Загружаю модель B..."):
                    model_b = load_model_from_source(ab_model_source, ab_model_id)
                st.sidebar.success("Модель B загружена")
            except Exception as e:
                st.sidebar.error(f"Не удалось загрузить модель B: {e}")
                st.stop()

    # ===================== Состояния =====================
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "suggestions" not in st.session_state:
        st.session_state["suggestions"] = []

    def add_to_history(record: dict):
        st.session_state["history"].append(record)
        # ограничиваем размер истории
        if len(st.session_state["history"]) > HISTORY_MAX:
            st.session_state["history"] = st.session_state["history"][-HISTORY_MAX:]

    def clear_history():
        st.session_state["history"] = []

    def add_suggestions(phrases: List[str]):
        s = [p for p in phrases if p and isinstance(p, str)]
        for p in reversed(s):
            if p not in st.session_state["suggestions"]:
                st.session_state["suggestions"].insert(0, p)
        st.session_state["suggestions"] = st.session_state["suggestions"][:200]

    # ===================== История в сайдбаре =====================
    st.sidebar.header("История проверок")
    if st.sidebar.button("Очистить историю"):
        clear_history()

    if st.session_state["history"]:
        history_bytes = json.dumps(st.session_state["history"], indent=2, ensure_ascii=False).encode('utf-8')
        st.sidebar.download_button("Скачать историю (JSON)", data=history_bytes, file_name="history.json", mime="application/json")
    else:
        st.sidebar.caption("История пустая")

    # ===================== Управление режимом c подтверждением =====================
    if "mode" not in st.session_state:
        st.session_state.mode = "Файл (CSV/XLSX/JSON)"
    if "pending_mode" not in st.session_state:
        st.session_state.pending_mode = None
    if "pending_confirm" not in st.session_state:
        st.session_state.pending_confirm = False
    if "mode_ui_v" not in st.session_state:
        st.session_state.mode_ui_v = 0

    radio_key = f"mode_selector_{st.session_state.mode}_{st.session_state.mode_ui_v}"
    mode_choice = st.radio(
        "Режим проверки",
        ["Файл (CSV/XLSX/JSON)", "Ручной ввод"],
        index=0 if st.session_state.mode == "Файл (CSV/XLSX/JSON)" else 1,
        horizontal=True,
        key=radio_key
    )

    if st.session_state.pending_mode is None and mode_choice != st.session_state.mode:
        st.session_state.pending_mode = mode_choice
        st.session_state.pending_confirm = False

    if st.session_state.pending_mode:
        col_warn, col_yes, col_close = st.columns([4, 1, 0.6])
        with col_warn:
            st.warning(
                f"Перейти в режим **{st.session_state.pending_mode}**? "
                "Текущие данные будут удалены."
            )
        with col_yes:
            if st.button("✅ Да"):
                if not st.session_state.pending_confirm:
                    st.session_state.pending_confirm = True
                    st.info("Нажмите ✅ ещё раз для подтверждения")
                else:
                    st.session_state.mode = st.session_state.pending_mode
                    st.session_state.pending_mode = None
                    st.session_state.pending_confirm = False
                    for k in ["uploaded_file", "manual_input"]:
                        st.session_state.pop(k, None)
                    st.rerun()
        with col_close:
            if st.button("❌", help="Отмена"):
                st.session_state.pending_mode = None
                st.session_state.pending_confirm = False
                st.session_state.mode_ui_v += 1

    mode_text = st.session_state.mode

    # ===================== Ручной ввод =====================
    def _set_manual_value(key: str, val: str):
        st.session_state[key] = val

    if mode_text == "Ручной ввод":
        st.header("Ручной ввод пар фраз")
        with st.expander("Проверить одну пару фраз (быстро)"):
            if "manual_text1" not in st.session_state:
                st.session_state["manual_text1"] = ""
            if "manual_text2" not in st.session_state:
                st.session_state["manual_text2"] = ""
            text1 = st.text_input("Фраза 1", key="manual_text1")
            text2 = st.text_input("Фраза 2", key="manual_text2")

            if st.button("Проверить пару", key="manual_check"):
                if not text1 or not text2:
                    st.warning("Введите обе фразы.")
                else:
                    t1 = preprocess_text(text1); t2 = preprocess_text(text2)
                    add_suggestions([t1, t2])
                    emb1 = encode_texts_in_batches(model_a, [t1], batch_size)
                    emb2 = encode_texts_in_batches(model_a, [t2], batch_size)
                    score_a = float(util.cos_sim(emb1[0], emb2[0]).item())
                    lex = jaccard_tokens(t1, t2)

                    st.subheader("Результат (модель A)")
                    col1, col2, col3 = st.columns([1,1,1])
                    col1.metric("Score A", f"{score_a:.4f}")
                    col2.metric("Jaccard (lexical)", f"{lex:.4f}")

                    is_suspicious_single = False
                    if enable_detector and (score_a >= semantic_threshold) and (lex <= lexical_threshold):
                        is_suspicious_single = True
                        st.warning("Обнаружено НЕОЧЕВИДНОЕ совпадение: высокая семантическая схожесть, низкая лексическая похожесть.")

                    if model_b is not None:
                        emb1b = encode_texts_in_batches(model_b, [t1], batch_size)
                        emb2b = encode_texts_in_batches(model_b, [t2], batch_size)
                        score_b = float(util.cos_sim(emb1b[0], emb2b[0]).item())
                        delta = score_b - score_a
                        col3.metric("Score B", f"{score_b:.4f}", delta=f"{delta:+.4f}")
                        comp_df = pd.DataFrame({"model": ["A","B"], "score":[score_a, score_b]})
                        chart = alt.Chart(comp_df).mark_bar().encode(
                            x=alt.X('model:N', title=None),
                            y=alt.Y('score:Q', scale=alt.Scale(domain=[0,1]), title="Cosine similarity score"),
                            tooltip=['model','score']
                        )
                        st.altair_chart(chart.properties(width=300), use_container_width=False)
                    else:
                        col3.write("")

                    if st.button("Сохранить результат в историю", key="save_manual_single"):
                        rec = {
                            "source": "manual_single",
                            "pair": {"phrase_1": t1, "phrase_2": t2},
                            "score": score_a,
                            "score_b": float(score_b) if (model_b is not None) else None,
                            "lexical_score": lex,
                            "is_suspicious": is_suspicious_single,
                            "model_a": model_id,
                            "model_b": ab_model_id if enable_ab_test else None,
                            "timestamp": pd.Timestamp.now().isoformat()
                        }
                        add_to_history(rec)
                        st.success("Сохранено в истории.")

        with st.expander("Ввести несколько пар (каждая пара на новой строке). Формат: `фраза1 || фраза2` / TAB / `,`"):
            bulk_text = st.text_area("Вставьте пары (по одной в строке)", height=180, key="bulk_pairs")
            st.caption("Если разделитель встречается в тексте — используйте `||`.")
            if st.button("Проверить все пары (ручной ввод)", key="manual_bulk_check"):
                lines = [l.strip() for l in bulk_text.splitlines() if l.strip()]
                if not lines:
                    st.warning("Ничего не введено.")
                else:
                    parsed = []
                    for ln in lines:
                        if "||" in ln:
                            p1, p2 = ln.split("||", 1)
                        elif "\t" in ln:
                            p1, p2 = ln.split("\t", 1)
                        elif "," in ln:
                            p1, p2 = ln.split(",", 1)
                        else:
                            p1, p2 = ln, ""
                        parsed.append((preprocess_text(p1), preprocess_text(p2)))
                    parsed = [(a,b) for a,b in parsed if a and b]
                    if not parsed:
                        st.warning("Нет корректных пар.")
                    else:
                        add_suggestions([p for pair in parsed for p in pair])
                        phrases_all = list({p for pair in parsed for p in pair})
                        phrase2idx = {p:i for i,p in enumerate(phrases_all)}
                        with st.spinner("Кодирую фразы моделью A..."):
                            embeddings_a = encode_texts_in_batches(model_a, phrases_all, batch_size)
                        embeddings_b = None
                        if model_b is not None:
                            with st.spinner("Кодирую фразы моделью B..."):
                                embeddings_b = encode_texts_in_batches(model_b, phrases_all, batch_size)
                        rows = []
                        for p1, p2 in parsed:
                            emb1 = embeddings_a[phrase2idx[p1]]
                            emb2 = embeddings_a[phrase2idx[p2]]
                            score_a = float(util.cos_sim(emb1, emb2).item())
                            score_b = None
                            if embeddings_b is not None:
                                emb1b = embeddings_b[phrase2idx[p1]]
                                emb2b = embeddings_b[phrase2idx[p2]]
                                score_b = float(util.cos_sim(emb1b, emb2b).item())
                            lex = jaccard_tokens(p1, p2)
                            rows.append({"phrase_1": p1, "phrase_2": p2, "score": score_a, "score_b": score_b, "lexical_score": lex})
                        res_df = pd.DataFrame(rows)
                        st.subheader("Результаты (ручной массовый ввод)")
                        styled = style_suspicious_and_low(res_df, semantic_threshold, lexical_threshold, low_score_threshold)
                        st.dataframe(styled, use_container_width=True)
                        csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Скачать результаты CSV", data=csv_bytes, file_name="manual_results.csv", mime="text/csv")

                        if enable_detector:
                            susp_df = res_df[(res_df["score"] >= semantic_threshold) & (res_df["lexical_score"] <= lexical_threshold)]
                            if not susp_df.empty:
                                st.markdown("### Неочевидные совпадения (high semantic, low lexical)")
                                st.write(f"Найдено {len(susp_df)} пар.")
                                st.dataframe(susp_df, use_container_width=True)
                                susp_csv = susp_df.to_csv(index=False).encode('utf-8')
                                st.download_button("Скачать suspicious CSV", data=susp_csv, file_name="suspicious_manual_bulk.csv", mime="text/csv")
                                if st.button("Сохранить suspicious в историю", key="save_susp_manual"):
                                    rec = {
                                        "source": "manual_bulk_suspicious",
                                        "pairs_count": len(susp_df),
                                        "results": susp_df.to_dict(orient="records"),
                                        "model_a": model_id,
                                        "model_b": ab_model_id if enable_ab_test else None,
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "semantic_threshold": semantic_threshold,
                                        "lexical_threshold": lexical_threshold
                                    }
                                    add_to_history(rec)
                                    st.success("Сохранено в истории.")

    # ===================== Блок: файл =====================
    if mode_text == "Файл (CSV/XLSX/JSON)":
        st.header("1. Загрузите CSV, Excel или JSON с колонками: phrase_1, phrase_2, topics (опционально)")
        uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx", "xls", "json", "ndjson"])

        if uploaded_file is not None:
            try:
                from utils import read_uploaded_file_bytes  # локальный импорт, чтобы не утяжелять верх
                df, file_hash = read_uploaded_file_bytes(uploaded_file)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
                st.stop()

            required_cols = {"phrase_1", "phrase_2"}
            if not required_cols.issubset(set(df.columns)):
                st.error(f"Файл должен содержать колонки: {required_cols}")
                st.stop()

            # --- Редактор датасета
            st.subheader("✏️ Редактирование датасета перед проверкой")
            st.caption("Можно изменять, добавлять и удалять строки. Изменения временные (только в этой сессии).")
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="dataset_editor")
            edited_csv = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Скачать обновлённый датасет (CSV)", data=edited_csv, file_name="edited_dataset.csv", mime="text/csv")
            df = edited_df.copy()

            # --- Препроцессинг
            df["phrase_1"] = df["phrase_1"].map(preprocess_text)
            df["phrase_2"] = df["phrase_2"].map(preprocess_text)
            if "topics" in df.columns:
                df["topics_list"] = df["topics"].map(parse_topics_field)
            else:
                df["topics_list"] = [[] for _ in range(len(df))]

            # Признаки по каждой фразе (простые флаги)
            for col in ["phrase_1", "phrase_2"]:
                flags = df[col].map(simple_flags)
                df[f"{col}_len_tok"] = flags.map(lambda d: d["len_tok"])
                df[f"{col}_len_char"] = flags.map(lambda d: d["len_char"])
                df[f"{col}_has_neg"] = flags.map(lambda d: d["has_neg"])
                df[f"{col}_has_num"] = flags.map(lambda d: d["has_num"])
                df[f"{col}_has_date"] = flags.map(lambda d: d["has_date"])
                if _MORPH is not None:
                    df[f"{col}_pos1"] = df[col].map(pos_first_token)
                else:
                    df[f"{col}_pos1"] = "NA"

            add_suggestions(list(set(df["phrase_1"].tolist() + df["phrase_2"].tolist())))

            # --- Энкодинг
            phrases_all = list(set(df["phrase_1"].tolist() + df["phrase_2"].tolist()))
            phrase2idx = {p: i for i, p in enumerate(phrases_all)}
            with st.spinner("Кодирую фразы моделью A..."):
                embeddings_a = encode_texts_in_batches(model_a, phrases_all, batch_size)
            embeddings_b = None
            if enable_ab_test and model_b is not None:
                with st.spinner("Кодирую фразы моделью B..."):
                    embeddings_b = encode_texts_in_batches(model_b, phrases_all, batch_size)

            # --- Счёт метрик на парах
            scores, scores_b, lexical_scores = [], [], []
            for _, row in df.iterrows():
                p1, p2 = row["phrase_1"], row["phrase_2"]
                emb1_a, emb2_a = embeddings_a[phrase2idx[p1]], embeddings_a[phrase2idx[p2]]
                score_a = float(util.cos_sim(emb1_a, emb2_a).item())
                scores.append(score_a)
                if embeddings_b is not None:
                    emb1_b, emb2_b = embeddings_b[phrase2idx[p1]], embeddings_b[phrase2idx[p2]]
                    scores_b.append(float(util.cos_sim(emb1_b, emb2_b).item()))
                lex_score = jaccard_tokens(p1, p2)
                lexical_scores.append(lex_score)

            df["score"] = scores
            if embeddings_b is not None:
                df["score_b"] = scores_b
            df["lexical_score"] = lexical_scores

            # --- Панели аналитики (вкладки)
            st.subheader("2. Аналитика")
            tabs = st.tabs(["Сводка", "Разведка (Explore)", "Срезы (Slices)", "A/B тест", "Экспорт"])

            # = Svodka =
            with tabs[0]:
                total = len(df)
                low_cnt = int((df["score"] < low_score_threshold).sum())
                susp_cnt = int(((df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)).sum())
                colA, colB, colC, colD = st.columns(4)
                colA.metric("Размер датасета", f"{total}")
                colB.metric("Средний score", f"{df['score'].mean():.4f}")
                colC.metric("Медиана score", f"{df['score'].median():.4f}")
                colD.metric(f"Низкие (<{low_score_threshold:.2f})", f"{low_cnt} ({(low_cnt / max(total,1)):.0%})")
                st.caption(f"Неочевидные совпадения (high-sem/low-lex): {susp_cnt} ({(susp_cnt / max(total,1)):.0%})")

            # = Explore =
            with tabs[1]:
                st.markdown("#### Распределения и взаимосвязи")
                left, right = st.columns(2)
                with left:
                    chart = alt.Chart(pd.DataFrame({"score": df["score"]})).mark_bar().encode(
                        alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Cosine similarity score"),
                        y='count()', tooltip=['count()']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                with right:
                    chart_lex = alt.Chart(pd.DataFrame({"lexical_score": df["lexical_score"]})).mark_bar().encode(
                        alt.X("lexical_score:Q", bin=alt.Bin(maxbins=30), title="Jaccard (лексика)"),
                        y='count()', tooltip=['count()']
                    ).interactive()
                    st.altair_chart(chart_lex, use_container_width=True)

                st.markdown("##### Точечный график: семантика vs лексика")
                scatter_df = df[["score","lexical_score"]].copy()
                sc = alt.Chart(scatter_df).mark_point(opacity=0.6).encode(
                    x=alt.X("lexical_score:Q", title="Jaccard (лексика)"),
                    y=alt.Y("score:Q", title="Cosine similarity (семантика)", scale=alt.Scale(domain=[0,1])),
                    tooltip=["score","lexical_score"]
                ).interactive()
                st.altair_chart(sc, use_container_width=True)

                if enable_detector:
                    st.markdown("##### Неочевидные совпадения")
                    susp_df = df[(df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)]
                    if susp_df.empty:
                        st.info("Не найдено пар под текущие пороги.")
                    else:
                        st.write(f"Пар: {len(susp_df)}")
                        st.dataframe(susp_df[["phrase_1","phrase_2","score","lexical_score"]], use_container_width=True)

            # = Slices =
            with tabs[2]:
                st.markdown("#### Срезы качества")
                # простые флаги
                df["_any_neg"] = df["phrase_1_has_neg"] | df["phrase_2_has_neg"]
                df["_any_num"] = df["phrase_1_has_num"] | df["phrase_2_has_num"]
                df["_any_date"] = df["phrase_1_has_date"] | df["phrase_2_has_date"]
                # длина (по сумме токенов обеих фраз)
                def _len_bucket(r):
                    n = int(r["phrase_1_len_tok"] + r["phrase_2_len_tok"])
                    if n <= 4: return "[0,4]"
                    if n <= 9: return "[5,9]"
                    if n <= 19: return "[10,19]"
                    return "[20,+)"
                df["_len_bucket"] = df.apply(_len_bucket, axis=1)

                cols1 = st.columns(3)
                with cols1[0]:
                    st.markdown("**По длине**")
                    agg_len = df.groupby("_len_bucket")["score"].agg(["count","mean","median"]).reset_index().sort_values("_len_bucket")
                    st.dataframe(agg_len, use_container_width=True)
                with cols1[1]:
                    st.markdown("**Отрицания/Числа/Даты**")
                    flags_view = []
                    for flag in ["_any_neg","_any_num","_any_date"]:
                        sub = df[df[flag]]
                        flags_view.append({"флаг":flag, "count":len(sub), "mean":float(sub["score"].mean()) if len(sub)>0 else np.nan})
                    st.dataframe(pd.DataFrame(flags_view), use_container_width=True)
                with cols1[2]:
                    if _MORPH is None:
                        st.info("Морфология (POS) недоступна: не установлен pymorphy2")
                    else:
                        st.markdown("**POS первого токена**")
                        pos_agg = df.groupby("phrase_1_pos1")["score"].agg(["count","mean"]).reset_index().rename(columns={"phrase_1_pos1":"POS"})
                        st.dataframe(pos_agg.sort_values("count", ascending=False), use_container_width=True)

                topic_mode = st.checkbox("Агрегация по topics", value=("topics_list" in df.columns))
                if topic_mode:
                    st.markdown("**По темам (topics)**")
                    exploded = df.explode("topics_list")
                    exploded["topics_list"] = exploded["topics_list"].fillna("")
                    exploded = exploded[exploded["topics_list"].astype(str)!=""]
                    if exploded.empty:
                        st.info("В датасете нет непустых topics.")
                    else:
                        top_agg = exploded.groupby("topics_list")["score"].agg(["count","mean","median"]).reset_index().sort_values("count", ascending=False)
                        st.dataframe(top_agg, use_container_width=True)

            # = AB test =
            with tabs[3]:
                if (not enable_ab_test) or ("score_b" not in df.columns):
                    st.info("A/B тест отключён или нет столбца score_b.")
                else:
                    st.markdown("#### Сравнение моделей A vs B")
                    colx, coly, colz = st.columns(3)
                    colx.metric("Средний A", f"{df['score'].mean():.4f}")
                    coly.metric("Средний B", f"{df['score_b'].mean():.4f}")
                    colz.metric("Δ (B - A)", f"{(df['score_b'].mean()-df['score'].mean()):+.4f}")

                    n_boot = st.slider("Бутстрэп итераций", 200, 2000, 500, 100)
                    mean_diff, low, high = bootstrap_diff_ci(df["score_b"].to_numpy(), df["score"].to_numpy(), n_boot=n_boot)
                    st.write(f"ДИ (95%) для Δ (B−A): **[{low:+.4f}, {high:+.4f}]**, средняя разница: **{mean_diff:+.4f}**")
                    ab_df = pd.DataFrame({"A": df["score"], "B": df["score_b"]})
                    ab_chart = alt.Chart(ab_df.reset_index()).mark_point(opacity=0.5).encode(
                        x=alt.X("A:Q", scale=alt.Scale(domain=[0,1])),
                        y=alt.Y("B:Q", scale=alt.Scale(domain=[0,1])),
                        tooltip=["A","B"]
                    ).interactive()
                    st.altair_chart(ab_chart, use_container_width=True)

                    delta_df = df.copy()
                    delta_df["delta"] = delta_df["score_b"] - delta_df["score"]
                    st.markdown("**Топ, где B ≫ A**")
                    st.dataframe(
                        delta_df.sort_values("delta", ascending=False).head(10)[["phrase_1","phrase_2","score","score_b","delta"]],
                        use_container_width=True
                    )
                    st.markdown("**Топ, где A ≫ B**")
                    st.dataframe(
                        delta_df.sort_values("delta", ascending=True).head(10)[["phrase_1","phrase_2","score","score_b","delta"]],
                        use_container_width=True
                    )

            # = Export =
            with tabs[4]:
                st.markdown("#### Экспорт отчёта (JSON)")
                report = {
                    "file_name": uploaded_file.name,
                    "file_hash": file_hash,
                    "n_pairs": int(len(df)),
                    "model_a": model_id,
                    "model_b": ab_model_id if enable_ab_test else None,
                    "thresholds": {
                        "semantic_threshold": float(semantic_threshold),
                        "lexical_threshold": float(lexical_threshold),
                        "low_score_threshold": float(low_score_threshold)
                    },
                    "summary": {
                        "mean_score": float(df["score"].mean()),
                        "median_score": float(df["score"].median()),
                        "low_count": int((df["score"] < low_score_threshold).sum()),
                        "suspicious_count": int(((df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)).sum())
                    }
                }
                rep_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button("💾 Скачать отчёт JSON", data=rep_bytes, file_name="synonym_checker_report.json", mime="application/json")

            # --- Выгрузка + подсветка
            st.subheader("3. Результаты и выгрузка")
            result_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Скачать результаты CSV", data=result_csv, file_name="results.csv", mime="text/csv")
            styled_df = style_suspicious_and_low(df, semantic_threshold, lexical_threshold, low_score_threshold)
            st.dataframe(styled_df, use_container_width=True)

            # --- Suspicious блок и история
            if enable_detector:
                susp_df = df[(df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)]
                st.markdown("### Неочевидные совпадения (high semantic, low lexical)")
                if susp_df.empty:
                    st.write("Не найдено.")
                else:
                    st.write(f"Найдено {len(susp_df)} пар.")
                    st.dataframe(susp_df, use_container_width=True)
                    susp_csv = susp_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Скачать suspicious CSV", data=susp_csv, file_name="suspicious_file_mode.csv", mime="text/csv")
                    if st.button("Сохранить suspicious в историю", key="save_susp_file"):
                        rec = {
                            "source": "file_suspicious",
                            "file_hash": file_hash,
                            "file_name": uploaded_file.name,
                            "pairs_count": len(susp_df),
                            "results": susp_df.to_dict(orient="records"),
                            "model_a": model_id,
                            "model_b": ab_model_id if enable_ab_test else None,
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "semantic_threshold": semantic_threshold,
                            "lexical_threshold": lexical_threshold
                        }
                        add_to_history(rec)
                        st.success("Сохранено в истории.")
        else:
            st.info("Загрузите файл для начала проверки.")

    # ===================== История внизу =====================
    if st.session_state["history"]:
        st.header("История проверок")
        for idx, rec in enumerate(reversed(st.session_state["history"])):
            st.markdown(f"### Проверка #{len(st.session_state['history']) - idx}")
            if rec.get("source") == "manual_single":
                p = rec.get("pair", {})
                st.markdown(f"**Ручной ввод (single)**  |  **Дата:** {rec.get('timestamp','-')}")
                st.markdown(f"**Фразы:** `{p.get('phrase_1','')}`  — `{p.get('phrase_2','')}`")
                st.markdown(f"**Score A:** {rec.get('score', '-')}, **Score B:** {rec.get('score_b', '-')}, **Lexical:** {rec.get('lexical_score','-')}")
                if rec.get("is_suspicious"):
                    st.warning("Эта пара помечена как неочевидное совпадение (high semantic, low lexical).")
            elif rec.get("source") == "manual_bulk":
                st.markdown(f"**Ручной ввод (bulk)**  |  **Дата:** {rec.get('timestamp','-')}")
                st.markdown(f"**Пар:** {rec.get("pairs_count", 0)}  |  **Модель A:** {rec.get('model_a','-')}")
                saved_df = pd.DataFrame(rec.get("results", []))
                if not saved_df.empty:
                    styled_hist_df = style_suspicious_and_low(saved_df, rec.get("semantic_threshold", 0.8), rec.get("lexical_threshold", 0.3), 0.75)
                    st.dataframe(styled_hist_df, use_container_width=True)
            elif rec.get("source") in ("manual_bulk_suspicious",):
                st.markdown(f"**Ручной suspicious**  |  **Дата:** {rec.get('timestamp','-')}")
                st.markdown(f"**Пар:** {rec.get("pairs_count", 0)}  |  **Модель A:** {rec.get('model_a','-')}")
                saved_df = pd.DataFrame(rec.get("results", []))
                if not saved_df.empty:
                    st.dataframe(saved_df, use_container_width=True)
            elif rec.get("source") == "file_suspicious":
                st.markdown(f"**Файл (suspicious)**  |  **Файл:** {rec.get('file_name','-')}  |  **Дата:** {rec.get('timestamp','-')}")
                st.markdown(f"**Пар:** {rec.get("pairs_count", 0)}  |  **Модель A:** {rec.get('model_a','-')}")
                saved_df = pd.DataFrame(rec.get("results", []))
                if not saved_df.empty:
                    st.dataframe(saved_df, use_container_width=True)
            else:
                st.markdown(f"**Файл:** {rec.get('file_name','-')}  |  **Дата:** {rec.get('timestamp','-')}")
                st.markdown(f"**Модель A:** {rec.get('model_a','-')}  |  **Модель B:** {rec.get('model_b','-')}")
                saved_df = pd.DataFrame(rec.get("results", []))
                if not saved_df.empty:
                    styled_hist_df = style_suspicious_and_low(saved_df, 0.8, 0.3, 0.75)
                    st.dataframe(styled_hist_df, use_container_width=True)
            st.markdown("---")

# ===================== РЕЖИМ: МУЛЬТИМОДАЛЬНЫЕ МОДЕЛИ =====================
elif mode == "Работа с мультимодальными моделями":
    st.header("🖼️ Мультимодальные сценарии")

    # ===================== История мультимодала =====================
    if "mm_history" not in st.session_state:
        st.session_state["mm_history"] = []

    def add_mm_history(record: dict):
        st.session_state["mm_history"].append(record)
        if len(st.session_state["mm_history"]) > 300:
            st.session_state["mm_history"] = st.session_state["mm_history"][-300:]

    st.sidebar.header("История (мультимодальные)")
    if st.sidebar.button("Очистить историю мультимодала"):
        st.session_state["mm_history"] = []
    if st.session_state["mm_history"]:
        import json

        def safe_json_dumps(obj) -> str:
            """Универсальная сериализация: всё, что не json-совместимо → str"""
            def convert(o):
                if isinstance(o, (str, int, float, bool)) or o is None:
                    return o
                if isinstance(o, dict):
                    return {k: convert(v) for k, v in o.items()}
                if isinstance(o, (list, tuple, set)):
                    return [convert(v) for v in o]
                return str(o)

            return json.dumps(convert(obj), indent=2, ensure_ascii=False)

        mm_bytes = safe_json_dumps(st.session_state["mm_history"]).encode("utf-8")

        st.sidebar.download_button(
            "Скачать историю (JSON)",
            data=mm_bytes,
            file_name="mm_history.json",
            mime="application/json"
        )
    else:
        st.sidebar.caption("История мультимодала пуста")

    # ===================== Выбор моделей =====================
    st.sidebar.header("Настройки мультимодальных моделей")

    # CLIP (A)
    clip_source = st.sidebar.selectbox("Источник CLIP (A)", ["huggingface", "google_drive"], index=0, key="clip_source_a")
    clip_id = st.sidebar.text_input("CLIP (A) Model ID / GDrive File ID", value="openai/clip-vit-base-patch32", key="clip_id_a")

    # BLIP
    blip_source = st.sidebar.selectbox("Источник BLIP", ["huggingface", "google_drive"], index=0, key="blip_source")
    blip_id = st.sidebar.text_input("BLIP Model ID / GDrive File ID", value="Salesforce/blip-image-captioning-base", key="blip_id")

    # A/B тест для CLIP
    enable_mm_ab = st.sidebar.checkbox("A/B тест: вторая CLIP модель (B)", value=False)
    if enable_mm_ab:
        clip_source_b = st.sidebar.selectbox("Источник CLIP (B)", ["huggingface", "google_drive"], index=0, key="clip_source_b")
        clip_id_b = st.sidebar.text_input("CLIP (B) Model ID / GDrive File ID", value="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", key="clip_id_b")
    else:
        clip_source_b, clip_id_b = None, None

    from multimodal import load_blip_model, load_clip_model, check_text_image_pair, generate_caption
    from utils import bootstrap_diff_ci
    from PIL import Image
    import pandas as pd
    import numpy as np
    import json
    import torch
    import io, zipfile
    import altair as alt

    try:
        with st.spinner("Загружаю CLIP (A)..."):
            clip_model_a, clip_proc_a = load_clip_model(clip_source, clip_id)
        st.sidebar.success("CLIP (A) загружена")
    except Exception as e:
        st.sidebar.error(f"Ошибка загрузки CLIP (A): {e}")
        st.stop()

    clip_model_b, clip_proc_b = None, None
    if enable_mm_ab and clip_id_b:
        try:
            with st.spinner("Загружаю CLIP (B)..."):
                clip_model_b, clip_proc_b = load_clip_model(clip_source_b, clip_id_b)
            st.sidebar.success("CLIP (B) загружена")
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки CLIP (B): {e}")
            st.stop()

    try:
        with st.spinner("Загружаю BLIP..."):
            blip_model_u, blip_proc_u = load_blip_model(blip_source, blip_id)
        st.sidebar.success("BLIP загружена")
    except Exception as e:
        st.sidebar.error(f"Ошибка загрузки BLIP: {e}")
        st.stop()

    # ===================== BLIP Caption =====================
    with st.expander("🖼️ Генерация описания картинки (BLIP)"):
        st.markdown("Загрузите изображение, и BLIP предложит текстовое описание.")

        uploaded_blip = st.file_uploader("Загрузите картинку для подписи", type=["jpg","jpeg","png"], key="blip_img")

        if uploaded_blip:
            img = Image.open(uploaded_blip).convert("RGB")
            st.image(img, caption="Загруженное изображение", use_column_width=True)

            if st.button("Сгенерировать описание", key="blip_caption_btn"):
                with st.spinner("Генерация описания..."):
                    caption = generate_caption(blip_model_u, blip_proc_u, img)
                st.success(f"BLIP Caption: **{caption}**")
                add_mm_history({
                    "type": "blip_caption",
                    "filename": getattr(uploaded_blip, "name", ""),
                    "caption": caption,
                    "timestamp": pd.Timestamp.now().isoformat()
                })

    # ===================== CLIP + BLIP сравнение =====================
    with st.expander("🤝 Сценарий: сравнение текста и BLIP-описания"):
        st.markdown("Загрузите картинку, получите описание через BLIP и сравните его с вашим текстом при помощи CLIP.")

        uploaded_joint = st.file_uploader("Загрузите картинку", type=["jpg","jpeg","png"], key="joint_img")
        user_text = st.text_input("Введите свой текст для сравнения", key="joint_text")

        if uploaded_joint:
            img = Image.open(uploaded_joint).convert("RGB")
            st.image(img, caption="Загруженное изображение", use_column_width=True)

            if st.button("Сравнить текст и BLIP-описание", key="joint_btn"):
                with st.spinner("Генерация описания BLIP..."):
                    blip_caption = generate_caption(blip_model_u, blip_proc_u, img)

                with st.spinner("Сравнение через CLIP..."):
                    score_user = check_text_image_pair(clip_model_a, clip_proc_a, user_text, img) if user_text else None
                    score_blip = check_text_image_pair(clip_model_a, clip_proc_a, blip_caption, img)

                st.subheader("Результаты")
                st.markdown(f"**BLIP Caption:** {blip_caption}")
                st.metric("CLIP score (BLIP caption ↔ image)", f"{score_blip:.4f}")

                rec = {
                    "type": "clip_compare",
                    "filename": getattr(uploaded_joint, "name", ""),
                    "blip_caption": blip_caption,
                    "score_blip": float(score_blip),
                    "user_text": user_text if user_text else None,
                    "score_user": float(score_user) if score_user is not None else None,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "clip_model_a": clip_id,
                    "blip_model": blip_id
                }

                if score_user is not None:
                    delta = score_user - score_blip
                    st.metric("CLIP score (ваш текст ↔ image)", f"{score_user:.4f}", delta=f"{delta:+.4f}")
                    if score_user > score_blip:
                        st.success("✅ Ваш текст лучше соответствует картинке, чем автогенерация BLIP!")
                    else:
                        st.info("ℹ️ BLIP-описание оказалось ближе к картинке, чем ваш текст.")

                add_mm_history(rec)

    # ===================== Пакетный анализ: CLIP Retrieval =====================
    def _recall_at_k(sim_matrix: np.ndarray, k: int) -> float:
        n = sim_matrix.shape[0]
        ranks = np.argsort(-sim_matrix, axis=1)
        hits = sum([1 if i in ranks[i, :k] else 0 for i in range(n)])
        return hits / max(n, 1)

    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return a @ b.T

    with st.expander("📦 Пакетный анализ: CLIP Retrieval (text↔image)"):
        st.markdown("Загрузите CSV (`text,image`) и ZIP с файлами изображений. Для каждой строки CSV `image` должен совпадать с именем файла в ZIP.")

        csv_file = st.file_uploader("CSV с парами (text,image)", type=["csv"], key="mm_clip_csv")
        zip_file = st.file_uploader("ZIP с изображениями", type=["zip"], key="mm_clip_zip")

        if csv_file and zip_file:
            df_pairs = pd.read_csv(csv_file)
            if not {"text", "image"}.issubset(df_pairs.columns):
                st.error("CSV должен содержать колонки: text, image")
            else:
                zbytes = io.BytesIO(zip_file.read())
                with zipfile.ZipFile(zbytes) as zf:
                    names = set(zf.namelist())

                    imgs, ok_rows = [], []
                    for idx, row in df_pairs.iterrows():
                        fname = str(row["image"])
                        if fname in names:
                            with zf.open(fname) as f:
                                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                            imgs.append(img)
                            ok_rows.append(idx)
                        else:
                            st.warning(f"Нет файла в ZIP: {fname}")

                    if imgs:
                        df_eval = df_pairs.loc[ok_rows].reset_index(drop=True)
                        texts = df_eval["text"].astype(str).tolist()

                        with st.spinner("Кодирую тексты (CLIP A)..."):
                            inputs_t = clip_proc_a(text=texts, return_tensors="pt", padding=True, truncation=True)
                            with torch.no_grad():
                                t_emb_a = clip_model_a.get_text_features(**inputs_t).cpu().numpy()

                        with st.spinner("Кодирую изображения (CLIP A)..."):
                            inputs_i = clip_proc_a(images=imgs, return_tensors="pt")
                            with torch.no_grad():
                                i_emb_a = clip_model_a.get_image_features(**inputs_i).cpu().numpy()

                        sim_a = _cosine_sim(t_emb_a, i_emb_a)

                        r1 = _recall_at_k(sim_a, 1)
                        r5 = _recall_at_k(sim_a, 5) if sim_a.shape[1] >= 5 else np.nan
                        r10 = _recall_at_k(sim_a, 10) if sim_a.shape[1] >= 10 else np.nan

                        st.subheader("Метрики (CLIP A)")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Recall@1", f"{r1:.3f}")
                        c2.metric("Recall@5", f"{r5:.3f}" if not np.isnan(r5) else "—")
                        c3.metric("Recall@10", f"{r10:.3f}" if not np.isnan(r10) else "—")

                        best_scores = sim_a.max(axis=1)
                        chart = alt.Chart(pd.DataFrame({"best_score": best_scores})).mark_bar().encode(
                            alt.X("best_score:Q", bin=alt.Bin(maxbins=30), title="Лучший score по строке (text→image)"),
                            y='count()', tooltip=['count()']
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)

                        # A/B сравнение
                        if clip_model_b is not None:
                            with st.spinner("Кодирую тексты (CLIP B)..."):
                                inputs_t_b = clip_proc_b(text=texts, return_tensors="pt", padding=True, truncation=True)
                                with torch.no_grad():
                                    t_emb_b = clip_model_b.get_text_features(**inputs_t_b).cpu().numpy()
                            with st.spinner("Кодирую изображения (CLIP B)..."):
                                inputs_i_b = clip_proc_b(images=imgs, return_tensors="pt")
                                with torch.no_grad():
                                    i_emb_b = clip_model_b.get_image_features(**inputs_i_b).cpu().numpy()
                            sim_b = _cosine_sim(t_emb_b, i_emb_b)

                            r1_b = _recall_at_k(sim_b, 1)
                            r5_b = _recall_at_k(sim_b, 5) if sim_b.shape[1] >= 5 else np.nan
                            r10_b = _recall_at_k(sim_b, 10) if sim_b.shape[1] >= 10 else np.nan

                            st.subheader("A/B сравнение (B против A)")
                            d1, d2, d3 = st.columns(3)
                            d1.metric("Δ Recall@1", f"{(r1_b - r1):+.3f}")
                            d2.metric("Δ Recall@5", f"{(r5_b - r5):+.3f}" if not (np.isnan(r5) or np.isnan(r5_b)) else "—")
                            d3.metric("Δ Recall@10", f"{(r10_b - r10):+.3f}" if not (np.isnan(r10) or np.isnan(r10_b)) else "—")

                            mean_diff, low, high = bootstrap_diff_ci(sim_b.max(axis=1), sim_a.max(axis=1), n_boot=500)
                            st.caption(f"Бутстрэп ДИ (95%) для Δ best_score (B−A): [{low:+.4f}, {high:+.4f}], средняя: {mean_diff:+.4f}")

    # ===================== Анализ BLIP Caption (BLEU / ROUGE) =====================
    def _ngrams(tokens: list, n: int) -> set:
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def _bleu(ref: str, hyp: str, n: int = 4) -> float:
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        score = 0.0
        for i in range(1, n+1):
            ref_ngr = _ngrams(ref_tokens, i)
            hyp_ngr = _ngrams(hyp_tokens, i)
            inter = len(ref_ngr & hyp_ngr)
            total = max(len(hyp_ngr), 1)
            score += inter / total
        return score / n

    def _rouge_l(ref: str, hyp: str) -> float:
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        dp = [[0]*(len(hyp_tokens)+1) for _ in range(len(ref_tokens)+1)]
        for i in range(1, len(ref_tokens)+1):
            for j in range(1, len(hyp_tokens)+1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[-1][-1]
        return lcs / max(len(ref_tokens), 1)

    with st.expander("📊 Анализ BLIP Caption против эталонных описаний"):
        st.markdown("Загрузите CSV с колонками: `image`, `reference_caption`. ZIP с изображениями обязателен. BLIP сгенерирует подписи, а затем сравнит с эталонными BLEU и ROUGE-L.")

        csv_blip = st.file_uploader("CSV (image, reference_caption)", type=["csv"], key="blip_eval_csv")
        zip_blip = st.file_uploader("ZIP с изображениями", type=["zip"], key="blip_eval_zip")

        if csv_blip and zip_blip:
            df_ref = pd.read_csv(csv_blip)
            if not {"image", "reference_caption"}.issubset(df_ref.columns):
                st.error("CSV должен содержать колонки: image, reference_caption")
            else:
                zbytes = io.BytesIO(zip_blip.read())
                with zipfile.ZipFile(zbytes) as zf:
                    names = set(zf.namelist())
                    refs, hyps, bleus, rouges = [], [], [], []

                    for idx, row in df_ref.iterrows():
                        fname = str(row["image"])
                        ref_caption = str(row["reference_caption"])
                        if fname not in names:
                            st.warning(f"Нет файла в ZIP: {fname}")
                            continue
                        with zf.open(fname) as f:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")

                        with torch.no_grad():
                            hyp_caption = generate_caption(blip_model_u, blip_proc_u, img)

                        bleu_score = _bleu(ref_caption, hyp_caption)
                        rouge_score = _rouge_l(ref_caption, hyp_caption)

                        refs.append(ref_caption)
                        hyps.append(hyp_caption)
                        bleus.append(bleu_score)
                        rouges.append(rouge_score)

                    if bleus:
                        df_eval = pd.DataFrame({
                            "image": df_ref["image"].iloc[:len(bleus)],
                            "reference": refs,
                            "hypothesis": hyps,
                            "BLEU": bleus,
                            "ROUGE-L": rouges
                        })

                        st.subheader("Результаты BLEU / ROUGE-L")
                        st.dataframe(df_eval)

                        c1, c2 = st.columns(2)
                        c1.metric("Средний BLEU", f"{np.mean(bleus):.3f}")
                        c2.metric("Средний ROUGE-L", f"{np.mean(rouges):.3f}")

                        st.markdown("**Распределение BLEU**")
                        chart_bleu = alt.Chart(pd.DataFrame({"BLEU": bleus})).mark_bar().encode(
                            alt.X("BLEU:Q", bin=alt.Bin(maxbins=30)),
                            y='count()', tooltip=['count()']
                        ).interactive()
                        st.altair_chart(chart_bleu, use_container_width=True)

                        st.markdown("**Распределение ROUGE-L**")
                        chart_rouge = alt.Chart(pd.DataFrame({"ROUGE-L": rouges})).mark_bar().encode(
                            alt.X("ROUGE-L:Q", bin=alt.Bin(maxbins=30)),
                            y='count()', tooltip=['count()']
                        ).interactive()
                        st.altair_chart(chart_rouge, use_container_width=True)

                        mm_bytes_eval = df_eval.to_csv(index=False).encode("utf-8")
                        st.download_button("Скачать результаты", data=mm_bytes_eval, file_name="blip_eval_results.csv", mime="text/csv")
