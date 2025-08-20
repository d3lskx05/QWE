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
    # ===================== РЕЖИМ: МУЛЬТИМОДАЛЬНЫЕ МОДЕЛИ =====================
    import io, json, zipfile
    from typing import List, Optional, Dict
    import numpy as np
    import pandas as pd
    import streamlit as st
    from PIL import Image

    # Используем твой модуль загрузки моделей (без конфликтов с текстовым анализом)
    from multimodal import load_blip_model, load_clip_model, generate_caption

    # Внешние библиотеки (по возможности graceful)
    try:
        import torch
    except Exception:
        torch = None
    try:
        import altair as alt
    except Exception:
        alt = None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    try:
        from openai import OpenAI  # OpenAI SDK >=1.0
    except Exception:
        OpenAI = None

    st.header("🖼️ Продвинутая мультимодальная аналитика")

    # ===================== История =====================
    if "mm_history" not in st.session_state:
        st.session_state["mm_history"] = []

    def mm_add_history(record: Dict):
        st.session_state["mm_history"].append(record)
        if len(st.session_state["mm_history"]) > 300:
            st.session_state["mm_history"] = st.session_state["mm_history"][-300:]

    # ===================== Боковая панель: настройки =====================
    st.sidebar.header("Настройки мультимодальных моделей")

    # Источник и ID моделей (оставляем как у тебя)
    clip_source_a = st.sidebar.selectbox("Источник CLIP (A)", ["huggingface", "google_drive"], index=0, key="mm_clip_source_a")
    clip_id_a = st.sidebar.text_input("CLIP (A) Model ID", value="openai/clip-vit-base-patch32", key="mm_clip_id_a")

    enable_ab = st.sidebar.checkbox("A/B: Включить CLIP (B)", value=False, key="mm_enable_ab")
    if enable_ab:
        clip_source_b = st.sidebar.selectbox("Источник CLIP (B)", ["huggingface", "google_drive"], index=0, key="mm_clip_source_b")
        clip_id_b = st.sidebar.text_input("CLIP (B) Model ID", value="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", key="mm_clip_id_b")
    else:
        clip_source_b, clip_id_b = None, None

    blip_source = st.sidebar.selectbox("Источник BLIP", ["huggingface", "google_drive"], index=0, key="mm_blip_source")
    blip_caption_id = st.sidebar.text_input("BLIP Caption Model ID", value="Salesforce/blip-image-captioning-base", key="mm_blip_caption_id")
    blip_vqa_id = st.sidebar.text_input("BLIP VQA Model ID", value="Salesforce/blip-vqa-base", key="mm_blip_vqa_id")

    # LLM-оценка (опционально)
    use_llm_eval = st.sidebar.checkbox("LLM-оценка (OpenAI-совместимый API)", value=False, key="mm_use_llm")
    if use_llm_eval:
        llm_api_key = st.sidebar.text_input("API Key", type="password", key="mm_llm_key")
        llm_api_base = st.sidebar.text_input("API Base (опционально)", key="mm_llm_base")
        llm_model_name = st.sidebar.text_input("Модель (LLM)", value="gpt-4o-mini", key="mm_llm_model")
    else:
        llm_api_key = llm_api_base = llm_model_name = None

    # ===================== Кеш-загрузка моделей =====================
    @st.cache_resource(show_spinner=False)
    def mm_load_clip(source: str, model_id: str):
        try:
            model, proc = load_clip_model(source, model_id)
            return model, proc
        except Exception as e:
            st.warning(f"Не удалось загрузить CLIP {model_id}: {e}")
            return None, None

    @st.cache_resource(show_spinner=False)
    def mm_load_blip(source: str, model_id: str):
        try:
            model, proc = load_blip_model(source, model_id)
            return model, proc
        except Exception as e:
            st.warning(f"Не удалось загрузить BLIP {model_id}: {e}")
            return None, None

    clip_model_a, clip_proc_a = mm_load_clip(clip_source_a, clip_id_a)
    clip_model_b, clip_proc_b = (mm_load_clip(clip_source_b, clip_id_b) if (enable_ab and clip_id_b) else (None, None))
    blip_cap_model, blip_cap_proc = mm_load_blip(blip_source, blip_caption_id)
    blip_vqa_model, blip_vqa_proc = mm_load_blip(blip_source, blip_vqa_id)

    # ===================== Метрики и утилиты (префикс mm_ для изоляции) =====================
    def mm_ngrams(tokens: List[str], n: int):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def mm_bleu_n(ref: str, hyp: str, n: int = 4) -> float:
        ref_t, hyp_t = ref.lower().split(), hyp.lower().split()
        if not hyp_t:
            return 0.0
        score = 0.0
        for k in range(1, n+1):
            ref_ngr, hyp_ngr = mm_ngrams(ref_t, k), mm_ngrams(hyp_t, k)
            denom = max(len(hyp_ngr), 1)
            score += len(ref_ngr & hyp_ngr) / denom
        return score / n

    def mm_rouge_l(ref: str, hyp: str) -> float:
        ref_t, hyp_t = ref.lower().split(), hyp.lower().split()
        dp = [[0]*(len(hyp_t)+1) for _ in range(len(ref_t)+1)]
        for i in range(1, len(ref_t)+1):
            for j in range(1, len(hyp_t)+1):
                if ref_t[i-1] == hyp_t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1] / max(len(ref_t), 1)

    def mm_distinct_n(hyps: List[str], n: int = 2) -> float:
        ngrams = []
        for h in hyps:
            t = h.lower().split()
            ngrams.extend([tuple(t[i:i+n]) for i in range(len(t)-n+1)])
        return len(set(ngrams)) / max(len(ngrams), 1)

    def mm_self_bleu(hyps: List[str], n: int = 4) -> float:
        scores = []
        for i, h in enumerate(hyps):
            others = hyps[:i] + hyps[i+1:]
            if not others:
                continue
            ref = " ".join(others)
            scores.append(mm_bleu_n(ref, h, n=n))
        return float(np.mean(scores)) if scores else 0.0

    def mm_light_cider(refs: List[str], hyps: List[str], n_max: int = 4) -> float:
        scores = []
        for r, h in zip(refs, hyps):
            r_t, h_t = r.lower().split(), h.lower().split()
            if not h_t:
                scores.append(0.0); continue
            s = 0.0
            for n in range(1, n_max+1):
                r_ngr, h_ngr = mm_ngrams(r_t, n), mm_ngrams(h_t, n)
                s += len(r_ngr & h_ngr) / max(len(h_ngr), 1)
            scores.append(s/n_max)
        return float(np.mean(scores)) if scores else 0.0

    def mm_light_spice(refs: List[str], hyps: List[str]) -> float:
        scores = []
        for r, h in zip(refs, hyps):
            scores.append(len(set(r.lower().split()) & set(h.lower().split())) / max(len(h.split()), 1))
        return float(np.mean(scores)) if scores else 0.0

    def mm_normalize(mat: np.ndarray) -> np.ndarray:
        return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    def mm_cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return mm_normalize(a) @ mm_normalize(b).T

    def mm_clipscore(clip_model, clip_proc, hyps: List[str], imgs: List[Image.Image]) -> float:
        if clip_model is None or clip_proc is None or not hyps or not imgs or torch is None:
            return 0.0
        with torch.no_grad():
            t = clip_model.get_text_features(**clip_proc(text=hyps, return_tensors="pt", padding=True, truncation=True)).cpu().numpy()
            i = clip_model.get_image_features(**clip_proc(images=imgs, return_tensors="pt")).cpu().numpy()
        sim = mm_cosine_sim(t, i)
        return float(np.mean(np.diag(sim)))

    def mm_recall_at_k(sim_matrix: np.ndarray, k: int) -> float:
        ranks = np.argsort(-sim_matrix, axis=1)
        hits = sum(1 if i in ranks[i, :k] else 0 for i in range(sim_matrix.shape[0]))
        return hits / sim_matrix.shape[0] if sim_matrix.size else 0.0

    def mm_mean_average_precision(sim_matrix: np.ndarray) -> float:
        if sim_matrix.size == 0:
            return 0.0
        y_true = np.eye(sim_matrix.shape[0])
        aps = []
        for i in range(sim_matrix.shape[0]):
            y = y_true[i]
            scores = sim_matrix[i]
            order = np.argsort(-scores)
            y_sorted = y[order]
            prec, hit = [], 0
            for j, rel in enumerate(y_sorted, start=1):
                if rel > 0:
                    hit += 1
                    prec.append(hit / j)
            aps.append(np.mean(prec) if prec else 0.0)
        return float(np.mean(aps))

    def mm_ndcg(sim_matrix: np.ndarray, k: int = 10) -> float:
        n = sim_matrix.shape[0]
        if n == 0:
            return 0.0
        ndcgs = []
        for i in range(n):
            scores = sim_matrix[i]
            idx = np.argsort(-scores)[:k]
            gains = [1 if j == i else 0 for j in idx]
            discounts = [1/np.log2(r+2) for r in range(len(gains))]
            dcg = sum(g * d for g, d in zip(gains, discounts))
            ndcgs.append(dcg/1.0)
        return float(np.mean(ndcgs)) if ndcgs else 0.0

    def mm_median_rank(sim_matrix: np.ndarray) -> float:
        if sim_matrix.size == 0:
            return 0.0
        ranks = np.argsort(-sim_matrix, axis=1)
        positions = [int(np.where(ranks[i] == i)[0][0]) + 1 for i in range(sim_matrix.shape[0])]
        return float(np.median(positions)) if positions else 0.0

    def mm_bootstrap_metric_diff(rows: int, metric_fn, sim_a: np.ndarray, sim_b: np.ndarray, iters: int = 300, k: Optional[int] = None, seed: int = 42):
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(iters):
            idx = rng.integers(0, rows, size=rows)
            sa, sb = sim_a[idx][:, idx], sim_b[idx][:, idx]
            if metric_fn in (mm_recall_at_k, mm_ndcg):
                diffs.append(metric_fn(sb, k) - metric_fn(sa, k))
            else:
                diffs.append(metric_fn(sb) - metric_fn(sa))
        return np.array(diffs)

    # Приближённый CLIP-FID (в CLIP-пространстве)
    def mm_frechet_distance(mu1, sigma1, mu2, sigma2):
        try:
            from scipy import linalg
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            diff = mu1 - mu2
            return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
        except Exception:
            diff = mu1 - mu2
            return diff.dot(diff) + np.sum(sigma1 + sigma2 - 2*np.sqrt(np.maximum(sigma1 * sigma2, 1e-12)))

    def mm_clip_fid(clip_model, clip_proc, real_imgs: List[Image.Image], gen_imgs: List[Image.Image]) -> float:
        if clip_model is None or clip_proc is None or torch is None or not real_imgs or not gen_imgs:
            return 0.0
        with torch.no_grad():
            real = clip_model.get_image_features(**clip_proc(images=real_imgs, return_tensors="pt")).cpu().numpy()
            gen = clip_model.get_image_features(**clip_proc(images=gen_imgs, return_tensors="pt")).cpu().numpy()
        mu_r, mu_g = real.mean(axis=0), gen.mean(axis=0)
        cov_r, cov_g = np.cov(real, rowvar=False), np.cov(gen, rowvar=False)
        return float(mm_frechet_distance(mu_r, cov_r, mu_g, cov_g))

    # Генерация BLIP (caption) и простой VQA-вызов
    def mm_blip_generate_caption(model, proc, img: Image.Image, max_new_tokens: int = 30) -> str:
        try:
            return generate_caption(model, proc, img)
        except Exception:
            # Фоллбэк через .generate (если доступно)
            if model is None or proc is None or torch is None:
                return ""
            inputs = proc(images=img, text="A photo of", return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            try:
                return proc.batch_decode(out, skip_special_tokens=True)[0]
            except Exception:
                return ""

    def mm_blip_vqa_answer(model, proc, img: Image.Image, question: str, max_new_tokens: int = 20) -> str:
        if model is None or proc is None or torch is None:
            return ""
        inputs = proc(images=img, text=question, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        try:
            return proc.batch_decode(out, skip_special_tokens=True)[0]
        except Exception:
            return ""

    # ===================== Вкладки =====================
    tab_cap, tab_ret, tab_gen, tab_vqa, tab_hist = st.tabs(
        ["🖼️ Caption Eval", "🔎 Retrieval Eval", "🎨 ImageGen Eval", "❓ VQA Eval", "🗂️ История"]
    )

    # ===================== 1) Caption Evaluation =====================
    with tab_cap:
        st.subheader("🖼️ Оценка Captioning")
        c1, c2, c3 = st.columns(3)
        with c1:
            csv_cap = st.file_uploader("CSV (image, reference_caption)", type=["csv"], key="mm_cap_csv")
        with c2:
            zip_cap = st.file_uploader("ZIP images", type=["zip"], key="mm_cap_zip")
        with c3:
            do_generate = st.checkbox("Генерировать BLIP подписи", value=True, key="mm_cap_gen")

        if st.button("Запустить Caption Eval", type="primary", key="mm_cap_run"):
            if not (csv_cap and zip_cap):
                st.warning("Загрузите CSV и ZIP с изображениями.")
            else:
                df = pd.read_csv(csv_cap)
                zbytes = io.BytesIO(zip_cap.read())
                with zipfile.ZipFile(zbytes) as zf:
                    refs, hyps, imgs = [], [], []
                    for _, row in df.iterrows():
                        fname = str(row.get("image", ""))
                        ref = str(row.get("reference_caption", ""))
                        if not fname or fname not in zf.namelist():
                            st.warning(f"Нет файла в ZIP: {fname}")
                            continue
                        with zf.open(fname) as f:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                        hyp = mm_blip_generate_caption(blip_cap_model, blip_cap_proc, img) if do_generate else ref
                        refs.append(ref); hyps.append(hyp); imgs.append(img)

                if refs:
                    bleu = float(np.mean([mm_bleu_n(r, h) for r, h in zip(refs, hyps)]))
                    rouge = float(np.mean([mm_rouge_l(r, h) for r, h in zip(refs, hyps)]))
                    cider_val = mm_light_cider(refs, hyps)
                    spice_val = mm_light_spice(refs, hyps)
                    dist2 = mm_distinct_n(hyps, n=2)
                    sbleu = mm_self_bleu(hyps)
                    clip_score_val = mm_clipscore(clip_model_a, clip_proc_a, hyps, imgs)

                    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                    m1.metric("BLEU", f"{bleu:.3f}")
                    m2.metric("ROUGE-L", f"{rouge:.3f}")
                    m3.metric("CIDEr (light)", f"{cider_val:.3f}")
                    m4.metric("SPICE (light)", f"{spice_val:.3f}")
                    m5.metric("Distinct-2", f"{dist2:.3f}")
                    m6.metric("Self-BLEU", f"{sbleu:.3f}")
                    m7.metric("CLIPScore", f"{clip_score_val:.3f}")

                    # Опциональная LLM-оценка нескольких примеров
                    llm_judgements = []
                    if use_llm_eval and llm_api_key and OpenAI is not None:
                        try:
                            client = OpenAI(api_key=llm_api_key, base_url=llm_api_base or None)
                            max_items = min(12, len(refs))
                            with st.expander("🤖 LLM-оценка примеров (до 12)"):
                                for i in range(max_items):
                                    prompt = (
                                        "Оцени подпись изображения по релевантности и полноте. "
                                        "Верни ЧИСЛО 1–10 и краткое объяснение в 1–2 предложения.\n"
                                        f"Эталон: {refs[i]}\nКапшен: {hyps[i]}"
                                    )
                                    resp = client.chat.completions.create(
                                        model=llm_model_name or "gpt-4o-mini",
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=0
                                    )
                                    txt = resp.choices[0].message.content.strip()
                                    st.write(f"#{i+1} → {txt}")
                                    llm_judgements.append({"ref": refs[i], "hyp": hyps[i], "judge": txt})
                        except Exception as e:
                            st.info(f"LLM-оценка недоступна: {e}")

                    # Быстрая ручная оценка
                    with st.expander("🧑‍⚖️ Быстрая ручная оценка"):
                        ratings = []
                        grid_cols = st.columns(4)
                        for idx in range(min(12, len(imgs))):
                            with grid_cols[idx % 4]:
                                st.image(imgs[idx], use_column_width=True)
                                st.caption(f"Ref: {refs[idx]}\n\nHyp: {hyps[idx]}")
                                r = st.slider("Оценка (1–5)", 1, 5, 3, key=f"mm_cap_rate_{idx}")
                                ratings.append({"ref": refs[idx], "hyp": hyps[idx], "rating": int(r)})

                    mm_add_history({
                        "type": "caption_eval",
                        "metrics": {"bleu": bleu, "rouge_l": rouge, "cider_light": cider_val, "spice_light": spice_val,
                                    "distinct_2": dist2, "self_bleu": sbleu, "clipscore": clip_score_val},
                        "llm": llm_judgements[:],
                    })
                    st.success("Готово. Результаты добавлены в Историю.")

    # ===================== 2) Retrieval Evaluation =====================
    with tab_ret:
        st.subheader("🔎 Оценка Retrieval (Text→Image)")
        c1, c2, c3 = st.columns(3)
        with c1:
            csv_ret = st.file_uploader("CSV (text,image)", type=["csv"], key="mm_ret_csv")
        with c2:
            zip_ret = st.file_uploader("ZIP images", type=["zip"], key="mm_ret_zip")
        with c3:
            n_boot = st.slider("Бутстрэп итераций", 100, 800, 300, 50, key="mm_ret_boot")

        if st.button("Запустить Retrieval", type="primary", key="mm_ret_run"):
            if not (csv_ret and zip_ret):
                st.warning("Загрузите CSV и ZIP с изображениями.")
            else:
                df = pd.read_csv(csv_ret)
                zbytes = io.BytesIO(zip_ret.read())
                with zipfile.ZipFile(zbytes) as zf:
                    imgs, texts = [], []
                    for _, row in df.iterrows():
                        text = str(row.get("text", str(row.iloc[0])))
                        fname = str(row.get("image", str(row.iloc[1])))
                        if fname not in zf.namelist():
                            st.warning(f"Нет файла в ZIP: {fname}")
                            continue
                        with zf.open(fname) as f:
                            imgs.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
                        texts.append(text)

                if imgs:
                    if clip_model_a is None or clip_proc_a is None or torch is None:
                        st.error("CLIP (A) не загружен.")
                    else:
                        with torch.no_grad():
                            t_emb_a = clip_model_a.get_text_features(**clip_proc_a(text=texts, return_tensors="pt", padding=True, truncation=True)).cpu().numpy()
                            i_emb_a = clip_model_a.get_image_features(**clip_proc_a(images=imgs, return_tensors="pt")).cpu().numpy()
                        sim_a = mm_cosine_sim(t_emb_a, i_emb_a)

                        r1, r5, r10 = mm_recall_at_k(sim_a, 1), mm_recall_at_k(sim_a, 5), mm_recall_at_k(sim_a, 10)
                        map_score = mm_mean_average_precision(sim_a)
                        ndcg10 = mm_ndcg(sim_a, 10)
                        med_rank = mm_median_rank(sim_a)

                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("R@1", f"{r1:.3f}")
                        m2.metric("R@5", f"{r5:.3f}")
                        m3.metric("R@10", f"{r10:.3f}")
                        m4.metric("mAP", f"{map_score:.3f}")
                        m5.metric("nDCG@10", f"{ndcg10:.3f}")
                        m6.metric("Median Rank", f"{med_rank:.1f}")

                        # Кривая Recall@k
                        ks = list(range(1, min(21, sim_a.shape[1] + 1)))
                        curve = pd.DataFrame({"k": ks, "Recall@k": [mm_recall_at_k(sim_a, k) for k in ks]})
                        try:
                            st.line_chart(curve.set_index("k"))
                        except Exception:
                            st.dataframe(curve)

                        # Симметричная метрика Image→Text
                        r1_img = mm_recall_at_k(sim_a.T, 1)
                        st.caption(f"Image→Text R@1: {r1_img:.3f}")

                        ab_result = None
                        df_boot_long = None

                        # ======== A/B сравнение с бутстрэпом ========
                        if enable_ab and clip_model_b is not None and clip_proc_b is not None:
                            with torch.no_grad():
                                t_emb_b = clip_model_b.get_text_features(**clip_proc_b(text=texts, return_tensors="pt", padding=True, truncation=True)).cpu().numpy()
                                i_emb_b = clip_model_b.get_image_features(**clip_proc_b(images=imgs, return_tensors="pt")).cpu().numpy()
                            sim_b = mm_cosine_sim(t_emb_b, i_emb_b)

                            diffs_r1 = mm_bootstrap_metric_diff(sim_a.shape[0], mm_recall_at_k, sim_a, sim_b, iters=int(n_boot), k=1)
                            diffs_map = mm_bootstrap_metric_diff(sim_a.shape[0], mm_mean_average_precision, sim_a, sim_b, iters=int(n_boot))
                            diffs_ndcg = mm_bootstrap_metric_diff(sim_a.shape[0], mm_ndcg, sim_a, sim_b, iters=int(n_boot), k=10)

                            m_r1, m_map, m_ndcg = float(np.mean(diffs_r1)), float(np.mean(diffs_map)), float(np.mean(diffs_ndcg))
                            d1, d2, d3 = st.columns(3)
                            d1.metric("ΔR@1 (B−A)", f"{m_r1:+.3f}")
                            d2.metric("ΔmAP (B−A)", f"{m_map:+.3f}")
                            d3.metric("ΔnDCG@10 (B−A)", f"{m_ndcg:+.3f}")

                            # Визуализация распределений
                            try:
                                import pandas as _pd
                                df_boot = _pd.DataFrame({"ΔR@1": diffs_r1, "ΔmAP": diffs_map, "ΔnDCG@10": diffs_ndcg})
                                df_boot_long = df_boot.melt(var_name="metric", value_name="delta")
                                cols = st.columns(3)
                                if alt is not None:
                                    for ci, metric_name in enumerate(["ΔR@1", "ΔmAP", "ΔnDCG@10"]):
                                        chart = alt.Chart(df_boot[[metric_name]].rename(columns={metric_name: "delta"})).mark_bar().encode(
                                            x=alt.X("delta:Q", bin=alt.Bin(maxbins=40), title=metric_name),
                                            y=alt.Y("count():Q", title="Count")
                                        ).properties(height=180)
                                        cols[ci].altair_chart(chart, use_container_width=True)
                                elif plt is not None:
                                    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                                    axes[0].hist(diffs_r1, bins=30); axes[0].set_title("ΔR@1")
                                    axes[1].hist(diffs_map, bins=30); axes[1].set_title("ΔmAP")
                                    axes[2].hist(diffs_ndcg, bins=30); axes[2].set_title("ΔnDCG@10")
                                    for ax in axes: ax.grid(True, linestyle=":", alpha=0.4)
                                    st.pyplot(fig)
                            except Exception:
                                pass

                            # Экспорт CSV распределений
                            if df_boot_long is not None:
                                csv_bytes = df_boot_long.to_csv(index=False).encode("utf-8")
                                st.download_button("Скачать распределения бутстрэпа (CSV)", data=csv_bytes,
                                                   file_name="ab_bootstrap_distributions.csv", mime="text/csv")

                        # LLM-сводка результатов (опционально)
                        llm_summary = None
                        if use_llm_eval and llm_api_key and OpenAI is not None:
                            try:
                                client = OpenAI(api_key=llm_api_key, base_url=llm_api_base or None)
                                prompt = (
                                    "Суммируй результаты retrieval-оценки. Объясни значения метрик и выводы A/B-теста кратко.\n"
                                    f"R@1={r1:.3f}, R@5={r5:.3f}, R@10={r10:.3f}, mAP={map_score:.3f}, nDCG@10={ndcg10:.3f}, MedRank={med_rank:.1f}.\n"
                                    f"A/B: {'включён' if enable_ab else 'выключен'}."
                                )
                                resp = client.chat.completions.create(
                                    model=llm_model_name or "gpt-4o-mini",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0
                                )
                                llm_summary = resp.choices[0].message.content.strip()
                                st.info(llm_summary)
                            except Exception as e:
                                st.info(f"LLM-сводка недоступна: {e}")

                        mm_add_history({
                            "type": "retrieval_eval",
                            "metrics": {"r1": r1, "r5": r5, "r10": r10, "map": map_score, "ndcg@10": ndcg10,
                                        "median_rank": med_rank, "img2text_r1": float(r1_img)},
                            "ab_enabled": bool(enable_ab),
                            "llm_summary": llm_summary
                        })
                        st.success("Готово. Результаты добавлены в Историю.")

    # ===================== 3) Image Generation Evaluation (approx) =====================
    with tab_gen:
        st.subheader("🎨 Оценка генерации изображений (приближённо)")
        st.caption("CLIP-FID (приближённый) в CLIP-пространстве + соответствие prompt→image по CLIPScore.")

        c1, c2, c3 = st.columns(3)
        with c1:
            zip_real = st.file_uploader("ZIP Real Images", type=["zip"], key="mm_gen_zip_real")
        with c2:
            zip_gen = st.file_uploader("ZIP Generated Images", type=["zip"], key="mm_gen_zip_gen")
        with c3:
            csv_prompts = st.file_uploader("CSV (prompt, gen_image)", type=["csv"], key="mm_gen_csv_prompts")

        if st.button("Оценить генерацию", type="primary", key="mm_gen_run"):
            # Загрузка изображений
            real_imgs, gen_imgs = [], []
            if zip_real:
                with zipfile.ZipFile(io.BytesIO(zip_real.read())) as zf:
                    for name in zf.namelist():
                        try:
                            with zf.open(name) as f:
                                real_imgs.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
                        except Exception:
                            continue
            if zip_gen:
                bytes_gen = io.BytesIO(zip_gen.read())
                with zipfile.ZipFile(bytes_gen) as zf:
                    for name in zf.namelist():
                        try:
                            with zf.open(name) as f:
                                gen_imgs.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
                        except Exception:
                            continue

            metrics = {}
            if real_imgs and gen_imgs:
                cfid = mm_clip_fid(clip_model_a, clip_proc_a, real_imgs, gen_imgs)
                metrics["CLIP-FID (approx)"] = cfid

            # Соответствие промптов и сгенерированных изображений
            if csv_prompts is not None and zip_gen is not None:
                dfp = pd.read_csv(csv_prompts)
                gen_map = {}
                with zipfile.ZipFile(bytes_gen) as zf:
                    for name in zf.namelist():
                        try:
                            gen_map[name] = zf.read(name)
                        except Exception:
                            pass
                hyps, imgs = [], []
                for _, row in dfp.iterrows():
                    prompt = str(row.get("prompt", ""))
                    gname = str(row.get("gen_image", ""))
                    if gname in gen_map:
                        try:
                            img = Image.open(io.BytesIO(gen_map[gname])).convert("RGB")
                        except Exception:
                            continue
                        imgs.append(img)
                        hyps.append(prompt)
                if imgs:
                    clipscore_ti = mm_clipscore(clip_model_a, clip_proc_a, hyps, imgs)
                    metrics["CLIPScore (prompt→image)"] = clipscore_ti

            if metrics:
                cols = st.columns(len(metrics))
                for i, (k, v) in enumerate(metrics.items()):
                    cols[i].metric(k, f"{v:.3f}")
                mm_add_history({"type": "imagegen_eval", "metrics": metrics})
                st.success("Готово. Результаты добавлены в Историю.")
            else:
                st.warning("Недостаточно данных для расчёта метрик.")

    # ===================== 4) VQA / Reasoning Eval (simple) =====================
    with tab_vqa:
        st.subheader("❓ VQA / Reasoning (простая проверка)")
        st.caption("Exact match по ответу + опциональный LLM-судья для семантического совпадения.")

        c1, c2, c3 = st.columns(3)
        with c1:
            csv_vqa = st.file_uploader("CSV (image, question, answer)", type=["csv"], key="mm_vqa_csv")
        with c2:
            zip_vqa = st.file_uploader("ZIP images", type=["zip"], key="mm_vqa_zip")
        with c3:
            max_samples = st.number_input("Макс. примеров", 1, 2000, 200, key="mm_vqa_max")

        if st.button("Запустить VQA", type="primary", key="mm_vqa_run"):
            if not (csv_vqa and zip_vqa):
                st.warning("Загрузите CSV и ZIP с изображениями.")
            else:
                df = pd.read_csv(csv_vqa)
                correct, total = 0, 0
                examples = []
                with zipfile.ZipFile(io.BytesIO(zip_vqa.read())) as zf:
                    for i, row in df.head(int(max_samples)).iterrows():
                        fname = str(row.get("image", ""))
                        q = str(row.get("question", ""))
                        ans = str(row.get("answer", "")).strip().lower()
                        if not fname or fname not in zf.namelist():
                            continue
                        with zf.open(fname) as f:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                        pred = mm_blip_vqa_answer(blip_vqa_model, blip_vqa_proc, img, q).strip().lower()
                        total += 1
                        ok = int(pred == ans)
                        correct += ok
                        if len(examples) < 16:
                            examples.append({"image": fname, "q": q, "gt": ans, "pred": pred, "ok": ok})

                acc = (correct / total) if total else 0.0
                st.metric("Accuracy (exact match)", f"{acc:.3f}")

                # LLM-судья (семантика)
                if use_llm_eval and llm_api_key and OpenAI is not None and examples:
                    try:
                        client = OpenAI(api_key=llm_api_key, base_url=llm_api_base or None)
                        agree = 0
                        with st.expander("🤖 LLM-судья (семантическое совпадение)"):
                            for ex in examples:
                                prompt = (
                                    "Проверь семантическую эквивалентность ответа. Ответь только: YES или NO.\n"
                                    f"Вопрос: {ex['q']}\nИстина: {ex['gt']}\nПредсказание: {ex['pred']}"
                                )
                                resp = client.chat.completions.create(
                                    model=llm_model_name or "gpt-4o-mini",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0
                                )
                                verdict = resp.choices[0].message.content.strip().upper()
                                st.write(f"{ex['image']} → {verdict}")
                                if verdict.startswith("Y"):
                                    agree += 1
                        st.caption(f"LLM-согласие на примерах: {agree}/{len(examples)}")
                    except Exception as e:
                        st.info(f"LLM-судья недоступен: {e}")

                mm_add_history({"type": "vqa_eval", "accuracy": float(acc), "samples": int(total)})
                st.success("Готово. Результаты добавлены в Историю.")

    # ===================== 5) История =====================
    with tab_hist:
        st.subheader("🗂️ История мультимодальных экспериментов")
        if st.session_state["mm_history"]:
            st.json(st.session_state["mm_history"][-10:])
            data = json.dumps(st.session_state["mm_history"], ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("Скачать историю (JSON)", data=data, file_name="mm_history.json", mime="application/json")
        else:
            st.info("История пуста. Запустите одну из оценок выше.")
