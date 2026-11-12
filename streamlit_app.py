import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Multilingual IR Lab", page_icon="🔎", layout="wide")
API_URL = os.getenv("IR_API_URL", "http://127.0.0.1:8000").rstrip("/")
DATA_DIR = Path(os.environ.get("MIRACL_DATA_DIR", str(Path(__file__).resolve().parent / "data")))
RESULTS_DIR = DATA_DIR / "miracl_eval" / "results"

st.title("🔎 Multilingual Information Retrieval")
st.caption("Compare lexical, semantic and hybrid retrieval over English and Hindi MIRACL passages.")

with st.sidebar:
    st.header("Search configuration")
    engine_label = st.selectbox("Retrieval engine", [
        "Engine A — BM25", "Engine B — Dense (multilingual-E5)",
        "Engine C — Hybrid (BM25 + Dense)",
    ])
    engine = {
        "Engine A — BM25": "A",
        "Engine B — Dense (multilingual-E5)": "B",
        "Engine C — Hybrid (BM25 + Dense)": "C",
    }[engine_label]
    scenario_label = st.selectbox("Query → document language", [
        "English → English", "Hindi → Hindi", "Hindi → English",
    ])
    scenario = {
        "English → English": "EN-EN",
        "Hindi → Hindi": "HI-HI",
        "Hindi → English": "HI-EN",
    }[scenario_label]
    top_k = st.slider("Number of results", 5, 50, 10, 5)
    bm25_mode = "bm25_translate"
    if engine == "A" and scenario == "HI-EN":
        bm25_mode = st.selectbox("Engine A HI → EN method", [
            "bm25_translate", "bm25_direct",
        ], format_func=lambda value: {
            "bm25_translate": "Translate Hindi to English, then BM25",
            "bm25_direct": "Direct Hindi BM25 (baseline)",
        }[value])
    if scenario == "HI-EN":
        st.caption("Hindi queries search the English corpus. Engine B uses multilingual E5; Engine C fuses E5 with translated-query BM25.")

tab_search, tab_results = st.tabs(["Search", "Evaluation results"])

with tab_search:
    st.subheader("Search")
    with st.expander("Example test queries", expanded=False):
        if st.button("Load example queries", key="load_examples"):
            try:
                response = requests.get(
                    f"{API_URL}/api/examples", params={"scenario": scenario}, timeout=15
                )
                response.raise_for_status()
                st.session_state["examples"] = response.json().get("queries", [])
            except requests.RequestException as exc:
                st.error(f"Could not load examples: {exc}")
        examples = st.session_state.get("examples", [])
        if examples:
            labels = [f"{item.get('qid', i)} — {item.get('query', '')}"
                      for i, item in enumerate(examples)]
            choice = st.selectbox("Choose a query", range(len(labels)), format_func=lambda i: labels[i])
            if st.button("Use selected query"):
                st.session_state["query_input"] = examples[choice]["query"]
        else:
            st.info("Load the MIRACL evaluation queries to select an example.")

    query = st.text_area("Enter your query", key="query_input", height=100,
                         placeholder="Type an English or Hindi query...")
    if st.button("Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            with st.spinner(f"Running Engine {engine}..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/search",
                        json={"engine": engine, "scenario": scenario, "query": query.strip(),
                              "top_k": top_k, "bm25_mode": bm25_mode},
                        timeout=300,
                    )
                    response.raise_for_status()
                    st.session_state["last_search"] = response.json()
                except requests.HTTPError:
                    try:
                        st.error(response.json().get("detail", response.text))
                    except Exception:
                        st.error(response.text)
                except requests.RequestException as exc:
                    st.error(f"Backend unavailable: {exc}. Start FastAPI and check IR_API_URL.")

    result = st.session_state.get("last_search")
    if result:
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Engine", f"Engine {result['engine']}")
        col2.metric("Retrieved passages", len(result.get("results", [])))
        col3.metric("Latency", f"{result.get('latency_ms', 0):.1f} ms")
        st.caption(
            f"Query: {result['query']} · Scenario: {result['scenario']} · "
            f"Target corpus: {result['target_corpus']}"
        )
        if result.get("translated_query"):
            st.caption(f"English translation used for BM25: {result['translated_query']}")
        for item in result.get("results", []):
            with st.container(border=True):
                left, right = st.columns([5, 1])
                left.markdown(f"**#{item['rank']} · {item.get('title') or 'Untitled passage'}**")
                left.caption(f"Document ID: {item['docid']}")
                right.metric("Score", f"{item['score']:.4f}")
                st.write(item.get("text", "Passage text unavailable."))
                if "bm25_component" in item:
                    st.caption(
                        f"BM25 component: {item['bm25_component']:.4f} · "
                        f"Dense component: {item['dense_component']:.4f}"
                    )

    st.divider()
    st.subheader("Evaluation readiness")
    try:
        response = requests.get(f"{API_URL}/api/evaluation/status", timeout=10)
        response.raise_for_status()
        status = response.json()
        rows = [
            {"Language/set": lang.upper(),
             **{name: ("Present" if present else "Missing") for name, present in values.items()}}
            for lang, values in status["files"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(status["note"])
    except requests.RequestException:
        st.info("Start the FastAPI backend to inspect evaluation-file availability.")


def _find_results_file(*names):
    for name in names:
        path = RESULTS_DIR / name
        if path.is_file():
            return path
    return None


def _load_engine_results(*names):
    path = _find_results_file(*names)
    if path is None:
        return None, None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return None, f"Could not read {path.name}: {exc}"
    if "Model" not in frame.columns:
        return None, f"{path.name} does not contain a Model column."
    models = frame["Model"].fillna("").astype(str)
    frame = frame[models.str.contains("→|->", regex=True)].copy()
    frame["Scenario"] = frame["Model"].map(_scenario_for_model)
    return frame, path.name


def _scenario_for_model(model):
    model = str(model).casefold()
    if "hi→en" in model or "hi->en" in model:
        return "Hindi → English"
    if "hi→hi" in model or "hi->hi" in model:
        return "Hindi → Hindi"
    if "en→en" in model or "en->en" in model:
        return "English → English"
    return "Other"


def _render_results(frame, source_name, selected_scenario, key):
    if frame is None:
        st.info(source_name or "No result CSV was found in the results folder.")
        return
    if selected_scenario != "All scenarios":
        frame = frame[frame["Scenario"] == selected_scenario]
    if frame.empty:
        st.info(f"No rows for {selected_scenario.lower()} in {source_name}.")
        return

    st.caption(f"Source: `{source_name}` · {len(frame)} result rows")
    columns = [name for name in [
        "Model", "n_queries", "P@5", "R@100", "MAP", "NDCG@5", "Latency (ms)", "Index (MB)"
    ] if name in frame.columns]
    table = frame[columns].copy()
    formats = {
        name: st.column_config.NumberColumn(format="%.3f")
        for name in ["P@5", "R@100", "MAP", "NDCG@5", "Latency (ms)", "Index (MB)"]
        if name in table.columns
    }
    st.dataframe(table, use_container_width=True, hide_index=True, column_config=formats)

    metric_columns = [name for name in ["MAP", "NDCG@5"] if name in frame.columns]
    if metric_columns:
        chart = frame.set_index("Model")[metric_columns].apply(pd.to_numeric, errors="coerce")
        st.bar_chart(chart, use_container_width=True, height=300)
    st.download_button(
        "Download displayed rows",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{key}_{selected_scenario.lower().replace(' ', '_').replace('→', 'to')}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


with tab_results:
    st.subheader("Retrieval evaluation")
    st.caption(
        "Compare precision at 5, recall in the top 100, MAP, NDCG at 5, and latency. "
        "Higher ranking metrics are better; latency is milliseconds per query."
    )
    selected_scenario = st.selectbox(
        "Query → document language",
        ["All scenarios", "English → English", "Hindi → Hindi", "Hindi → English"],
        key="result_scenario",
    )
    engine_a_tab, engine_b_tab, engine_c_tab = st.tabs([
        "Engine A · BM25", "Engine B · Dense E5", "Engine C · Hybrid",
    ])

    with engine_a_tab:
        st.markdown("BM25 lexical retrieval. HI → EN includes translated-query BM25 and the direct Hindi baseline.")
        frame, source = _load_engine_results("bm25_results_local.csv", "bm25_results.csv")
        _render_results(frame, source, selected_scenario, "engine_a")

    with engine_b_tab:
        st.markdown("Multilingual E5 dense retrieval against the selected target-language index.")
        frame, source = _load_engine_results("dense_results_local.csv", "dense_results.csv")
        _render_results(frame, source, selected_scenario, "engine_b")

    with engine_c_tab:
        st.markdown("Hybrid results show BM25, dense, and alpha-weighted fused rankings for comparison.")
        frame, source = _load_engine_results("hybrid_results.csv")
        _render_results(frame, source, selected_scenario, "engine_c")

        st.markdown("#### Alpha tuning")
        sweep_frames = []
        for lang in ["en", "hi"]:
            sweep_path = RESULTS_DIR / f"hybrid_alpha_sweep_{lang}.csv"
            if sweep_path.is_file():
                sweep = pd.read_csv(sweep_path)
                sweep["Query language"] = lang.upper()
                sweep_frames.append(sweep)
        if sweep_frames:
            sweep = pd.concat(sweep_frames, ignore_index=True)
            alpha_col, score_col = st.columns(2)
            with alpha_col:
                st.caption("NDCG@5 across candidate alpha values")
                st.line_chart(sweep.pivot(index="alpha", columns="Query language", values="NDCG@5"),
                              use_container_width=True)
            with score_col:
                st.caption("MAP across candidate alpha values")
                st.line_chart(sweep.pivot(index="alpha", columns="Query language", values="MAP"),
                              use_container_width=True)
        else:
            st.info("No alpha-sweep CSVs found.")

st.caption("This interface displays retrieval results. It does not generate answers or fabricate relevance metrics.")
