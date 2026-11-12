
import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multilingual IR Lab", page_icon="🔎", layout="wide")
API_URL = os.getenv("IR_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.title("🔎 Multilingual Information Retrieval")
st.caption("Compare lexical, semantic and hybrid retrieval over English and Hindi MIRACL passages.")

with st.sidebar:
    st.header("Search configuration")
    engine_label = st.selectbox("Retrieval engine", [
        "Engine A — BM25", "Engine B — Dense (multilingual-E5)", "Engine C — Hybrid (BM25 + Dense)"
    ])
    engine = {"Engine A — BM25": "A", "Engine B — Dense (multilingual-E5)": "B",
              "Engine C — Hybrid (BM25 + Dense)": "C"}[engine_label]
    scenario_label = st.selectbox("Query → document language", [
        "English → English", "Hindi → Hindi", "Hindi → English"
    ])
    scenario = {"English → English": "EN-EN", "Hindi → Hindi": "HI-HI",
                "Hindi → English": "HI-EN"}[scenario_label]
    top_k = st.slider("Number of results", 5, 50, 10, 5)
    st.caption("For HI → EN, the Hindi query is sent to the English target corpus. BM25 remains lexical; no translation is silently applied.")

st.subheader("Search")
with st.expander("Example test queries", expanded=False):
    if st.button("Load example queries", key="load_examples"):
        try:
            r = requests.get(f"{API_URL}/api/examples", params={"scenario": scenario}, timeout=15)
            r.raise_for_status()
            st.session_state["examples"] = r.json().get("queries", [])
        except requests.RequestException as e:
            st.error(f"Could not load examples: {e}")
    examples = st.session_state.get("examples", [])
    if examples:
        labels = [f"{x.get('qid', i)} — {x.get('query', '')}" for i, x in enumerate(examples)]
        choice = st.selectbox("Choose a query", range(len(labels)), format_func=lambda i: labels[i])
        if st.button("Use selected query"):
            st.session_state["query_input"] = examples[choice]["query"]
    else:
        st.info("Example-query placeholders are in data/examples/. Add your test queries or load MIRACL eval_queries.json.")

query = st.text_area("Enter your query", key="query_input", height=100,
                     placeholder="Type an English or Hindi query...")
if st.button("Search", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Enter a query first.")
    else:
        with st.spinner(f"Running Engine {engine}..."):
            try:
                response = requests.post(f"{API_URL}/api/search",
                    json={"engine": engine, "scenario": scenario, "query": query.strip(), "top_k": top_k},
                    timeout=300)
                response.raise_for_status()
                data = response.json()
                st.session_state["last_search"] = data
            except requests.HTTPError:
                try: st.error(response.json().get("detail", response.text))
                except Exception: st.error(response.text)
            except requests.RequestException as e:
                st.error(f"Backend unavailable: {e}. Start FastAPI and check IR_API_URL.")

data = st.session_state.get("last_search")
if data:
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Engine", f"Engine {data['engine']}")
    c2.metric("Retrieved passages", len(data.get("results", [])))
    c3.metric("Latency", f"{data.get('latency_ms', 0):.1f} ms")
    st.caption(f"Query: {data['query']} · Scenario: {data['scenario']} · Target corpus: {data['target_corpus']}")
    for item in data.get("results", []):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**#{item['rank']} · {item.get('title') or 'Untitled passage'}**")
            left.caption(f"Document ID: {item['docid']}")
            right.metric("Score", f"{item['score']:.4f}")
            st.write(item.get("text", "Passage text unavailable."))
            if "bm25_component" in item:
                st.caption(f"BM25 component: {item['bm25_component']:.4f} · Dense component: {item['dense_component']:.4f}")

st.divider()
st.subheader("Evaluation readiness")
try:
    status = requests.get(f"{API_URL}/api/evaluation/status", timeout=10).json()
    rows = []
    for lang, values in status["files"].items():
        rows.append({"Language": lang.upper(), **{k: ("Present" if v else "Missing") for k, v in values.items()}})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(status["note"])
except requests.RequestException:
    st.info("Start the FastAPI backend to inspect evaluation-file availability.")

st.caption("This interface displays retrieval results. It does not generate an LLM answer or fabricate relevance metrics.")
