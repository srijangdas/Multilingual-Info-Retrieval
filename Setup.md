
# Multilingual IR Lab — Streamlit + FastAPI

A local-first application for the repository's three engines:
- Engine A: BM25 (Pyserini/Lucene; k1=0.9, b=0.4)
- Engine B: Dense multilingual-E5-small + FAISS
- Engine C: Hybrid BM25 + Dense with min-max or weighted-RRF fusion

The app uses your existing indexes and passages. It does not rebuild them.

## 1. Requirements

- Python 3.10 or 3.11 recommended.
- Java JDK 21 recommended for Pyserini/Lucene. Ensure `java -version` works.
- Enough RAM and disk for your local indexes.
- The model checkpoint must match the model used when the dense indexes were built.

## 2. Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Pyserini may have platform-specific installation requirements. Install a compatible Java JDK and follow the Pyserini installation guide if its wheel/dependencies fail.

## 3. Add your data

Copy the folders from your Google Drive artifacts into `data/` using the exact layout described in `data/README_DATA.txt`.

The API expects:
- `miracl_en_indexes/bm25_index`, `dense.faiss`, `docids.json`, `passages.jsonl`
- `miracl_hi_indexes/bm25_index`, `dense.faiss`, `docids.json`, `passages.jsonl`
- `miracl_eval/en` and `miracl_eval/hi` for optional evaluation files.

The app ships with example-query placeholders. Replace `data/examples/*.json` or add your MIRACL `eval_queries.json` files. Query relevance labels (qrels) are never sent to the browser.

## 4. Configure

Copy `.env.example` to `.env` and adjust `MIRACL_DATA_DIR` if your artifacts are stored elsewhere.

Set `E5_MODEL` to the exact checkpoint used to build your FAISS vectors. The repository README/source specifies `intfloat/multilingual-e5-small`.

For Engine C, set `HYBRID_ALPHA` to the value tuned by your Engine C validation run. `HYBRID_METHOD` may be `minmax` or `rrf`.

### Build a fast passage lookup database (strongly recommended)

The retrieval indexes return document IDs; the UI then needs passage text for those IDs. A plain JSONL scan on every query is slow for a 1.5M-passage collection. Build an indexed SQLite lookup once:

```bash
python scripts/build_docstore.py --lang en
python scripts/build_docstore.py --lang hi
```

This creates `passages.sqlite` beside each language's index. Keep the original `passages.jsonl`; the SQLite files are additional lookup artifacts.

## 5. Run locally

Open terminal 1:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open terminal 2:

```bash
streamlit run streamlit_app.py
```

- Streamlit: http://localhost:8501
- FastAPI Swagger docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

The engines are loaded lazily, once per API process and target language, on the first search. The API process must be able to read your data folders.

## 6. Query scenarios

- EN-EN: English query searches English corpus.
- HI-HI: Hindi query searches Hindi corpus.
- HI-EN: Hindi query searches English corpus (E5 cross-lingual retrieval).

Note: BM25 is lexical. In HI-EN, Engine A compares Hindi query terms against English documents without translation, so weak results are expected. This is not a bug in the UI. Engine B and Engine C use the multilingual E5 query embedding for cross-lingual matching.

## 7. Evaluation

The current UI reports live retrieval latency and result scores. It also reports whether evaluation files are present. It intentionally does not show NDCG/MAP/Recall for an arbitrary user query because those metrics require qrels.

A full benchmark evaluation endpoint/dashboard can be wired to your exact evaluation scripts once you supply/confirm the final `eval_queries.json`, `eval_qrels.json`, hard-query files and Engine C alpha/split outputs. This starter includes the file locations and readiness check, not fabricated benchmark results.

## 8. Important implementation notes

- Engine A and Engine C require Pyserini plus Java.
- Engine B and Engine C require `faiss-cpu`, `sentence-transformers`, and PyTorch.
- The code loads one engine per target language on first use. Dense and Hybrid instances each load an encoder; depending on RAM, you may want to refactor to share one encoder instance between them.
- Passage text lookup scans `passages.jsonl` for returned IDs. For frequent searches over 1.5M passages, replace this with a SQLite/LMDB/SQLite FTS or indexed key-value document store, or preload passages if RAM permits. This starter keeps the on-disk scan simple and faithful to the source artifact layout.
- For remote access, do not expose the local API without authentication. Use a secured tunnel/reverse proxy and configure allowed origins rather than `*` for production.

## 9. Project layout

```text
backend/
  main.py
  config.py
  service.py
  engines/
    bm25.py
    dense.py
    hybrid.py
streamlit_app.py
data/
  miracl_en_indexes/   # add your files
  miracl_hi_indexes/   # add your files
  miracl_eval/         # add query/qrels files
  examples/            # editable query placeholders
```
