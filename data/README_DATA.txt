
Place your existing Google Drive/Colab artifacts under this directory.

Expected layout (as documented by the GitHub repository):

data/
  miracl_en_indexes/
    bm25_index/          # Pyserini/Lucene index directory
    dense.faiss
    docids.json
    passages.jsonl
  miracl_hi_indexes/
    bm25_index/
    dense.faiss
    docids.json
    passages.jsonl
  miracl_eval/
    en/
      eval_queries.json
      eval_qrels.json
      hard_queries.json       # optional
      hard_qrels.json         # optional
      hard_usable.json        # optional
      hard_meta.json          # optional
    hi/
      same files as English
  examples/
    en_en.json
    hi_hi.json
    hi_en.json

Important:
- Dense docids.json ordering must exactly match the FAISS vector row order.
- BM25 docids must use the same ID strings as the passage JSONL.
- Engine C requires BOTH BM25 and dense files for the selected target language.
- For HI-EN, the Hindi query is embedded with E5 and searches the English FAISS index.
- BM25 on HI-EN is lexical Hindi-vs-English matching and may perform poorly; this app does not translate queries.
- Do not commit large indexes, passage corpora, qrels or model weights to Git.

Optional but recommended: run scripts/build_docstore.py for each language to create passages.sqlite, an indexed lookup database used to fetch retrieved text without rescanning all JSONL passages on every request.
