# Multilingual Information Retrieval: Keyword Matching vs. Semantic Retrieval

A comparison of **keyword-based (BM25)** and **semantic (neural/dense)** retrieval on English and Hindi, built on the MIRACL benchmark. The project builds several retrieval engines over the same corpora and evaluates them with standard IR metrics.

---

## 1. Dataset

We use **MIRACL** (Multilingual Information Retrieval Across a Continuum of Languages), a Wikipedia-based retrieval benchmark with human-annotated relevance judgments.

| Language | Corpus | Queries (dev) |
|----------|--------|---------------|
| English  | MIRACL English corpus (subset: 2 shards, ~1M passages) | 799 total, of which 230 are covered by our indexed subset |
| Hindi    | MIRACL Hindi corpus | 350 |

- **Corpus**: Wikipedia passages (the documents we index and search).
- **Queries**: natural-language questions from MIRACL.
- **Qrels**: MIRACL's relevance judgments mapping each query to its relevant passages.

> Note: the English corpus was indexed as a subset (2 shards) because of Colab memory/storage limits, so some dev queries have relevant passages that are not in our index.

---

## 2. How the indexes were created

Indexes were built in Google Colab and saved to Google Drive so that separate engine notebooks can reuse them.

1. **Load corpus**: download the MIRACL English and Hindi passage corpora.
2. **BM25 (sparse) index**: tokenize passages and build an inverted index for keyword matching. `TODO: tool used, e.g. Pyserini/Lucene or rank_bm25, plus parameters k1, b`
3. **Dense index**: encode every passage with the **E5** multilingual embedding model and store the vectors in a vector index for nearest-neighbour search. `TODO: exact E5 checkpoint and vector index library, e.g. FAISS`
4. **Persist**: save indexes to Drive and record the **index size (MB)** of each for the results table.

The English index was built first, followed by the Hindi index.

---

## 3. Qrels and test queries

**Qrels (relevance judgments)**
- We use MIRACL's own relevance judgments, so we did not create labels ourselves.
- They are loaded once, saved to Drive, and reused by every engine notebook so all engines are scored against identical ground truth.

**Test queries**
- Queries are taken from the MIRACL **dev** set (English and Hindi).
- For the assignment we also built a **query suite of 20 hard queries per language** (English and Hindi), covering:
  - **Negation** (e.g. queries that exclude something)
  - **Homonyms** (words with multiple meanings)
  - **Temporal constraints** (time-bound questions)
- The 20 queries per language were selected from MIRACL dev queries, so relevance judgments already exist for them.
- For English, only queries whose judgments overlap our indexed subset are usable.

---

## 4. Retrieval engines

| Engine | Description |
|--------|-------------|
| **BM25 baseline** | Classic keyword/lexical retrieval over the sparse index |
| **E5 dense** | Semantic retrieval: query and passages embedded with E5, ranked by vector similarity |
| **Hybrid** | Combines BM25 and E5 scores/rankings (`TODO: fusion method, e.g. weighted sum or RRF`) |
| **BM25-Translate** | Translates the query (cross-lingual setup) and then retrieves with BM25 (`TODO: translation model used`) |

---

## 5. Evaluation

Every engine is evaluated against the MIRACL **qrels** on the same queries.

| Metric | What it measures |
|--------|------------------|
| **P@5** | Fraction of the top 5 results that are relevant |
| **R@100** | Fraction of all relevant passages found in the top 100 |
| **MAP** | Mean average precision across queries |
| **NDCG@5** | Ranking quality in the top 5, rewarding relevant results ranked higher |
| **Latency (ms)** | Average time per query |
| **Index size (MB)** | Disk size of the index |


## 6. Project structure

```
.
├── notebooks/
│   ├── 01_build_indexes.ipynb      # corpus loading + BM25 and dense index creation
│   ├── 02_bm25.ipynb               # BM25 engine
│   ├── 03_dense_e5.ipynb           # E5 dense engine
│   ├── 04_hybrid.ipynb             # hybrid engine
│   ├── 05_bm25_translate.ipynb     # translate + BM25 engine
│   └── 06_evaluation.ipynb         # metrics against qrels
├── data/                           # queries and qrels (stored on Drive)
├── indexes/                        # BM25 and dense indexes (stored on Drive)
└── README.md
```

---

## 7. Setup and usage

1. Open the notebooks in **Google Colab** and mount Google Drive.
2. Run the index-building notebook (English first, then Hindi).
3. Run each engine notebook, then the evaluation notebook to produce the results table.

---

## Limitations

- The English corpus is a **~1M passage subset**, so recall-style metrics (especially R@100) can be understated for queries whose relevant passages were not indexed.
- Only a portion of the English dev queries (230 of 799) are covered by the indexed subset.
- The 20-query hard suite is small and meant for qualitative error analysis, not statistically strong conclusions.

## Acknowledgements

- [MIRACL](https://github.com/project-miracl/miracl) dataset
- E5 multilingual embedding models
