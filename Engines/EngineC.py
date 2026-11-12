"""
Engine C: Hybrid retrieval (BM25 + dense E5) for the MIRACL en / hi indexes.

Built from the Engine A (BM25 / Pyserini) and Engine B (E5 + FAISS) notebooks.
Same index packs, same query/qrels files, same metrics, same Index (MB) / latency
definitions, so the numbers are directly comparable with the A and B tables.

    S_hybrid = alpha * S_bm25_norm + (1 - alpha) * S_dense_norm

Pipeline per query
    1. BM25 top-`depth`   (Pyserini / Lucene, k1=0.9, b=0.4 as in Engine A)
    2. Dense top-`depth`  (multilingual-e5-small + FAISS, "query: " prefix as in Engine B)
    3. Normalise each list (min-max, or weighted RRF), union the candidates, combine, sort.
       A passage missing from one list gets 0 from that list.

alpha is tuned ONLY on a validation split of the dev queries. The 20-query "hard"
suite is never used for tuning (it is taken from dev queries, so those ids are
removed from the tuning pool), and it is reported separately.

Expected local folder layout (download these three folders from Google Drive):
    <data-dir>/miracl_en_indexes/   (bm25_index/, docids.json, passages.jsonl, dense.faiss)
    <data-dir>/miracl_hi_indexes/   (same four items)
    <data-dir>/miracl_eval/         (en/ and hi/ with eval_queries.json, eval_qrels.json, hard_*.json)
<data-dir> defaults to ./data next to this script; change it with --data-dir.

Requirements:  pip install pyserini faiss-cpu sentence-transformers pandas numpy
               plus a Java JDK (21 recommended) on PATH or JAVA_HOME set (needed by Pyserini).

Usage:
    python engine_c.py tune --lang en hi --data-dir D:/ir_data     # pick alpha on validation queries
    python engine_c.py eval --lang en hi --data-dir D:/ir_data     # held-out + hard-suite table
    python engine_c.py demo --lang hi --query "..." --data-dir D:/ir_data   # uses the tuned alpha
    python engine_c.py demo --lang hi --query "..." --alpha 0.6 --data-dir D:/ir_data   # manual override

or from Python:
    import engine_c
    engine_c.configure("D:/ir_data")
    eng = engine_c.HybridEngine("en", encoder=engine_c.get_encoder())
    eng.search("who discovered penicillin", k=5, alpha=0.5)
"""
import argparse
import json
import math
import os
import random
import textwrap
import time
from functools import lru_cache

import numpy as np
import pandas as pd

E5_MODEL = "intfloat/multilingual-e5-small"
BM25_K1, BM25_B = 0.9, 0.4
DEFAULT_ALPHA = 0.5
DEFAULT_DEPTH = 100
MT_NAME = "Helsinki-NLP/opus-mt-hi-en"

BASE = PACKS = RESULTS_DIR = None

def configure(data_dir=None):
    """Point the module at your local copy of the Drive folders."""
    global BASE, PACKS, RESULTS_DIR
    root = data_dir or os.environ.get("MIRACL_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data")
    BASE = os.path.join(root, "miracl_eval")
    PACKS = {"en": os.path.join(root, "miracl_en_indexes"),
             "hi": os.path.join(root, "miracl_hi_indexes")}
    RESULTS_DIR = os.path.join(BASE, "results")
    return root

def ensure_java_home():
    """pyjnius (used by Pyserini) needs JAVA_HOME. If it is unset but `java` is on PATH,
    ask that java where it lives and set JAVA_HOME for this process."""
    import re
    import shutil
    import subprocess
    jh = os.environ.get("JAVA_HOME")
    if jh and os.path.isdir(jh):
        return jh
    java = shutil.which("java")
    if not java:
        raise SystemExit("Java not found. Pyserini needs a JDK (21 recommended). Install one "
                         "(e.g. Temurin 21) and set JAVA_HOME, then open a NEW terminal.")
    try:
        out = subprocess.run([java, "-XshowSettings:properties", "-version"],
                             capture_output=True, text=True).stderr
        m = re.search(r"java\.home = (.+)", out)
        home = m.group(1).strip() if m else None
    except OSError:
        home = None
    if not home or not os.path.isdir(home):
        raise SystemExit("Found java but could not work out JAVA_HOME. Set it manually to your JDK "
                         "folder (the one containing bin\\ and lib\\), then open a NEW terminal.")
    os.environ["JAVA_HOME"] = home
    print(f"JAVA_HOME was not set; using {home}")
    return home

def check_setup(langs):
    """Fail early with a clear message instead of a Java / FileNotFound stack trace."""
    ensure_java_home()
    missing = []
    for lang in langs:
        for name in ["bm25_index", "dense.faiss", "docids.json", "passages.jsonl"]:
            if not os.path.exists(os.path.join(PACKS[lang], name)):
                missing.append(os.path.join(PACKS[lang], name))
        for name in ["eval_queries.json", "eval_qrels.json"]:
            if not os.path.exists(os.path.join(BASE, lang, name)):
                missing.append(os.path.join(BASE, lang, name))
    if missing:
        raise SystemExit("Missing files:\n  " + "\n  ".join(missing) +
                         "\nCheck --data-dir and the folder layout in the file header.")

configure()

def load_set(lang, kind):
    """kind = 'eval' (all usable dev queries) or 'hard' (the 20-query suite, usable ones)"""
    E = f"{BASE}/{lang}"
    if kind == "eval":
        q = json.load(open(f"{E}/eval_queries.json", encoding="utf-8"))
        r = json.load(open(f"{E}/eval_qrels.json"))
        meta = {}
    else:
        usable = set(json.load(open(f"{E}/hard_usable.json")))
        q = {k: v for k, v in json.load(open(f"{E}/hard_queries.json", encoding="utf-8")).items()
             if k in usable}
        r = json.load(open(f"{E}/hard_qrels.json"))
        meta = json.load(open(f"{E}/hard_meta.json"))
    return q, r, meta

def load_en2hi_set():
    """Load Hindi eval queries and relevance judgments for their English source IDs."""
    root = f"{BASE}/en2hi"
    q = json.load(open(f"{root}/eval_queries.json", encoding="utf-8"))
    mapping = json.load(open(f"{root}/query_id_mapping.json", encoding="utf-8"))
    english_qrels = json.load(open(f"{BASE}/en/eval_qrels.json", encoding="utf-8"))
    hindi_to_english = mapping["hindi_to_english"]
    qrels = {hid: english_qrels[eid] for hid, eid in hindi_to_english.items() if hid in q}
    if set(q) != set(qrels):
        raise ValueError("en2hi queries, English qrels, and ID mapping do not align")
    return q, qrels

@lru_cache(maxsize=1)
def load_hi_en_translator():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MT_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_NAME).to(device).eval()

    def translate(query):
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=128, num_beams=5)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    return translate

def query_metrics(ranked, rel):
    """ranked: list of docids (top 100). rel: set of relevant docids. Binary relevance."""
    R = len(rel)
    p5 = sum(d in rel for d in ranked[:5]) / 5
    r100 = sum(d in rel for d in ranked[:100]) / R
    hits, s = 0, 0.0
    for i, d in enumerate(ranked[:100], 1):
        if d in rel:
            hits += 1
            s += hits / i
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:5], 1) if d in rel)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(R, 5) + 1))
    return p5, r100, s / R, dcg / idcg

METRIC_IDX = {"P@5": 0, "R@100": 1, "MAP": 2, "NDCG@5": 3}

def dir_size_mb(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6

def minmax(scores):
    """Scale to [0, 1]. If all scores are equal (e.g. a single hit) every item gets 1.0."""
    a = np.asarray(scores, dtype="float64")
    if a.size == 0:
        return a
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.ones_like(a)
    return (a - lo) / (hi - lo)

def fuse(cand, alpha, method="minmax", rrf_k=60, k=100):
    """
    cand = {"bm25": (ids, scores), "dense": (ids, scores)}  (each sorted best-first)
    Returns [(docid, fused, bm25_component, dense_component), ...] best-first, length <= k.
    method "minmax": alpha*minmax(bm25) + (1-alpha)*minmax(dense)
    method "rrf"   : alpha/(rrf_k+rank_bm25) + (1-alpha)/(rrf_k+rank_dense)
    """
    b_ids, b_sc = cand["bm25"]
    d_ids, d_sc = cand["dense"]
    if method == "minmax":
        bn = dict(zip(b_ids, minmax(b_sc)))
        dn = dict(zip(d_ids, minmax(d_sc)))
    elif method == "rrf":
        bn = {d: 1.0 / (rrf_k + r) for r, d in enumerate(b_ids, 1)}
        dn = {d: 1.0 / (rrf_k + r) for r, d in enumerate(d_ids, 1)}
    else:
        raise ValueError(f"unknown fusion method: {method}")

    union = list(b_ids) + [d for d in d_ids if d not in bn]
    scored = [(d, alpha * bn.get(d, 0.0) + (1 - alpha) * dn.get(d, 0.0), bn.get(d, 0.0), dn.get(d, 0.0))
              for d in union]
    scored.sort(key=lambda t: -t[1])
    return scored[:k]

def get_encoder(model_name=E5_MODEL, max_seq_length=256):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()
    model.max_seq_length = max_seq_length
    print(f"Query encoder: {model_name} on {device}")
    return model

class HybridEngine:
    def __init__(self, lang="en", pack=None, encoder=None, k1=BM25_K1, b=BM25_B,
                 load_passages=False):
        import faiss
        from pyserini.search.lucene import LuceneSearcher

        self.lang = lang
        self.pack = pack or PACKS[lang]
        self.encoder = encoder or get_encoder()

        self.searcher = LuceneSearcher(f"{self.pack}/bm25_index")
        if lang != "en":
            self.searcher.set_language(lang)
        self.searcher.set_bm25(k1=k1, b=b)

        self.index = faiss.read_index(f"{self.pack}/dense.faiss")
        self.docids = json.load(open(f"{self.pack}/docids.json"))
        assert self.index.ntotal == len(self.docids), \
            f"{lang}: dense.faiss and docids.json do not match"

        self.passages = None
        if load_passages:
            self.passages = {}
            with open(f"{self.pack}/passages.jsonl", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    self.passages[d["id"]] = (d["title"], d["text"])

        self._check_id_alignment()
        print(f"[{lang}] hybrid ready: {self.searcher.num_docs:,} BM25 passages, "
              f"{self.index.ntotal:,} dense vectors")

    def _check_id_alignment(self):
        """BM25 docids and docids.json must use the same id format, or fusion silently fails."""
        sample = [self.docids[0], self.docids[len(self.docids) // 2], self.docids[-1]]
        missing = [d for d in sample if self.searcher.doc(d) is None]
        if missing:
            print(f"WARNING [{self.lang}]: dense docids not found in the BM25 index, e.g. {missing}. "
                  f"Fusion will not work until both indexes use the same ids.")

    def _bm25(self, query, k):
        hits = self.searcher.search(query, k)
        return [h.docid for h in hits], [h.score for h in hits]

    def _dense(self, query, k):
        t0 = time.perf_counter()
        qv = self.encoder.encode([f"query: {query}"], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
        t1 = time.perf_counter()
        scores, idx = self.index.search(qv, k)
        t2 = time.perf_counter()
        keep = [j for j, i in enumerate(idx[0]) if i >= 0]
        ids = [self.docids[idx[0][j]] for j in keep]
        return ids, [float(scores[0][j]) for j in keep], (t1 - t0) * 1000, (t2 - t1) * 1000

    def candidates(self, query, depth=DEFAULT_DEPTH, bm25_query=None, translate_bm25=False):
        """Run both retrievers once. Cache this to try many alphas without re-retrieving."""
        t = time.perf_counter()
        if translate_bm25:
            bm25_query = load_hi_en_translator()(query)
        b_ids, b_sc = self._bm25(bm25_query if bm25_query is not None else query, depth)
        bm25_ms = (time.perf_counter() - t) * 1000
        d_ids, d_sc, enc_ms, faiss_ms = self._dense(query, depth)
        return {"bm25": (b_ids, b_sc), "dense": (d_ids, d_sc),
                "bm25_ms": bm25_ms, "encode_ms": enc_ms, "faiss_ms": faiss_ms}

    def search(self, query, k=10, alpha=DEFAULT_ALPHA, depth=DEFAULT_DEPTH, method="minmax",
               translate_bm25=False):
        cand = self.candidates(query, depth, translate_bm25=translate_bm25)
        fused = fuse(cand, alpha, method=method, k=k)
        texts = self.texts([d for d, *_ in fused])
        out = []
        for rank, (d, s, bs, ds) in enumerate(fused, 1):
            title, text = texts.get(d, ("?", "?"))
            out.append({"rank": rank, "docid": d, "score": s, "bm25_norm": bs, "dense_norm": ds,
                        "title": title, "text": text})
        return out

    def texts(self, ids):
        """{docid: (title, text)}. In-memory if load_passages=True, else one scan of passages.jsonl."""
        ids = set(ids)
        if self.passages is not None:
            return {d: self.passages[d] for d in ids if d in self.passages}
        out = {}
        with open(f"{self.pack}/passages.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in ids:
                    out[d["id"]] = (d["title"], d["text"])
                    if len(out) == len(ids):
                        break
        return out

    def index_mb(self):
        """BM25 index + dense.faiss + docids.json (what the hybrid needs on disk, excl. passage text)."""
        bm25 = dir_size_mb(f"{self.pack}/bm25_index")
        dense = (os.path.getsize(f"{self.pack}/dense.faiss") +
                 os.path.getsize(f"{self.pack}/docids.json")) / 1e6
        return bm25 + dense

def show(results, query, alpha=None, width=100, max_chars=400):
    print("=" * width)
    print(f"QUERY: {query}" + (f"   (alpha={alpha})" if alpha is not None else ""))
    print("=" * width)
    for r in results:
        print(f"\n#{r['rank']}  hybrid={r['score']:.3f}  (bm25={r['bm25_norm']:.2f}, "
              f"dense={r['dense_norm']:.2f})  docid={r['docid']}")
        print(f"Title: {r['title']}")
        body = r["text"][:max_chars] + ("..." if len(r["text"]) > max_chars else "")
        print(textwrap.fill(body, width=width, initial_indent="  ", subsequent_indent="  "))
    print("-" * width)

def split_path(lang):
    return f"{RESULTS_DIR}/hybrid_split_{lang}.json"

def alpha_path():
    return f"{RESULTS_DIR}/hybrid_alpha.json"

def make_split(lang, val_frac=0.5, seed=42):
    """Validation = random part of the dev queries minus the hard-suite ids. Test = the rest of dev."""
    q, _, _ = load_set(lang, "eval")
    try:
        hard_ids = set(load_set(lang, "hard")[0])
    except FileNotFoundError:
        hard_ids = set()
    pool = sorted(x for x in q if x not in hard_ids)
    random.Random(seed).shuffle(pool)
    val = pool[: int(len(pool) * val_frac)]
    val_set = set(val)
    test = [x for x in q if x not in val_set]
    return val, test

def tune_alpha(engine, lang, metric="NDCG@5", grid=None, method="minmax",
               depth=DEFAULT_DEPTH, val_frac=0.5, seed=42):
    grid = grid if grid is not None else [round(a, 2) for a in np.arange(0.0, 1.0001, 0.1)]
    q, r, _ = load_set(lang, "eval")
    val, test = make_split(lang, val_frac, seed)
    print(f"[{lang}] tuning alpha on {len(val)} validation queries "
          f"({len(test)} dev queries held out, hard suite excluded from tuning)")

    cache = {x: engine.candidates(q[x], depth) for x in val}
    rels = {x: {d for d, v in r[x].items() if v > 0} for x in val}

    rows = []
    for a in grid:
        per_q = [query_metrics([d for d, *_ in fuse(cache[x], a, method, k=100)], rels[x]) for x in val]
        m = np.mean(per_q, axis=0)
        rows.append({"alpha": a, "P@5": m[0], "R@100": m[1], "MAP": m[2], "NDCG@5": m[3]})
    df = pd.DataFrame(rows)
    best = float(df.loc[df[metric].idxmax(), "alpha"])
    print(df.round(4).to_string(index=False))
    print(f"[{lang}] best alpha by {metric} on validation: {best}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump({"val": val, "test": test, "seed": seed, "val_frac": val_frac},
              open(split_path(lang), "w"))
    alphas = json.load(open(alpha_path())) if os.path.exists(alpha_path()) else {}
    alphas[lang] = {"alpha": best, "method": method, "depth": depth, "tuned_on": metric}
    json.dump(alphas, open(alpha_path(), "w"), indent=2)
    df.to_csv(f"{RESULTS_DIR}/hybrid_alpha_sweep_{lang}.csv", index=False)
    return best, df

def evaluate_all(engine, lang, qs, rs, label, alpha, method, depth, idx_mb,
                 translate_bm25=False, scenario=None):
    """One retrieval pass per query. Latency is what each system would really cost:
       BM25-only = BM25 time, dense-only = encode + FAISS, hybrid = all three + fusion."""
    if not qs:
        return [], {}
    first_qid = next(iter(qs))
    engine.candidates(qs[first_qid], depth, translate_bm25=translate_bm25)
    per_q = {"bm25": {}, "dense": {}, "hybrid": {}}
    lat = {"bm25": [], "dense": [], "hybrid": []}
    for qid, text in qs.items():
        rel = {d for d, v in rs[qid].items() if v > 0}
        c = engine.candidates(text, depth, translate_bm25=translate_bm25)
        t = time.perf_counter()
        fused = fuse(c, alpha, method, k=100)
        fuse_ms = (time.perf_counter() - t) * 1000
        per_q["bm25"][qid] = query_metrics(c["bm25"][0], rel)
        per_q["dense"][qid] = query_metrics(c["dense"][0], rel)
        per_q["hybrid"][qid] = query_metrics([d for d, *_ in fused], rel)
        lat["bm25"].append(c["bm25_ms"])
        lat["dense"].append(c["encode_ms"] + c["faiss_ms"])
        lat["hybrid"].append(c["bm25_ms"] + c["encode_ms"] + c["faiss_ms"] + fuse_ms)

    source_lang, target_lang = ("hi", "en") if scenario == "en2hi" else (lang, lang)
    names = {"bm25": f"BM25 {source_lang}→{target_lang} {label}",
             "dense": f"E5 dense {source_lang}→{target_lang} {label}",
             "hybrid": f"Hybrid {source_lang}→{target_lang} {label} (α={alpha}, {method})"}
    rows = []
    for key in ["bm25", "dense", "hybrid"]:
        m = np.mean(list(per_q[key].values()), axis=0)
        rows.append({"Model": names[key], "n_queries": len(qs), "P@5": m[0], "R@100": m[1],
                     "MAP": m[2], "NDCG@5": m[3], "Latency (ms)": np.mean(lat[key]),
                     "Index (MB)": idx_mb if key == "hybrid" else None})
    return rows, per_q["hybrid"]

def run_eval(engine, lang, alpha=None, method=None, depth=None):
    saved = json.load(open(alpha_path())).get(lang) if os.path.exists(alpha_path()) else None
    if alpha is None:
        if saved is None:
            raise SystemExit(f"No tuned alpha for '{lang}'. Run `tune --lang {lang}` first, "
                             f"or pass --alpha explicitly.")
        alpha = saved["alpha"]
    method = method or (saved or {}).get("method", "minmax")
    depth = depth or (saved or {}).get("depth", DEFAULT_DEPTH)

    q, r, _ = load_set(lang, "eval")
    if not os.path.exists(split_path(lang)):
        raise SystemExit(f"No validation split for '{lang}'. Run `tune --lang {lang}` first.")
    test_ids = json.load(open(split_path(lang)))["test"]
    idx_mb = engine.index_mb()

    rows = []
    r1, _ = evaluate_all(engine, lang, {x: q[x] for x in test_ids}, r, "[held-out dev]",
                         alpha, method, depth, idx_mb)
    rows += r1

    try:
        hq, hr, hmeta = load_set(lang, "hard")
    except FileNotFoundError:
        hq = {}
    if hq:
        r2, per_q = evaluate_all(engine, lang, hq, hr, "[hard]", alpha, method, depth, idx_mb)
        rows += r2
        for cat in sorted(set(hmeta.values())):
            ids = [x for x in per_q if hmeta[x] == cat]
            m = np.mean([per_q[x] for x in ids], axis=0)
            rows.append({"Model": f"   └ hybrid {cat}", "n_queries": len(ids),
                         "P@5": m[0], "R@100": m[1], "MAP": m[2], "NDCG@5": m[3]})
    if lang == "en":
        try:
            hi_q, hi_r = load_en2hi_set()
        except FileNotFoundError:
            print("skipping en2hi/eval: translated query files missing")
        else:
            r3, _ = evaluate_all(engine, lang, hi_q, hi_r, "[en2hi eval]", alpha,
                                 method, depth, idx_mb, translate_bm25=True, scenario="en2hi")
            rows += r3
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Engine C: hybrid BM25 + dense retrieval")
    ap.add_argument("mode", choices=["tune", "eval", "demo"])
    ap.add_argument("--lang", nargs="+", default=["en", "hi"], choices=["en", "hi"])
    ap.add_argument("--alpha", type=float, default=None, help="override tuned alpha")
    ap.add_argument("--method", choices=["minmax", "rrf"], default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--metric", default="NDCG@5", choices=list(METRIC_IDX))
    ap.add_argument("--query", default=None, help="demo mode: the query text")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--data-dir", default=None,
                    help="folder holding miracl_en_indexes, miracl_hi_indexes, miracl_eval")
    args = ap.parse_args()

    print(f"data dir: {configure(args.data_dir)}")
    check_setup(args.lang)

    encoder = get_encoder()
    all_rows = []
    pd.set_option("display.width", 220)

    for lang in args.lang:
        eng = HybridEngine(lang, encoder=encoder, load_passages=False)
        if args.mode == "tune":
            tune_alpha(eng, lang, metric=args.metric, method=args.method or "minmax",
                       depth=args.depth or DEFAULT_DEPTH)
        elif args.mode == "eval":
            df = run_eval(eng, lang, args.alpha, args.method, args.depth)
            print(df.round(3).to_string(index=False))
            all_rows.append(df)
        else:
            if not args.query:
                raise SystemExit("demo mode needs --query")
            saved = (json.load(open(alpha_path())).get(lang)
                     if os.path.exists(alpha_path()) else None) or {}
            a = args.alpha if args.alpha is not None else saved.get("alpha", DEFAULT_ALPHA)
            if args.alpha is None and not saved:
                print(f"note: no tuned alpha found for '{lang}', using default {DEFAULT_ALPHA}. "
                      f"Run `tune` first.")
            show(eng.search(args.query, k=args.k, alpha=a,
                            depth=args.depth or saved.get("depth", DEFAULT_DEPTH),
                            method=args.method or saved.get("method", "minmax")),
                 args.query, alpha=a)

    if args.mode == "eval" and all_rows:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(f"{RESULTS_DIR}/hybrid_results.csv", index=False)
        print(f"saved {RESULTS_DIR}/hybrid_results.csv")

if __name__ == "__main__":
    main()
