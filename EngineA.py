"""
Engine A: BM25 (Pyserini / Lucene) for the MIRACL en / hi indexes. Local version of the
Engine A notebook (no Colab / Drive). Same engine, same metrics, same results table.

Expected local folder layout:
    <data-dir>/miracl_en_indexes/   (bm25_index/, passages.jsonl, ...)
    <data-dir>/miracl_hi_indexes/
    <data-dir>/miracl_eval/         (en/ and hi/ with eval_queries.json, eval_qrels.json, hard_*.json)
<data-dir> defaults to ./data next to this script (or env MIRACL_DATA_DIR).

Requirements:  pip install pyserini pandas numpy
               + a Java JDK (21 recommended). JAVA_HOME is auto-detected if `java` is on PATH.
               Only for the `translate` mode: pip install transformers sentencepiece sacremoses torch

Usage:
    python engine_a.py eval --lang en hi --data-dir D:/ir_data
    python engine_a.py demo --lang en --query "who discovered penicillin" --data-dir D:/ir_data
    python engine_a.py translate --data-dir D:/ir_data       # eyeball Hindi->English query translation

eval prints the table and saves results/bm25_results_local.csv (a "_local" name, so it does not
overwrite the bm25_results.csv you got from the Colab run; change it with --out).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BM25_K1, BM25_B = 0.9, 0.4
BASE = PACKS = None


def configure(data_dir=None):
    global BASE, PACKS
    root = data_dir or os.environ.get("MIRACL_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data")
    BASE = os.path.join(root, "miracl_eval")
    PACKS = {"en": os.path.join(root, "miracl_en_indexes"),
             "hi": os.path.join(root, "miracl_hi_indexes")}
    return root


configure()


def ensure_java_home():
    """pyjnius (used by Pyserini) needs JAVA_HOME. If unset but `java` is on PATH,
    ask that java where it lives and set JAVA_HOME for this process."""
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


def check_files(langs, need_eval=True):
    missing = []
    for lang in langs:
        for name in ["bm25_index", "passages.jsonl"]:
            if not os.path.exists(os.path.join(PACKS[lang], name)):
                missing.append(os.path.join(PACKS[lang], name))
        if need_eval:
            for name in ["eval_queries.json", "eval_qrels.json"]:
                if not os.path.exists(os.path.join(BASE, lang, name)):
                    missing.append(os.path.join(BASE, lang, name))
    if missing:
        raise SystemExit("Missing files:\n  " + "\n  ".join(missing) +
                         "\nCheck --data-dir and the folder layout in the file header.")


# ----------------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------------
class BM25Engine:
    def __init__(self, pack, lang="en", k1=BM25_K1, b=BM25_B, load_passages=False):
        from pyserini.search.lucene import LuceneSearcher
        self.lang, self.pack = lang, pack

        # docID -> (title, text), used to display results. Big (~0.5 GB file for en), so it is
        # optional: evaluation does not need it, and without it texts are fetched by one file scan.
        self.passages = None
        if load_passages:
            self.passages = {}
            with open(f"{pack}/passages.jsonl", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    self.passages[d["id"]] = (d["title"], d["text"])

        # BM25 index (the analyzer language must match the one used at indexing time)
        self.searcher = LuceneSearcher(f"{pack}/bm25_index")
        if lang != "en":
            self.searcher.set_language(lang)
        self.searcher.set_bm25(k1=k1, b=b)
        print(f"[{lang}] loaded: {self.searcher.num_docs:,} passages")

    def texts(self, ids):
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

    def search(self, query, k=5):
        hits = self.searcher.search(query, k)
        texts = self.texts([h.docid for h in hits])
        results = []
        for rank, h in enumerate(hits, 1):
            title, text = texts.get(h.docid, ("?", "?"))
            results.append({"rank": rank, "docid": h.docid, "score": h.score,
                            "title": title, "text": text})
        return results


def show(results, query, extra=None, width=100, max_chars=500):
    print("=" * width)
    print(f"QUERY: {query}")
    if extra:
        print(extra)
    print("=" * width)
    if not results:
        print("No results.")
    for r in results:
        print(f"\n#{r['rank']}  score={r['score']:.3f}  docid={r['docid']}")
        print(f"Title: {r['title']}")
        body = r["text"][:max_chars] + ("..." if len(r["text"]) > max_chars else "")
        print(textwrap.fill(body, width=width, initial_indent="  ", subsequent_indent="  "))
    print("-" * width)


# ----------------------------------------------------------------------------
# Data / metrics / evaluation (same as the notebook)
# ----------------------------------------------------------------------------
def load_set(lang, kind):
    """kind = 'eval' (all usable dev queries) or 'hard' (the 20-query suite)"""
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
    ap = s / R
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:5], 1) if d in rel)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(R, 5) + 1))
    return p5, r100, ap, dcg / idcg


def index_size_mb(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6


def evaluate(engine, queries, qrels, label, idx_mb):
    engine.searcher.search(next(iter(queries.values())), 100)       # warm-up, not timed
    per_q, lat = {}, []
    for qid, text in queries.items():
        rel = {d for d, v in qrels[qid].items() if v > 0}
        t = time.perf_counter()
        hits = engine.searcher.search(text, 100)                    # pure retrieval time
        lat.append((time.perf_counter() - t) * 1000)
        per_q[qid] = query_metrics([h.docid for h in hits], rel)
    m = np.mean(list(per_q.values()), axis=0)
    row = {"Model": label, "n_queries": len(queries),
           "P@5": m[0], "R@100": m[1], "MAP": m[2], "NDCG@5": m[3],
           "Latency (ms)": np.mean(lat), "Index (MB)": idx_mb}
    return row, per_q


def run_eval(langs, out_name):
    rows = []
    for lang in langs:
        engine = BM25Engine(PACKS[lang], lang=lang, load_passages=False)
        idx_mb = index_size_mb(f"{PACKS[lang]}/bm25_index")
        for kind in ["eval", "hard"]:
            need = "eval_queries" if kind == "eval" else "hard_usable"
            if not os.path.exists(f"{BASE}/{lang}/{need}.json"):
                print(f"skipping {lang}/{kind}: file missing")
                continue
            q, r, meta = load_set(lang, kind)
            row, per_q = evaluate(engine, q, r, f"BM25 {lang}→{lang} [{kind}]", idx_mb)
            rows.append(row)

            # per-category breakdown for the hard suite
            if kind == "hard":
                for cat in sorted(set(meta.values())):
                    ids = [x for x in per_q if meta[x] == cat]
                    m = np.mean([per_q[x] for x in ids], axis=0)
                    rows.append({"Model": f"   └ {cat}", "n_queries": len(ids),
                                 "P@5": m[0], "R@100": m[1], "MAP": m[2], "NDCG@5": m[3]})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    out_dir = f"{BASE}/results"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/{out_name}", index=False)
    print(f"saved {out_dir}/{out_name}")
    return df


# ----------------------------------------------------------------------------
# Hindi -> English query translation (the notebook's last cell; used for BM25-Translate)
# ----------------------------------------------------------------------------
MT_NAME = "Helsinki-NLP/opus-mt-hi-en"


def load_translator():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MT_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_NAME).to(device).eval()

    def translate_hi_en(text):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        return tok.decode(out[0], skip_special_tokens=True)

    return translate_hi_en


def run_translate(n):
    translate_hi_en = load_translator()
    hq, _, hmeta = load_set("hi", "hard")
    for qid, text in list(hq.items())[:n]:
        print(f"[{hmeta[qid]}] {text}\n   → {translate_hi_en(text)}\n")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Engine A: BM25 (Pyserini)")
    ap.add_argument("mode", choices=["eval", "demo", "translate"])
    ap.add_argument("--lang", nargs="+", default=None, choices=["en", "hi"],
                    help="default: both for eval, en for demo, hi for translate")
    ap.add_argument("--data-dir", default=None,
                    help="folder holding miracl_en_indexes, miracl_hi_indexes, miracl_eval")
    ap.add_argument("--query", default=None, help="demo mode: the query text")
    ap.add_argument("--k", type=int, default=5, help="demo mode: number of results")
    ap.add_argument("--n", type=int, default=8, help="translate mode: how many queries to show")
    ap.add_argument("--out", default="bm25_results_local.csv", help="eval mode: CSV file name")
    args = ap.parse_args()

    langs = args.lang or (["en"] if args.mode == "demo" else ["en", "hi"])
    print(f"data dir: {configure(args.data_dir)}")

    if args.mode == "translate":
        check_files(["hi"])
        run_translate(args.n)
        return

    ensure_java_home()
    check_files(langs, need_eval=(args.mode == "eval"))
    if args.mode == "eval":
        run_eval(langs, args.out)
    else:
        if not args.query:
            raise SystemExit("demo mode needs --query")
        for lang in langs:
            engine = BM25Engine(PACKS[lang], lang=lang, load_passages=False)
            show(engine.search(args.query, k=args.k), args.query, extra=f"[{lang}]")


if __name__ == "__main__":
    main()