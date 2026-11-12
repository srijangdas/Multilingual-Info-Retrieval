"""
Engine B: Dense retrieval (multilingual-e5-small + FAISS) for the MIRACL en / hi indexes.
Local version of the Engine B notebook (no Colab / Drive). Same engine, same metrics, same table.

Expected local folder layout:
    <data-dir>/miracl_en_indexes/   (dense.faiss, docids.json, passages.jsonl, ...)
    <data-dir>/miracl_hi_indexes/
    <data-dir>/miracl_eval/         (en/ and hi/ with eval_queries.json, eval_qrels.json, hard_*.json)
<data-dir> defaults to ./data next to this script (or env MIRACL_DATA_DIR).

Requirements:  pip install faiss-cpu sentence-transformers torch pandas numpy
(No Java needed for this engine.)

Modes:
    inspect   index sizes, query-set summary, and one query with its judged passages
    examples  top results for the first N queries, with similarity and the true qrels label
    eval      the results table (P@5, R@100, MAP, NDCG@5, latency, index size)
    demo      search one query of your own

Usage:
    python engine_b.py eval     --lang en hi --data-dir D:/ir_data
    python engine_b.py inspect  --lang en    --data-dir D:/ir_data
    python engine_b.py examples --lang hi --n 5 --data-dir D:/ir_data
    python engine_b.py demo     --lang en --query "who discovered penicillin" --data-dir D:/ir_data

eval saves results/dense_results_local.csv (a "_local" name, so it does not overwrite the
dense_results.csv you got from the Colab run; change it with --out).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

MODEL = "intfloat/multilingual-e5-small"
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

def check_files(langs, need_eval=True, need_passages=False):
    missing = []
    for lang in langs:
        names = ["dense.faiss", "docids.json"] + (["passages.jsonl"] if need_passages else [])
        for name in names:
            if not os.path.exists(os.path.join(PACKS[lang], name)):
                missing.append(os.path.join(PACKS[lang], name))
        if need_eval:
            for name in ["eval_queries.json", "eval_qrels.json"]:
                if not os.path.exists(os.path.join(BASE, lang, name)):
                    missing.append(os.path.join(BASE, lang, name))
    if missing:
        raise SystemExit("Missing files:\n  " + "\n  ".join(missing) +
                         "\nCheck --data-dir and the folder layout in the file header.")

def get_encoder(model_name=MODEL, max_seq_length=256):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()
    model.max_seq_length = max_seq_length
    print(f"Query encoder : {model_name} on {device}\n")
    return model

class DenseEngine:
    def __init__(self, lang, encoder, pack=None):
        import faiss
        self.lang = lang
        self.pack = pack or PACKS[lang]
        self.encoder = encoder
        self.index = faiss.read_index(f"{self.pack}/dense.faiss")
        self.docids = json.load(open(f"{self.pack}/docids.json"))
        assert self.index.ntotal == len(self.docids), f"{lang}: index and docids.json do not match"

    def search(self, query, k=100):
        t0 = time.perf_counter()
        qv = self.encoder.encode([f"query: {query}"], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
        t1 = time.perf_counter()
        scores, idx = self.index.search(qv, k)
        t2 = time.perf_counter()
        keep = [j for j, i in enumerate(idx[0]) if i >= 0]
        ids = [self.docids[idx[0][j]] for j in keep]
        return ids, scores[0][keep], (t1 - t0) * 1000, (t2 - t1) * 1000

    def index_mb(self):
        return (os.path.getsize(f"{self.pack}/dense.faiss") +
                os.path.getsize(f"{self.pack}/docids.json")) / 1e6

def fetch_texts(lang, ids):
    """scan passages.jsonl once and return {docid: (title, text)} for the wanted ids only"""
    ids, out = set(ids), {}
    with open(f"{PACKS[lang]}/passages.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in ids:
                out[d["id"]] = (d["title"], d["text"])
                if len(out) == len(ids):
                    break
    return out

def load_set(lang, kind):
    """kind='eval': all usable dev queries | kind='hard': the 20-query suite (usable ones)"""
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
    """Hindi queries paired with relevance judgments for their English source IDs."""
    root = os.path.join(BASE, "en2hi")
    q = json.load(open(f"{root}/eval_queries.json", encoding="utf-8"))
    mapping = json.load(open(f"{root}/query_id_mapping.json", encoding="utf-8"))
    english_qrels = json.load(open(f"{BASE}/en/eval_qrels.json", encoding="utf-8"))
    hindi_to_english = mapping["hindi_to_english"]
    r = {hid: english_qrels[eid] for hid, eid in hindi_to_english.items() if hid in q}
    if set(q) != set(r):
        raise ValueError("en2hi queries, English qrels, and ID mapping do not align")
    return q, r, {}

def load_sets(langs):
    sets = {}
    for lang in langs:
        for kind in ["eval", "hard"]:
            try:
                sets[(lang, kind)] = load_set(lang, kind)
            except FileNotFoundError:
                print(f"missing {lang}/{kind} files, skipping")
    if "en" in langs:
        try:
            sets[("en2hi", "eval")] = load_en2hi_set()
        except FileNotFoundError:
            print("missing en2hi/eval files, skipping")
    return sets

def query_metrics(ranked, rel):
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

def run_inspect(engines, sets):
    print(f"{'lang':<5}{'vectors':>12}{'dim':>6}{'dense.faiss (MB)':>20}{'first 3 docids':>30}")
    for lang, e in engines.items():
        mb = os.path.getsize(f"{e.pack}/dense.faiss") / 1e6
        print(f"{lang:<5}{e.index.ntotal:>12,}{e.index.d:>6}{mb:>20,.0f}{str(e.docids[:3]):>30}")

    print(f"\n{'set':<10}{'#queries':>9}{'avg relevant/query':>20}{'avg judged-0/query':>20}")
    for (lang, kind), (q, r, meta) in sets.items():
        rel = [sum(v > 0 for v in r[x].values()) for x in q]
        non = [sum(v == 0 for v in r[x].values()) for x in q]
        print(f"{lang + '/' + kind:<10}{len(q):>9}{np.mean(rel):>20.2f}{np.mean(non):>20.2f}")

    for lang in engines:
        q, r, _ = sets[(lang, "eval")]
        qid = next(iter(q))
        texts = fetch_texts(lang, r[qid].keys())
        print(f"\n{'=' * 80}\n[{lang}] QUERY {qid}: {q[qid]}\n{'=' * 80}")
        for docid, lab in r[qid].items():
            t, x = texts.get(docid, ("?", "?"))
            print(f"  label={lab} ({'RELEVANT' if lab else 'judged NOT relevant'}) | {docid} | {t}\n"
                  f"      {x[:150]}...")

def run_examples(engines, sets, n=5, k=5):
    for lang, eng in engines.items():
        q, r, _ = sets[(lang, "eval")]
        picked = list(q)[:n]
        runs = {x: eng.search(q[x]) for x in picked}
        need = {d for x in picked for d in runs[x][0][:k]} | \
               {d for x in picked for d, v in r[x].items() if v > 0}
        texts = fetch_texts(lang, need)

        for x in picked:
            ids, sc, _, _ = runs[x]
            rel = {d for d, v in r[x].items() if v > 0}
            print(f"\n{'=' * 90}\n[{lang}→{lang}] QUERY {x}: {q[x]}   (relevant in qrels: {len(rel)})\n{'=' * 90}")
            for rank, (d, s) in enumerate(zip(ids[:k], sc[:k]), 1):
                lab = r[x].get(d)
                tag = "RELEVANT" if lab == 1 else ("judged 0 " if lab == 0 else "unjudged ")
                t, tx = texts[d]
                print(f"#{rank} sim={s:.3f} | qrels: {tag} | {d} | {t}\n      {tx[:140]}...")
            ranks = [i + 1 for i, d in enumerate(ids) if d in rel]
            print(f"  → relevant passage rank(s) in top-100: {ranks if ranks else 'none found'}")

def evaluate(eng, q, r, label):
    eng.search(next(iter(q.values())))
    per_q, enc, srch = {}, [], []
    for qid, text in q.items():
        ids, _, te, ts = eng.search(text)
        enc.append(te)
        srch.append(ts)
        per_q[qid] = query_metrics(ids, {d for d, v in r[qid].items() if v > 0})
    m = np.mean(list(per_q.values()), axis=0)
    return per_q, {"Model": label, "n_queries": len(q), "P@5": m[0], "R@100": m[1], "MAP": m[2],
                   "NDCG@5": m[3], "Latency (ms)": np.mean(enc) + np.mean(srch),
                   "Encode (ms)": np.mean(enc), "FAISS (ms)": np.mean(srch),
                   "Index (MB)": eng.index_mb()}

def run_eval(engines, sets, out_name):
    rows = []
    for (lang, kind), (q, r, meta) in sets.items():
        target_lang = "en" if lang == "en2hi" else lang
        scenario = "HI→EN" if lang == "en2hi" else f"{lang}→{lang}"
        per_q, row = evaluate(engines[target_lang], q, r, f"E5 dense {scenario} [{kind}]")
        rows.append(row)
        if kind == "hard":
            for cat in sorted(set(meta.values())):
                ids = [x for x in per_q if meta[x] == cat]
                m = np.mean([per_q[x] for x in ids], axis=0)
                rows.append({"Model": f"   └ {cat}", "n_queries": len(ids),
                             "P@5": m[0], "R@100": m[1], "MAP": m[2], "NDCG@5": m[3]})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    out_dir = f"{BASE}/results"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/{out_name}", index=False)
    print(f"saved {out_dir}/{out_name}")

def run_demo(engines, query, k):
    for lang, eng in engines.items():
        ids, sc, te, ts = eng.search(query, k)
        texts = fetch_texts(lang, ids)
        print(f"\n{'=' * 90}\n[{lang}] QUERY: {query}   (encode {te:.0f} ms, faiss {ts:.0f} ms)\n{'=' * 90}")
        for rank, (d, s) in enumerate(zip(ids, sc), 1):
            t, tx = texts.get(d, ("?", "?"))
            print(f"#{rank} sim={s:.3f} | {d} | {t}\n      {tx[:200]}...")

def main():
    ap = argparse.ArgumentParser(description="Engine B: dense retrieval (E5 + FAISS)")
    ap.add_argument("mode", choices=["inspect", "examples", "eval", "demo"])
    ap.add_argument("--lang", nargs="+", default=None, choices=["en", "hi"],
                    help="default: both for eval/inspect/examples, en for demo")
    ap.add_argument("--data-dir", default=None,
                    help="folder holding miracl_en_indexes, miracl_hi_indexes, miracl_eval")
    ap.add_argument("--query", default=None, help="demo mode: the query text")
    ap.add_argument("--k", type=int, default=5, help="demo/examples: number of results shown")
    ap.add_argument("--n", type=int, default=5, help="examples mode: how many queries")
    ap.add_argument("--out", default="dense_results_local.csv", help="eval mode: CSV file name")
    args = ap.parse_args()

    langs = args.lang or (["en"] if args.mode == "demo" else ["en", "hi"])
    print(f"data dir: {configure(args.data_dir)}")
    check_files(langs, need_eval=(args.mode != "demo"),
                need_passages=(args.mode in ("inspect", "examples", "demo")))
    if args.mode == "demo" and not args.query:
        raise SystemExit("demo mode needs --query")

    encoder = get_encoder()
    engines = {lang: DenseEngine(lang, encoder) for lang in langs}
    sets = load_sets(langs) if args.mode != "demo" else {}

    if args.mode == "inspect":
        run_inspect(engines, sets)
    elif args.mode == "examples":
        run_examples(engines, sets, n=args.n, k=args.k)
    elif args.mode == "eval":
        run_eval(engines, sets, args.out)
    else:
        run_demo(engines, args.query, args.k)

if __name__ == "__main__":
    main()
