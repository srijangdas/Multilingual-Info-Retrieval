
import time
import numpy as np
from pathlib import Path
from .bm25 import BM25Engine
from .dense import DenseEngine
from .common import attach_passages

def _minmax(values):
    a = np.asarray(values, dtype="float64")
    if not len(a):
        return a
    lo, hi = a.min(), a.max()
    return np.ones_like(a) if hi-lo < 1e-12 else (a-lo)/(hi-lo)

def fuse(b_ids, b_scores, d_ids, d_scores, alpha=0.5, method="minmax", k=10, rrf_k=60):
    if method == "rrf":
        bm = {d: 1/(rrf_k+r) for r, d in enumerate(b_ids, 1)}
        dn = {d: 1/(rrf_k+r) for r, d in enumerate(d_ids, 1)}
    else:
        bm = dict(zip(b_ids, _minmax(b_scores)))
        dn = dict(zip(d_ids, _minmax(d_scores)))
    union = list(b_ids) + [d for d in d_ids if d not in bm]
    scored = [(d, alpha*float(bm.get(d, 0))+(1-alpha)*float(dn.get(d, 0)),
               float(bm.get(d, 0)), float(dn.get(d, 0))) for d in union]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]

class HybridEngine:
    """Engine C: BM25 + E5 dense; score fusion follows EngineC.py (min-max or weighted RRF)."""
    def __init__(self, pack: Path, lang: str, model_name: str):
        self.pack = pack
        self.bm25 = BM25Engine(pack, lang)
        self.dense = DenseEngine(pack, model_name)

    def search(self, query: str, k: int, alpha=0.5, method="minmax", depth=100):
        started = time.perf_counter()
        bh = self.bm25.searcher.search(query, depth)
        b_ids = [str(h.docid) for h in bh]
        b_scores = [float(h.score) for h in bh]
        qv = self.dense.encoder.encode([f"query: {query}"], normalize_embeddings=True,
                                       convert_to_numpy=True).astype("float32")
        ds, ix = self.dense.index.search(qv, depth)
        d_ids, d_scores = [], []
        for score, idx in zip(ds[0], ix[0]):
            if idx >= 0:
                d_ids.append(str(self.dense.docids[int(idx)]))
                d_scores.append(float(score))
        fused = fuse(b_ids, b_scores, d_ids, d_scores, alpha, method, k)
        ids = [x[0] for x in fused]
        scores = [x[1] for x in fused]
        extras = {x[0]: {"bm25_component": x[2], "dense_component": x[3]} for x in fused}
        results = attach_passages(self.pack, ids, scores, extras)
        return {"results": results, "latency_ms": (time.perf_counter()-started)*1000,
                "fusion": {"alpha": alpha, "method": method, "candidate_depth": depth}}
