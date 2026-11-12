
import json
import time
from pathlib import Path
import numpy as np
from .common import attach_passages

class DenseEngine:
    """Engine B: multilingual-E5 embeds Hindi queries directly for English-index search."""
    def __init__(self, pack: Path, model_name: str):
        import faiss
        from sentence_transformers import SentenceTransformer
        index_file, ids_file = pack / "dense.faiss", pack / "docids.json"
        if not index_file.exists() or not ids_file.exists():
            raise FileNotFoundError(f"Dense artifacts missing in {pack}; expected dense.faiss and docids.json")
        self.pack = pack
        self.index = faiss.read_index(str(index_file))
        with ids_file.open("r", encoding="utf-8") as f:
            self.docids = json.load(f)
        if self.index.ntotal != len(self.docids):
            raise ValueError(f"FAISS/docids mismatch in {pack}: {self.index.ntotal} vs {len(self.docids)}")
        self.encoder = SentenceTransformer(model_name)
        self.encoder.max_seq_length = 256

    def search(self, query: str, k: int):
        started = time.perf_counter()
        qv = self.encoder.encode([f"query: {query}"], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
        scores, indices = self.index.search(qv, k)
        ids, vals = [], []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                ids.append(str(self.docids[int(idx)]))
                vals.append(float(score))
        results = attach_passages(self.pack, ids, vals)
        return {"results": results, "latency_ms": (time.perf_counter()-started)*1000}
