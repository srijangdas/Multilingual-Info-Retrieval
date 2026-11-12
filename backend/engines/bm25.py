
import time
from pathlib import Path
from .common import attach_passages

class BM25Engine:
    """Engine A: Pyserini/Lucene BM25, k1=0.9, b=0.4 as in the repository."""
    def __init__(self, pack: Path, lang: str):
        from pyserini.search.lucene import LuceneSearcher
        index_path = pack / "bm25_index"
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")
        self.pack, self.lang = pack, lang
        self.searcher = LuceneSearcher(str(index_path))
        if lang != "en":
            self.searcher.set_language(lang)
        self.searcher.set_bm25(k1=0.9, b=0.4)

    def search(self, query: str, k: int):
        started = time.perf_counter()
        hits = self.searcher.search(query, k)
        ids = [str(h.docid) for h in hits]
        scores = [float(h.score) for h in hits]
        results = attach_passages(self.pack, ids, scores)
        return {"results": results, "latency_ms": (time.perf_counter()-started)*1000}
