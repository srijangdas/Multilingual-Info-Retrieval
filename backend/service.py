from functools import lru_cache
from pathlib import Path
from backend.config import pack_dir, MODEL_NAME, HYBRID_ALPHA, HYBRID_METHOD, HYBRID_DEPTH
from backend.engines.bm25 import BM25Engine
from backend.engines.dense import DenseEngine
from backend.engines.hybrid import HybridEngine

class EngineRegistry:
    """Lazy-load engines once per API process. Each language index is loaded on first use."""
    def __init__(self):
        self.instances = {}

    def get(self, engine: str, lang: str):
        key = (engine, lang)
        if key in self.instances:
            return self.instances[key]
        pack = pack_dir(lang)
        if engine == "A":
            obj = BM25Engine(pack, lang)
        elif engine == "B":
            obj = DenseEngine(pack, MODEL_NAME)
        elif engine == "C":
            obj = HybridEngine(pack, lang, MODEL_NAME)
        else:
            raise ValueError("Engine must be A, B, or C")
        self.instances[key] = obj
        return obj

registry = EngineRegistry()

def run_search(engine: str, scenario: str, query: str, top_k: int,
               bm25_mode: str = "bm25_translate"):

    target_lang = "hi" if scenario == "HI-HI" else "en"
    instance = registry.get(engine, target_lang)
    if engine == "C":
        return instance.search(query, top_k, alpha=HYBRID_ALPHA,
                               method=HYBRID_METHOD, depth=max(HYBRID_DEPTH, top_k),
                               translate_bm25=(scenario == "HI-EN"))
    if engine == "A" and scenario == "HI-EN":
        if bm25_mode == "bm25_direct":
            return instance.bm25_direct(query, top_k)
        if bm25_mode == "bm25_translate":
            return instance.bm25_translate(query, top_k)
        raise ValueError("Engine A HI-EN bm25_mode must be bm25_translate or bm25_direct")
    return instance.search(query, top_k)
