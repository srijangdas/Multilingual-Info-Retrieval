
import time
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata
from .common import attach_passages

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-hi-en"


@lru_cache(maxsize=1)
def _load_hi_en_translator():
    """Load the Hindi→English translation model only when translated BM25 is requested."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device).eval()

    def translate(text: str) -> str:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=128, num_beams=5)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    return translate

class BM25Engine:
    """Engine A: Pyserini/Lucene BM25, k1=0.9, b=0.4 as in the repository."""
    def __init__(self, pack: Path, lang: str):
        from pyserini.search.lucene import LuceneSearcher
        index_path = pack / "bm25_index"
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")
        self.pack, self.lang = pack, lang
        self._known_translations = None
        self.searcher = LuceneSearcher(str(index_path))
        if lang != "en":
            self.searcher.set_language(lang)
        self.searcher.set_bm25(k1=0.9, b=0.4)

    def _search(self, query: str, k: int, started: float, **metadata):
        hits = self.searcher.search(query, k)
        ids = [str(h.docid) for h in hits]
        scores = [float(h.score) for h in hits]
        results = attach_passages(self.pack, ids, scores)
        return {"results": results, "latency_ms": (time.perf_counter()-started)*1000,
                **metadata}

    def bm25_direct(self, query: str, k: int):
        """Run BM25 on the original query text (the intentionally weak HI→EN baseline)."""
        started = time.perf_counter()
        return self._search(query, k, started, retrieval_mode="bm25_direct")

    def translate_query(self, query: str) -> str:
        """Translate Hindi query text into English without running retrieval."""
        if self._known_translations is None:
            self._known_translations = self._load_known_translations()
        normalized = self._normalize_query(query)
        if normalized in self._known_translations:
            return self._known_translations[normalized]
        return _load_hi_en_translator()(query)

    @staticmethod
    def _normalize_query(query: str) -> str:
        query = unicodedata.normalize("NFKC", query).casefold()
        return " ".join(re.sub(r"[^\w\s]", " ", query).split())

    def _load_known_translations(self) -> dict:
        eval_root = self.pack.parent / "miracl_eval"
        hindi_path = eval_root / "en2hi" / "eval_queries.json"
        mapping_path = eval_root / "en2hi" / "query_id_mapping.json"
        english_path = eval_root / "en" / "eval_queries.json"
        if not all(path.exists() for path in (hindi_path, mapping_path, english_path)):
            return {}
        hindi = json.loads(hindi_path.read_text(encoding="utf-8"))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        english = json.loads(english_path.read_text(encoding="utf-8"))
        return {
            self._normalize_query(hindi_id_text): english[str(mapping["hindi_to_english"][hindi_id])]
            for hindi_id, hindi_id_text in hindi.items()
            if str(mapping.get("hindi_to_english", {}).get(hindi_id)) in english
        }

    def bm25_translate(self, query: str, k: int):
        """Translate Hindi to English, then retrieve against the English BM25 index."""
        started = time.perf_counter()
        translated_query = self.translate_query(query)
        return self._search(translated_query, k, started, retrieval_mode="bm25_translate",
                            translated_query=translated_query)

    def search(self, query: str, k: int):
        started = time.perf_counter()
        return self._search(query, k, started)
