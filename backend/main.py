
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import json
from backend.config import DATA_DIR, eval_dir, DEFAULT_TOP_K, MAX_TOP_K
from backend.service import run_search

app = FastAPI(title="Multilingual IR API", version="1.0.0",
              description="Engine A BM25, Engine B Dense E5, Engine C Hybrid.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

class SearchRequest(BaseModel):
    engine: Literal["A", "B", "C"]
    scenario: Literal["EN-EN", "HI-HI", "HI-EN"]
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    bm25_mode: Literal["bm25_translate", "bm25_direct"] = "bm25_translate"

@app.get("/health")
def health():
    return {"status": "ok", "data_dir": str(DATA_DIR), "data_dir_exists": DATA_DIR.exists()}

@app.get("/api/examples")
def examples(scenario: str = "EN-EN"):
    if scenario == "HI-EN":
        path = DATA_DIR / "miracl_eval" / "en2hi" / "eval_queries.json"
    else:
        lang = "hi" if scenario == "HI-HI" else "en"
        path = eval_dir(lang) / "eval_queries.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {"source": str(path), "queries": [{"qid": str(k), "query": v} for k, v in list(data.items())[:50]]}
    # Replace this placeholder file with your actual query examples.
    placeholder = DATA_DIR / "examples" / f"{scenario.lower().replace('-', '_')}.json"
    if placeholder.exists():
        return {"source": str(placeholder), "queries": json.loads(placeholder.read_text(encoding="utf-8"))}
    return {"source": "placeholder", "queries": [
        {"qid": "example-1", "query": "Who developed the theory of relativity?"},
        {"qid": "example-2", "query": "What are the causes of climate change?"}
    ] if scenario != "HI-HI" else [
        {"qid": "example-1", "query": "भारत का संविधान कब लागू हुआ?"},
        {"qid": "example-2", "query": "जलवायु परिवर्तन के मुख्य कारण क्या हैं?"}
    ]}

@app.post("/api/search")
def search(req: SearchRequest):
    try:
        payload = run_search(req.engine, req.scenario, req.query.strip(), req.top_k,
                             bm25_mode=req.bm25_mode)
        return {"engine": req.engine, "scenario": req.scenario, "query": req.query,
                "target_corpus": "hi" if req.scenario == "HI-HI" else "en",
                **payload}
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

@app.get("/api/evaluation/status")
def evaluation_status():
    expected = {}
    for lang in ("en", "hi"):
        folder = eval_dir(lang)
        expected[lang] = {
            "eval_queries": (folder / "eval_queries.json").exists(),
            "eval_qrels": (folder / "eval_qrels.json").exists(),
            "hard_queries": (folder / "hard_queries.json").exists(),
            "hard_qrels": (folder / "hard_qrels.json").exists(),
        }
    en2hi = DATA_DIR / "miracl_eval" / "en2hi"
    expected["en2hi"] = {
        "eval_queries": (en2hi / "eval_queries.json").exists(),
        "query_id_mapping": (en2hi / "query_id_mapping.json").exists(),
        "english_qrels": (eval_dir("en") / "eval_qrels.json").exists(),
    }
    return {"files": expected, "note": "Evaluation requires matching query and qrels files; no benchmark scores are fabricated."}
