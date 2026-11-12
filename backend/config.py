import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("MIRACL_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
MODEL_NAME = os.getenv("E5_MODEL", "intfloat/multilingual-e5-small")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
HYBRID_METHOD = os.getenv("HYBRID_METHOD", "minmax")
HYBRID_DEPTH = int(os.getenv("HYBRID_DEPTH", "100"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "100"))

def pack_dir(lang: str) -> Path:
    return DATA_DIR / f"miracl_{lang}_indexes"

def eval_dir(lang: str) -> Path:
    return DATA_DIR / "miracl_eval" / lang
