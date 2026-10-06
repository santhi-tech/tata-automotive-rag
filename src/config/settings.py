from __future__ import annotations
 
import os
from pathlib import Path
 
from dotenv import dotenv_values
 
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
PROJECT_ROOT = SRC_ROOT.parent
 
 
def _find_env_file() -> Path | None:
    for base in (Path.cwd(), SRC_ROOT, PROJECT_ROOT):
        candidate = base / ".env"
        if candidate.is_file():
            return candidate
    return None
 
 
_ENV_FILE = _find_env_file()
_ENV = {**(dotenv_values(_ENV_FILE) if _ENV_FILE else {}), **os.environ}
 
 
def get(name: str, default: str = "") -> str:
    """Config lookup. Precedence: real environment variable > ``.env`` file > ``default``."""
    value = _ENV.get(name)
    return value if value not in (None, "") else default
 
 
# --- PostgreSQL -------------------------------------------------------------
DB_PARAMS = {
    "user": get("POSTGRES_USER") or None,
    "password": get("POSTGRES_PASSWORD") or None,
    "database": get("POSTGRES_DB") or None,
    "host": get("POSTGRES_HOST", "localhost"),
    "port": int(get("POSTGRES_PORT", "5432")),
}
 
# --- Images: ONE durable root for every category ----------------------------
# Stored paths are *relative to this directory* (e.g. "motorcycles/images/x.jpg"),
# so the database stays valid when the repo moves between machines / OSes.
IMAGE_DATA_DIR = Path(get("IMAGE_DATA_DIR") or SRC_ROOT / "data").expanduser().resolve()
 
# --- Embeddings / vector search --------------------------------------------
EMBEDDING_DIM = int(get("EMBEDDING_DIM", "768"))  # must equal VECTOR(n) in the schema
# Default "l2" matches the existing vector_l2_ops indexes and the old `<->` queries.
VECTOR_DISTANCE = get("VECTOR_DISTANCE", "l2").lower()
_OPERATORS = {"l2": "<->", "cosine": "<=>"}
if VECTOR_DISTANCE not in _OPERATORS:
    raise ValueError(f"VECTOR_DISTANCE must be one of {sorted(_OPERATORS)}, got {VECTOR_DISTANCE!r}")
VECTOR_OPERATOR = _OPERATORS[VECTOR_DISTANCE]
VECTOR_OPCLASS = "vector_l2_ops" if VECTOR_DISTANCE == "l2" else "vector_cosine_ops"
