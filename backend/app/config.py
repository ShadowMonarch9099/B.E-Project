"""Settings, read once from the environment.

`backend/.env` is loaded first if it exists (see `.env.example`); variables already set in
the shell win. Secrets such as ANTHROPIC_API_KEY are read here or by the SDK and are never
logged or sent to the browser.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env", override=False)
# Windows without Developer Mode cannot symlink; the Hugging Face cache still works, so don't nag.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Uploads
MAX_FILE_MB = _int("SAKSHYA_MAX_FILE_MB", 200)
MAX_FILES = _int("SAKSHYA_MAX_FILES", 20)
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Storage. The database holds documents, pages, chunks, embeddings, conversations and
# query logs; the original files live under DATA_DIR/uploads.
DATABASE_URL = os.environ.get(
    "SAKSHYA_DATABASE_URL", "postgresql+psycopg://sakshya:sakshya@localhost:5433/sakshya"
)
DATA_DIR = Path(os.environ.get("SAKSHYA_DATA_DIR", REPO_DIR / "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"

# Ingestion
OCR_WORKERS = _int("SAKSHYA_OCR_WORKERS", 4)
