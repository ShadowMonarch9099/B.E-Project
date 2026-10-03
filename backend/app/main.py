"""Sakshya API.

- /api/detect and /api/ocr analyse a file in memory and store nothing (plan section 6.1 steps 1-2).
- /api/documents stores documents and ingests them in the background (sections 6.2 and 6.8).
"""

import logging
import os
from contextlib import asynccontextmanager

import magika
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import catalog, config
from .api import conversations, documents, search
from .db import init_db
from .db.session import status as db_status
from .detection import analyse, detect as detect_type, get_magika
from .ingest import docx_convert
from .ingest.pipeline import backfill_embeddings, resume_pending
from .llm import get_provider
from .ml import embedder, langid, reranker, translate, warmup
from .ocr import OcrUnavailable, engine_status, run_ocr

logging.basicConfig(level=os.environ.get("SAKSHYA_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("alembic").setLevel(logging.WARNING)

MAX_FILE_MB = config.MAX_FILE_MB
MAX_FILES = config.MAX_FILES
MAX_FILE_BYTES = config.MAX_FILE_BYTES


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_magika()  # load the model once at startup, not on the first upload
    warmup.start()  # the retrieval and language models load in the background
    if init_db():
        resume_pending()
        backfill_embeddings()
    yield


app = FastAPI(title="Sakshya API", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SAKSHYA_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(conversations.router)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "detector": {"name": "Magika", "version": magika.__version__, "model": get_magika().get_model_name()},
        "ocr": engine_status(),
        "database": db_status(),
        "docx_converter": {"name": "LibreOffice", **docx_convert.status()},
        "embedder": embedder.status(),
        "reranker": reranker.status(),
        "langid": langid.status(),
        "translator": translate.status(),
        "models": warmup.status(),
        "llm": get_provider().status(),
        "limits": {"max_file_mb": MAX_FILE_MB, "max_files": MAX_FILES},
    }


@app.post("/api/detect")
def detect(files: list[UploadFile] = File(...)) -> dict:
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=413, detail=f"Upload at most {MAX_FILES} files at a time.")
    results = []
    for upload in files:
        name = upload.filename or "untitled"
        data = upload.file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            results.append({"filename": name, "error": f"File is larger than {MAX_FILE_MB} MB."})
            continue
        results.append(analyse(data, name))
    return {"results": results}


@app.post("/api/ocr")
def ocr(file: UploadFile = File(...)) -> dict:
    """OCR one scanned PDF or image. PDF pages that already have a text layer are read directly."""
    name = file.filename or "untitled"
    data = file.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_FILE_MB} MB.")
    label = detect_type(data, name).label
    if label != "pdf" and label not in catalog.OCR_IMAGE_LABELS:
        raise HTTPException(status_code=422, detail="OCR works on PDFs and images only.")
    try:
        result = run_ocr(data, label)
    except OcrUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"OCR is not available: {exc}.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"filename": name, **result}
