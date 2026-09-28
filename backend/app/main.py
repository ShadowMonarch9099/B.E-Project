"""Sakshya API: document type detection and OCR (plan section 6.1, steps 1-2).

Files are analysed in memory and never stored.
"""

import os
from contextlib import asynccontextmanager

import magika
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import catalog
from .detection import analyse, detect as detect_type, get_magika
from .ocr import OcrUnavailable, engine_status, run_ocr

MAX_FILE_MB = int(os.environ.get("SAKSHYA_MAX_FILE_MB", "200"))
MAX_FILES = int(os.environ.get("SAKSHYA_MAX_FILES", "20"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_magika()  # load the model once at startup, not on the first upload
    yield


app = FastAPI(title="Sakshya API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SAKSHYA_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "detector": {"name": "Magika", "version": magika.__version__, "model": get_magika().get_model_name()},
        "ocr": engine_status(),
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
