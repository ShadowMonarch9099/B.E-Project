"""Sakshya API: document type detection (first slice of plan section 6.1).

Files are analysed in memory and never stored.
"""

import os
from contextlib import asynccontextmanager

import magika
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .detection import analyse, get_magika

MAX_FILE_MB = int(os.environ.get("SAKSHYA_MAX_FILE_MB", "50"))
MAX_FILES = int(os.environ.get("SAKSHYA_MAX_FILES", "20"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_magika()  # load the model once at startup, not on the first upload
    yield


app = FastAPI(title="Sakshya API", version="0.1.0", lifespan=lifespan)
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
