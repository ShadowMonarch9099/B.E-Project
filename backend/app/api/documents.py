"""Documents: upload and ingest, list, status, pages, the original file, delete (plan section 6.8)."""

from pathlib import PurePath

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config, storage
from ..catalog import OCR_IMAGE_LABELS
from ..db import Chunk, Document, Page, get_session, is_ready
from ..detection import analyse
from ..ingest.pipeline import PENDING, run_ingestion

router = APIRouter(prefix="/api/documents", tags=["documents"])

INGESTIBLE = {"pdf", "docx"} | OCR_IMAGE_LABELS


def require_db() -> None:
    if not is_ready():
        raise HTTPException(status_code=503, detail=(
            "The database is not reachable, so documents cannot be stored. Start it with "
            "`docker compose up -d db` (or set SAKSHYA_DATABASE_URL) and restart the backend."))


def _kind(detection: dict) -> str | None:
    label = detection["type"]["label"]
    if label == "pdf":
        return detection.get("pdf_text_layer")
    if label == "docx":
        return "docx"
    return "image" if label in OCR_IMAGE_LABELS else None


def summary(doc: Document, chunks: int | None = None) -> dict:
    det = doc.detection or {}
    return {
        "id": doc.id,
        "filename": doc.filename,
        "size_bytes": doc.size_bytes,
        "label": doc.label,
        "stamp": (det.get("type") or {}).get("stamp", doc.label.upper()),
        "kind": doc.kind,
        "status": doc.status,
        "stage": doc.stage,
        "progress": {"done": doc.progress_done, "total": doc.progress_total},
        "page_count": doc.page_count,
        "anchor": doc.anchor,
        "has_viewer": doc.viewer_path is not None,
        "error": doc.error,
        "notes": doc.notes or [],
        "stats": doc.stats or {},
        "chunks": chunks,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "indexed_at": doc.indexed_at.isoformat() if doc.indexed_at else None,
    }


def _chunk_counts(s: Session, ids: list[str]) -> dict[str, int]:
    if not ids:
        return {}
    rows = s.execute(select(Chunk.doc_id, func.count()).where(Chunk.doc_id.in_(ids)).group_by(Chunk.doc_id))
    return dict(rows.all())


def _get(s: Session, doc_id: str) -> Document:
    doc = s.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="No such document.")
    return doc


@router.post("", dependencies=[Depends(require_db)])
def upload(background: BackgroundTasks, file: UploadFile = File(...), s: Session = Depends(get_session)):
    """Store a PDF, DOCX or image and start ingesting it. Same bytes again -> the existing document."""
    name = file.filename or "untitled"
    received = storage.receive(file.file, config.MAX_FILE_BYTES)
    if received.too_large:
        storage.discard(received)
        raise HTTPException(status_code=413, detail=f"File is larger than {config.MAX_FILE_MB} MB.")

    existing = s.scalar(select(Document).where(Document.sha256 == received.sha256))
    if existing is not None:
        storage.discard(received)
        return {"duplicate": True, "document": summary(existing), "detection": existing.detection}

    data = received.path.read_bytes()
    detection = analyse(data, name)
    del data
    label = detection["type"]["label"]
    if label not in INGESTIBLE or detection["ingestion"]["status"] != "in_scope":
        storage.discard(received)
        return JSONResponse(status_code=422, content={
            "detail": "Detected, but not ingested: " + detection["ingestion"]["route"],
            "detection": detection})

    doc = Document(filename=name, sha256=received.sha256, size_bytes=received.size, label=label,
                   kind=_kind(detection), detection=detection, original_path="", status="queued",
                   notes=[], stats={})
    s.add(doc)
    s.flush()
    ext = PurePath(name).suffix.lower().lstrip(".") or label
    doc.original_path = storage.keep(received, doc.id, ext)
    s.commit()
    background.add_task(run_ingestion, doc.id)
    return JSONResponse(status_code=201, content={"duplicate": False, "document": summary(doc),
                                                  "detection": detection})


@router.get("", dependencies=[Depends(require_db)])
def list_documents(s: Session = Depends(get_session)) -> dict:
    docs = s.scalars(select(Document).order_by(Document.created_at.desc())).all()
    counts = _chunk_counts(s, [d.id for d in docs])
    return {"documents": [summary(d, counts.get(d.id, 0)) for d in docs]}


@router.get("/{doc_id}", dependencies=[Depends(require_db)])
def get_document(doc_id: str, s: Session = Depends(get_session)) -> dict:
    doc = _get(s, doc_id)
    pages = s.execute(select(Page.page_no, Page.source, Page.ocr_conf, Page.ocr_reason, Page.check_reason,
                             Page.word_count, Page.width, Page.height, Page.lang, Page.script)
                      .where(Page.doc_id == doc_id).order_by(Page.page_no)).all()
    return {**summary(doc, _chunk_counts(s, [doc_id]).get(doc_id, 0)), "detection": doc.detection,
            "pages": [dict(p._mapping) for p in pages]}


@router.get("/{doc_id}/pages", dependencies=[Depends(require_db)])
def get_pages(doc_id: str, s: Session = Depends(get_session)) -> dict:
    """Every page's text, source and OCR confidence (the OCR panel)."""
    _get(s, doc_id)
    pages = s.scalars(select(Page).where(Page.doc_id == doc_id).order_by(Page.page_no)).all()
    return {"pages": [{"page_no": p.page_no, "source": p.source, "ocr_conf": p.ocr_conf,
                       "ocr_reason": p.ocr_reason, "check_reason": p.check_reason,
                       "word_count": p.word_count, "text": p.text_raw, "lang": p.lang, "script": p.script}
                      for p in pages]}


@router.get("/{doc_id}/file", dependencies=[Depends(require_db)])
def get_file(doc_id: str, original: bool = False, s: Session = Depends(get_session)):
    """The PDF the viewer opens (or, with ?original=true, the file as uploaded)."""
    doc = _get(s, doc_id)
    rel = doc.original_path if original or not doc.viewer_path else doc.viewer_path
    path = storage.resolve(rel)
    if not path.exists():
        raise HTTPException(status_code=404, detail="The stored file is missing.")
    is_pdf = rel.endswith(".pdf")
    filename = doc.filename if original or doc.label == "pdf" else PurePath(doc.filename).stem + ".pdf"
    return FileResponse(path, media_type="application/pdf" if is_pdf else "application/octet-stream",
                        filename=filename, content_disposition_type="inline")


@router.post("/{doc_id}/reindex", dependencies=[Depends(require_db)])
def reindex(doc_id: str, background: BackgroundTasks, s: Session = Depends(get_session)) -> dict:
    doc = _get(s, doc_id)
    if doc.status in PENDING:
        raise HTTPException(status_code=409, detail="The document is already being indexed.")
    # Detection rules improve over time, so look at the stored file again.
    detection = analyse(storage.resolve(doc.original_path).read_bytes(), doc.filename)
    doc.detection, doc.kind = detection, _kind(detection)
    doc.status, doc.stage, doc.error = "queued", None, None
    s.commit()
    background.add_task(run_ingestion, doc.id)
    return summary(doc)


@router.delete("/{doc_id}", status_code=204, dependencies=[Depends(require_db)])
def delete_document(doc_id: str, s: Session = Depends(get_session)) -> None:
    doc = _get(s, doc_id)
    s.delete(doc)
    s.commit()
    storage.remove(doc_id)
