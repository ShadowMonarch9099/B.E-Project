"""Background ingestion (plan section 6.2).

detect (at upload) -> viewer PDF -> read every page (text layer or OCR) -> store pages.

Runs as a FastAPI background task. One document is ingested at a time, because OCR already
uses several CPU cores per document. The document's status moves queued -> processing ->
indexed (or failed), with progress counted in pages. Documents left queued or processing
by a restart are picked up again when the app starts.
"""

import logging
import threading
from collections import Counter

from sqlalchemy import delete, select

from .. import storage
from ..catalog import OCR_IMAGE_LABELS
from ..db import Document, Page, session_scope
from ..db.models import utcnow
from ..inspect_file import script_mix
from ..ocr import engine_status
from . import docx_convert
from .pages import ExtractedPage, docx_paragraphs, extract_pdf, image_to_pdf

log = logging.getLogger("sakshya.ingest")

_lock = threading.Lock()
PENDING = ("queued", "processing")
OCR_KINDS = {"scanned", "legacy", "broken", "image"}  # documents whose pages all go to OCR


class Cancelled(Exception):
    """The document was deleted while it was being ingested."""


def run_ingestion(doc_id: str) -> None:
    with _lock:
        try:
            _ingest(doc_id)
        except Cancelled:
            log.info("Ingestion of %s stopped: the document was deleted", doc_id)
        except Exception as exc:
            log.exception("Ingestion failed for %s", doc_id)
            try:
                _update(doc_id, status="failed", stage=None, error=f"{type(exc).__name__}: {exc}"[:500])
            except Cancelled:
                pass


def resume_pending() -> None:
    """Re-queue documents a previous run left unfinished (called once at startup)."""
    with session_scope() as s:
        ids = s.scalars(select(Document.id).where(Document.status.in_(PENDING))
                        .order_by(Document.created_at)).all()
        for doc in s.scalars(select(Document).where(Document.id.in_(ids))):
            doc.status, doc.stage = "queued", None
    if ids:
        log.info("Resuming ingestion of %d document(s)", len(ids))
        threading.Thread(target=lambda: [run_ingestion(i) for i in ids], name="ingest-resume",
                         daemon=True).start()


def _update(doc_id: str, **fields) -> None:
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            raise Cancelled()
        for key, value in fields.items():
            setattr(doc, key, value)


def _prepare_viewer(doc: Document, notes: list[str]) -> tuple[str | None, str]:
    """(viewer PDF path relative to the upload folder, anchor kind) for a document."""
    original = storage.resolve(doc.original_path)
    if doc.label == "pdf":
        return doc.original_path, "page"
    rel = storage.viewer_relpath(doc.id)
    if doc.label in OCR_IMAGE_LABELS:
        storage.resolve(rel).write_bytes(image_to_pdf(original.read_bytes(), doc.label))
        return rel, "page"
    if doc.label == "docx":
        try:
            docx_convert.docx_to_pdf(original, storage.resolve(rel))
            return rel, "page"
        except Exception as exc:
            notes.append(f"Citations point to paragraphs, not pages: {exc}. Install LibreOffice and "
                         "re-index to cite real pages.")
            return None, "paragraph"
    raise ValueError(f"Sakshya does not ingest {doc.label} files.")


def _page_row(doc_id: str, page: ExtractedPage) -> Page:
    text, layout = page.build()
    return Page(doc_id=doc_id, page_no=page.page_no, source=page.source, ocr_reason=page.ocr_reason,
                ocr_conf=page.ocr_conf, check_reason=page.check_reason, width=page.width,
                height=page.height, text_raw=text, text_norm=text, layout=layout,
                word_count=sum(len(line) for para in layout for line in para))


def _page_stats(doc_id: str) -> dict:
    with session_scope() as s:
        rows = s.execute(select(Page.source, Page.ocr_conf, Page.check_reason, Page.word_count, Page.text_raw)
                         .where(Page.doc_id == doc_id)).all()
    sources = Counter(r.source for r in rows)
    confs = [r.ocr_conf for r in rows if r.ocr_conf is not None]
    return {
        "pages": len(rows),
        "pages_by_source": dict(sources),
        "ocr_share": round(sources.get("ocr", 0) / len(rows), 3) if rows else 0.0,
        "mean_ocr_conf": round(sum(confs) / len(confs), 3) if confs else None,
        "check_pages": sum(1 for r in rows if r.check_reason),
        "words": sum(r.word_count for r in rows),
        "scripts": script_mix("\n".join(r.text_raw for r in rows)),
    }


def _ingest(doc_id: str) -> None:
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return
        doc.status, doc.stage, doc.error = "processing", "preparing", None
        doc.progress_done = doc.progress_total = 0
        s.execute(delete(Page).where(Page.doc_id == doc_id))  # re-ingest from scratch; chunks cascade
        s.flush()
        s.expunge(doc)

    notes: list[str] = []
    viewer_path, anchor = _prepare_viewer(doc, notes)
    first_stage = "ocr" if doc.kind in OCR_KINDS else "reading"
    _update(doc_id, viewer_path=viewer_path, anchor=anchor, notes=notes, stage=first_stage)

    if anchor == "paragraph":
        pages = iter([docx_paragraphs(storage.resolve(doc.original_path))])
        _update(doc_id, progress_total=1, page_count=1)
    else:
        ocr_ok = engine_status()["available"]
        if not ocr_ok:
            notes.append("OCR is not available, so pages without a usable text layer were left empty.")
        pages = extract_pdf(storage.resolve(viewer_path), ocr_ok,
                            source_for_text="docx" if doc.label == "docx" else "native",
                            on_total=lambda n: _update(doc_id, progress_total=n, page_count=n))

    done = 0
    for page in pages:
        with session_scope() as s:
            row = s.get(Document, doc_id)
            if row is None:
                raise Cancelled()
            s.add(_page_row(doc_id, page))
            done += 1
            row.progress_done = done
            row.stage = "ocr" if page.source == "ocr" else "reading"

    _update(doc_id, status="indexed", stage=None, indexed_at=utcnow(), notes=notes, stats=_page_stats(doc_id))
