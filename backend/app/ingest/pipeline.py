"""Background ingestion (plan section 6.2).

detect (at upload) -> viewer PDF -> read every page (text layer or OCR) -> store pages ->
page-anchored chunks -> language of each chunk -> E5 vectors -> (searchable) -> English
shadow index for the translate route.

Runs as a FastAPI background task. One document is ingested at a time, because OCR already
uses several CPU cores per document. The document's status moves queued -> processing ->
indexed (or failed), with progress counted in pages. Documents left queued or processing
by a restart are picked up again when the app starts.
"""

import logging
import os
import queue
import threading
from collections import Counter

from sqlalchemy import delete, select

from .. import storage
from ..catalog import OCR_IMAGE_LABELS
from ..db import Chunk, Document, Page, session_scope
from ..db.models import utcnow
from ..index import dense
from ..inspect_file import script_mix
from ..ocr import engine_status
from . import docx_convert, language, textcheck, tokens
from .chunking import chunk_page
from .normalize import normalize, script_of
from .pages import ExtractedPage, docx_paragraphs, extract_pdf, image_to_pdf
from ..ml.embedder import get_embedder

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


# One worker thread ingests queued documents in turn. Uploads only put an id on the queue, so a
# batch of uploads never ties up the web server's thread pool (which once froze the API while
# 40 documents waited for the ingestion lock).
_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


def _work() -> None:
    while True:
        doc_id = _queue.get()
        try:
            run_ingestion(doc_id)
        finally:
            _queue.task_done()


def enqueue(doc_id: str) -> None:
    """Queue a document for ingestion. SAKSHYA_INGEST_INLINE=1 (the tests) ingests it now instead."""
    global _worker
    if os.environ.get("SAKSHYA_INGEST_INLINE") == "1":
        run_ingestion(doc_id)
        return
    _queue.put(doc_id)
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_work, name="ingest", daemon=True)
            _worker.start()


def queue_length() -> int:
    return _queue.qsize()


def resume_pending() -> None:
    """Re-queue documents a previous run left unfinished (called once at startup)."""
    with session_scope() as s:
        ids = s.scalars(select(Document.id).where(Document.status.in_(PENDING))
                        .order_by(Document.created_at)).all()
        for doc in s.scalars(select(Document).where(Document.id.in_(ids))):
            doc.status, doc.stage = "queued", None
    if ids:
        log.info("Resuming ingestion of %d document(s)", len(ids))
        for doc_id in ids:
            enqueue(doc_id)


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
                height=page.height, text_raw=text, text_norm=normalize(text), layout=layout,
                word_count=sum(len(line) for para in layout for line in para), script=script_of(text))


def _chunk_document(doc_id: str) -> int:
    """Split every stored page into page-anchored chunks (plan section 6.2 steps 5-7)."""
    with session_scope() as s:
        pages = s.scalars(select(Page).where(Page.doc_id == doc_id).order_by(Page.page_no)).all()
        s.expunge_all()
    _update(doc_id, stage="chunking", progress_done=0, progress_total=len(pages))
    seq = 0
    for i, page in enumerate(pages, start=1):
        drafts = chunk_page(page.page_no, page.text_raw, page.layout or [])
        with session_scope() as s:
            row = s.get(Document, doc_id)
            if row is None:
                raise Cancelled()
            for d in drafts:
                s.add(Chunk(doc_id=doc_id, page_id=page.id, page_no=page.page_no, para_idx=d.para_idx,
                            chunk_idx=d.chunk_idx, seq=seq, char_start=d.char_start, char_end=d.char_end,
                            bbox=d.bbox, text_raw=d.text_raw, text_norm=normalize(d.text_raw),
                            script=script_of(d.text_raw), source=page.source,
                            ocr_conf=d.ocr_conf if page.source == "ocr" else None, token_count=d.token_count))
                seq += 1
            row.progress_done = i
    return seq


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

    check = None
    if anchor == "paragraph":
        pages = iter([docx_paragraphs(storage.resolve(doc.original_path))])
        _update(doc_id, progress_total=1, page_count=1)
    else:
        ocr_ok = engine_status()["available"]
        if not ocr_ok:
            notes.append("OCR is not available, so pages without a usable text layer were left empty.")
        if ocr_ok and doc.label != "docx" and doc.kind not in OCR_KINDS and textcheck.enabled():
            _update(doc_id, stage="checking")
            check = textcheck.check_pdf(storage.resolve(viewer_path))
            if check is not None:
                _set_stats(doc_id, text_check=check.as_dict())
                if not check.trusted:
                    notes.append(check.note())
        pages = extract_pdf(storage.resolve(viewer_path), ocr_ok,
                            source_for_text="docx" if doc.label == "docx" else "native",
                            on_total=lambda n: _update(doc_id, progress_total=n, page_count=n),
                            distrust_devanagari=check is not None and not check.trusted)

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

    chunks = _chunk_document(doc_id)
    languages = _identify(doc_id)
    _embed(doc_id)
    stats = {**_page_stats(doc_id), "chunks": chunks, "tokenizer": tokens.tokenizer_name(),
             "embedding_model": get_embedder().name, "languages": languages,
             "shadow": {"state": "pending" if language.SHADOW_ON else "off"},
             **({"text_check": check.as_dict()} if check is not None else {})}
    _update(doc_id, status="indexed", stage=None, indexed_at=utcnow(), notes=notes, stats=stats)
    if language.SHADOW_ON:
        _shadow(doc_id)


def _identify(doc_id: str) -> dict:
    """IndicLID on every chunk (plan section 6.2 step 4)."""
    _update(doc_id, stage="langid", progress_done=0, progress_total=0)
    return language.identify_chunks(doc_id, on_progress=lambda done, total: _update(
        doc_id, progress_done=done, progress_total=total))


def _set_stats(doc_id: str, **fields) -> None:
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            raise Cancelled()
        doc.stats = {**(doc.stats or {}), **fields}


def _shadow(doc_id: str) -> None:
    """Build the English shadow index; the document is already searchable meanwhile."""
    _set_stats(doc_id, shadow={"state": "building", "done": 0, "total": None})
    try:
        info = language.build_shadow(doc_id, on_progress=lambda done, total: _set_stats(
            doc_id, shadow={"state": "building", "done": done, "total": total}))
    except Cancelled:
        raise
    except Exception as exc:  # the translate route falls back to the direct route
        log.exception("English shadow index failed for %s", doc_id)
        _set_stats(doc_id, shadow={"state": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return
    _set_stats(doc_id, shadow=info)


def _embed(doc_id: str) -> None:
    """Embed every chunk with multilingual-E5 (plan section 6.2 step 8)."""
    _update(doc_id, stage="embedding", progress_done=0, progress_total=0)
    dense.embed_chunks(doc_id, on_progress=lambda done, total: _update(doc_id, progress_done=done,
                                                                       progress_total=total))


def backfill(wait: bool = False) -> None:
    """Bring indexed documents up to date with stages added later: chunk language, vectors for
    the current embedding model, and the English shadow index. Runs in the background at
    startup, so loading or downloading a model never delays the server."""

    def work() -> None:
        with session_scope() as s:
            ids = s.scalars(select(Document.id).where(Document.status == "indexed")
                            .order_by(Document.created_at)).all()
        for doc_id in ids:
            with _lock:
                try:
                    if language.needs_language(doc_id):
                        _set_stats(doc_id, languages=language.identify_chunks(doc_id))
                    if not dense.has_embeddings(doc_id):
                        log.info("Embedding %s for %s", doc_id, get_embedder().name)
                        _update(doc_id, status="processing")
                        _embed(doc_id)
                        _update(doc_id, status="indexed", stage=None)
                        _set_stats(doc_id, embedding_model=get_embedder().name)
                    if language.needs_shadow(doc_id):
                        _shadow(doc_id)
                except Cancelled:
                    pass
                except Exception as exc:
                    log.exception("Backfill failed for %s", doc_id)
                    try:
                        _update(doc_id, status="failed", stage=None, error=f"{type(exc).__name__}: {exc}"[:500])
                    except Cancelled:
                        pass

    thread = threading.Thread(target=work, name="backfill", daemon=True)
    thread.start()
    if wait:
        thread.join()


backfill_embeddings = backfill  # the name used before the shadow index existed
