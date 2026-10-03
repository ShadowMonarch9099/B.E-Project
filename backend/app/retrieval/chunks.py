"""Loading chunks with the facts retrieval, citations and the UI need."""

from dataclasses import asdict, dataclass

from sqlalchemy import select

from ..db import Chunk, Document, Page, session_scope


@dataclass
class ChunkInfo:
    id: int
    doc_id: str
    filename: str
    page_no: int
    para_idx: int
    chunk_idx: int
    anchor: str  # "page" or "paragraph"
    text: str  # text_raw, for display and for the answer model
    text_norm: str
    text_en: str | None
    lang: str | None
    script: str | None
    source: str
    ocr_conf: float | None
    check_page: bool  # the page's OCR confidence is low
    bbox: list
    page_width: float | None
    page_height: float | None

    def as_dict(self, with_text: bool = True) -> dict:
        d = asdict(self)
        d.pop("text_norm")
        if not with_text:
            d.pop("text")
            d.pop("text_en")
        return d

    def label(self) -> str:
        """How a citation names this chunk: "<file> · p. 4" or "<file> · ¶ 12"."""
        where = f"¶ {self.para_idx + 1}" if self.anchor == "paragraph" else f"p. {self.page_no}"
        return f"{self.filename} · {where}"


def load_chunks(ids: list[int]) -> dict[int, ChunkInfo]:
    if not ids:
        return {}
    with session_scope() as s:
        rows = s.execute(
            select(Chunk, Document.filename, Document.anchor, Page.width, Page.height, Page.check_reason)
            .join(Document, Document.id == Chunk.doc_id).join(Page, Page.id == Chunk.page_id)
            .where(Chunk.id.in_(list(set(ids))))).all()
    out = {}
    for c, filename, anchor, width, height, check in rows:
        out[c.id] = ChunkInfo(c.id, c.doc_id, filename, c.page_no, c.para_idx, c.chunk_idx, anchor, c.text_raw,
                              c.text_norm, c.text_en, c.lang, c.script, c.source, c.ocr_conf,
                              check == "low_confidence", c.bbox or [], width, height)
    return out
