"""Questions and documents for an evaluation run.

Questions come from a Sakshya-Bench export (`python -m app.bench export bench.jsonl`) or
straight from the database. Gold pages are (app document id, page): the manifest identifies a
document by SHA-256, and the app stores the same SHA-256, so the two are joined on it.
"""

import json
from pathlib import Path

from sqlalchemy import Integer, cast, func, select

from app.bench import manifest as man
from app.db import Document, Page, session_scope


def load_questions(path: str | None = None, split: str | None = None, include_drafts: bool = False) -> list[dict]:
    if path:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]
    else:
        from app.bench.store import records

        rows = records(include_drafts)
    if not include_drafts:
        rows = [r for r in rows if r.get("status", "verified") == "verified"]
    if split and split != "all":
        rows = [r for r in rows if r.get("split") == split]
    return rows


def check_split(split: str | None, allow_unfrozen: bool = False) -> dict:
    """The test split may only be evaluated once it is frozen (plan section 8.2)."""
    m = man.load()
    if split == "test" and not m.get("frozen") and not allow_unfrozen:
        raise SystemExit("The test split is not frozen yet. Run `python -m app.bench freeze` first "
                         "(or pass --allow-unfrozen for a dry run that must not be reported).")
    man.check_frozen(m)
    return m


def documents() -> dict[str, dict]:
    """sha256 -> facts of the indexed app document: id, filename, scanned (OCR share >= 0.5),
    manifest id and split."""
    sha_map = man.split_of_sha(man.load())
    out = {}
    with session_scope() as s:
        rows = s.execute(select(Document.id, Document.sha256, Document.filename)
                         .where(Document.status == "indexed")).all()
        ocr = dict(s.execute(select(Page.doc_id, func.avg(cast(Page.source == "ocr", Integer)))
                             .group_by(Page.doc_id)).all())
    for doc_id, sha, filename in rows:
        manifest_id, split = sha_map.get(sha, (None, None))
        share = float(ocr.get(doc_id) or 0.0)
        out[sha] = {"doc_id": doc_id, "filename": filename, "ocr_share": round(share, 3),
                    "doc_type": "scanned" if share >= 0.5 else "native", "manifest_id": manifest_id, "split": split}
    return out


def gold_pages(record: dict, docs: dict[str, dict]) -> set[tuple[str, int]]:
    """The gold (app document id, page) pairs of a question: its primary annotation and, when
    double-annotated, the second annotator's page too."""
    out = set()
    for key in ("gold", "second"):
        g = record.get(key)
        if g and g.get("sha256") in docs and g.get("page"):
            out.add((docs[g["sha256"]]["doc_id"], int(g["page"])))
    return out
