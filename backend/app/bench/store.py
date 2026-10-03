"""Sakshya-Bench questions in the database: create, annotate, export, agreement (plan section 8).

Rules from the plan:
- a person writes or edits and verifies every question and gold span; a question drafted by
  an LLM is stored as "draft" and is not exported until someone verifies it;
- 15% of questions are annotated by a second person, who does not see the first annotation
  (chosen deterministically from the question id, so the share stays near 15%);
- every exported question names its split, which follows its gold document's split.
"""

import hashlib
import json
from sqlalchemy import select

from ..db import BenchAnnotation, BenchQuestion, Document, Page, session_scope
from ..ingest.chunking import _boxes
from ..ingest.normalize import script_of
from . import manifest as man
from .metrics import agreement

SLICES = ("same_language", "cross_language", "romanized", "unanswerable")
SECOND_SHARE = 15  # percent of questions that get a second annotation


def needs_second(question_id: str) -> bool:
    return int(hashlib.sha256(question_id.encode()).hexdigest()[:8], 16) % 100 < SECOND_SHARE


def _lang(text: str) -> tuple[str | None, str | None]:
    from ..retrieval.query import MODELS_ON

    script = script_of(text)
    if not MODELS_ON:
        return None, script
    from ..ml.langid import get_langid

    r = get_langid().identify(text)
    return (r.lang if r.lang != "other" else None), ("latin" if r.romanized else script)


def span_boxes(doc_id: str, page_no: int, start: int | None, end: int | None) -> tuple[str | None, list]:
    """The page text and line boxes of an evidence span (offsets into the page's text_raw)."""
    if start is None or end is None:
        return None, []
    with session_scope() as s:
        page = s.scalar(select(Page).where(Page.doc_id == doc_id, Page.page_no == page_no))
        if page is None:
            raise ValueError("No such page.")
        if not (0 <= start < end <= len(page.text_raw)):
            raise ValueError("The span is outside the page text.")
        boxes, _ = _boxes(page.layout or [], start, end)
        return page.text_raw[start:end], boxes


def annotation_view(a: BenchAnnotation, docs: dict) -> dict:
    d = docs.get(a.gold_doc_id)
    return {"annotator": a.annotator, "role": a.role, "doc_id": a.gold_doc_id,
            "filename": d.filename if d else None, "page": a.gold_page, "span_start": a.span_start,
            "span_end": a.span_end, "span_text": a.span_text, "bbox": a.bbox}


def question_view(q: BenchQuestion, docs: dict, hide_annotations: bool = False) -> dict:
    out = {"id": q.id, "text": q.text, "slice": q.slice, "lang": q.lang, "script": q.script,
           "answerable": q.answerable, "doc_id": q.doc_id, "status": q.status, "needs_second": q.needs_second,
           "created_by": q.created_by, "notes": q.notes,
           "created_at": q.created_at.isoformat() if q.created_at else None,
           "annotated_by": sorted(a.role for a in q.annotations)}
    if not hide_annotations:
        out["annotations"] = [annotation_view(a, docs) for a in sorted(q.annotations, key=lambda a: a.role)]
    return out


def _docs(s, ids) -> dict:
    ids = [i for i in set(ids) if i]
    return {d.id: d for d in s.scalars(select(Document).where(Document.id.in_(ids)))} if ids else {}


def add_annotation(s, q: BenchQuestion, role: str, annotator: str, gold: dict | None) -> BenchAnnotation:
    gold = gold or {}
    doc_id, page = gold.get("doc_id"), gold.get("page")
    if q.answerable and not (doc_id and page):
        raise ValueError("An answerable question needs a gold document and page.")
    text, boxes = (span_boxes(doc_id, page, gold.get("span_start"), gold.get("span_end"))
                   if doc_id and page else (None, []))
    a = BenchAnnotation(question_id=q.id, annotator=annotator, role=role,
                        gold_doc_id=doc_id if q.answerable else None, gold_page=page if q.answerable else None,
                        span_start=gold.get("span_start"), span_end=gold.get("span_end"),
                        span_text=text if q.answerable else None, bbox=boxes)
    s.add(a)
    return a


def create_question(text: str, slice_: str, answerable: bool, doc_id: str | None, annotator: str,
                    gold: dict | None, status: str = "verified", notes: str | None = None) -> dict:
    if slice_ not in SLICES:
        raise ValueError(f"slice must be one of {', '.join(SLICES)}")
    if (slice_ == "unanswerable") == answerable:
        raise ValueError("The unanswerable slice is exactly the questions that are not answerable.")
    lang, script = _lang(text)
    with session_scope() as s:
        q = BenchQuestion(text=text.strip(), slice=slice_, lang=lang, script=script, answerable=answerable,
                          doc_id=doc_id, status=status, created_by=annotator, notes=notes)
        s.add(q)
        s.flush()
        q.needs_second = needs_second(q.id)
        add_annotation(s, q, "primary", annotator, gold)
        s.flush()
        s.refresh(q)
        return question_view(q, _docs(s, [doc_id, (gold or {}).get("doc_id")]))


def list_questions(doc_id: str | None = None, status: str | None = None) -> list[dict]:
    with session_scope() as s:
        stmt = select(BenchQuestion).order_by(BenchQuestion.created_at.desc())
        if doc_id:
            stmt = stmt.where(BenchQuestion.doc_id == doc_id)
        if status:
            stmt = stmt.where(BenchQuestion.status == status)
        qs = s.scalars(stmt).all()
        docs = _docs(s, [a.gold_doc_id for q in qs for a in q.annotations] + [q.doc_id for q in qs])
        return [question_view(q, docs) for q in qs]


def second_queue(annotator: str) -> list[dict]:
    """Questions chosen for a second annotation that this person did not write and that have
    no second annotation yet; the first annotation is hidden."""
    with session_scope() as s:
        qs = s.scalars(select(BenchQuestion).where(BenchQuestion.needs_second.is_(True),
                                                   BenchQuestion.status == "verified",
                                                   BenchQuestion.created_by != annotator)).all()
        return [question_view(q, {}, hide_annotations=True) for q in qs
                if "second" not in {a.role for a in q.annotations}]


def annotate_second(question_id: str, annotator: str, gold: dict | None) -> dict:
    with session_scope() as s:
        q = s.get(BenchQuestion, question_id)
        if q is None:
            raise LookupError("No such question.")
        if q.created_by == annotator:
            raise ValueError("The second annotation must come from a different person.")
        if any(a.role == "second" for a in q.annotations):
            raise ValueError("This question already has a second annotation.")
        add_annotation(s, q, "second", annotator, gold)
        s.flush()
        s.refresh(q)
        return question_view(q, _docs(s, [a.gold_doc_id for a in q.annotations]))


def records(include_drafts: bool = False) -> list[dict]:
    """The exportable benchmark: one record per question, with its split."""
    try:
        sha_map = man.split_of_sha(man.load())
    except FileNotFoundError:
        sha_map = {}
    with session_scope() as s:
        qs = s.scalars(select(BenchQuestion).order_by(BenchQuestion.created_at)).all()
        docs = _docs(s, [a.gold_doc_id for q in qs for a in q.annotations] + [q.doc_id for q in qs])
        out = []
        for q in qs:
            if q.status != "verified" and not include_drafts:
                continue
            ann = {a.role: a for a in q.annotations}

            def gold(a):
                if a is None or not a.gold_doc_id:
                    return None
                d = docs.get(a.gold_doc_id)
                manifest_id = sha_map.get(d.sha256, (None, None))[0] if d else None
                return {"document": manifest_id, "sha256": d.sha256 if d else None,
                        "filename": d.filename if d else None, "page": a.gold_page, "span": a.span_text,
                        "span_start": a.span_start, "span_end": a.span_end, "annotator": a.annotator}

            base_doc = docs.get(q.doc_id) or (docs.get(ann["primary"].gold_doc_id) if "primary" in ann else None)
            split = sha_map.get(base_doc.sha256, (None, None))[1] if base_doc else None
            out.append({"id": q.id, "question": q.text, "slice": q.slice, "lang": q.lang, "script": q.script,
                        "answerable": q.answerable, "status": q.status, "split": split or "unassigned",
                        "document": (sha_map.get(base_doc.sha256, (None, None))[0] if base_doc else None),
                        "sha256": base_doc.sha256 if base_doc else None,
                        "gold": gold(ann.get("primary")), "second": gold(ann.get("second")),
                        "created_by": q.created_by,
                        "created_at": q.created_at.isoformat() if q.created_at else None})
        return out


def jsonl(include_drafts: bool = False) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records(include_drafts))


def stats() -> dict:
    recs = records(include_drafts=True)
    verified = [r for r in recs if r["status"] == "verified"]
    by = lambda key: {k: sum(1 for r in verified if r[key] == k) for k in sorted({r[key] or "unknown" for r in verified})}
    pairs = []
    with session_scope() as s:
        for q in s.scalars(select(BenchQuestion).where(BenchQuestion.status == "verified")):
            ann = {a.role: a for a in q.annotations}
            if "primary" in ann and "second" in ann:
                p, sd = ann["primary"], ann["second"]
                pairs.append(({"doc": p.gold_doc_id, "page": p.gold_page, "span_text": p.span_text},
                              {"doc": sd.gold_doc_id, "page": sd.gold_page, "span_text": sd.span_text}))
    total = len(verified)
    return {"questions": total, "drafts": len(recs) - total, "by_slice": by("slice"), "by_lang": by("lang"),
            "by_split": by("split"),
            "slice_share": {k: round(v / total, 3) for k, v in by("slice").items()} if total else {},
            "chosen_for_second": sum(1 for q in list_questions(status="verified") if q["needs_second"]),
            "agreement": agreement(pairs)}
