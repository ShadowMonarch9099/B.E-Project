"""Sakshya-Bench annotation API (plan section 8): questions, second annotations, export, agreement."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..bench import manifest as man
from ..bench import store
from .documents import require_db

router = APIRouter(prefix="/api/bench", tags=["bench"], dependencies=[Depends(require_db)])


class Gold(BaseModel):
    doc_id: str | None = None
    page: int | None = Field(None, ge=1)
    span_start: int | None = Field(None, ge=0)
    span_end: int | None = Field(None, ge=0)


class NewQuestion(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)
    slice: Literal["same_language", "cross_language", "romanized", "unanswerable"]
    answerable: bool
    doc_id: str | None = None
    annotator: str = Field(..., min_length=1, max_length=64)
    gold: Gold | None = None
    status: Literal["draft", "verified"] = "verified"
    notes: str | None = Field(None, max_length=2000)


class SecondAnnotation(BaseModel):
    annotator: str = Field(..., min_length=1, max_length=64)
    gold: Gold | None = None


@router.post("/questions", status_code=201)
def create(body: NewQuestion) -> dict:
    try:
        return store.create_question(body.text, body.slice, body.answerable, body.doc_id, body.annotator.strip(),
                                     body.gold.model_dump() if body.gold else None, body.status, body.notes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/questions")
def list_questions(doc_id: str | None = Query(None), status: str | None = Query(None)) -> dict:
    return {"questions": store.list_questions(doc_id, status)}


@router.delete("/questions/{question_id}", status_code=204)
def delete(question_id: str) -> None:
    from ..db import BenchQuestion, session_scope

    with session_scope() as s:
        q = s.get(BenchQuestion, question_id)
        if q is not None:
            s.delete(q)


@router.post("/questions/{question_id}/verify")
def verify(question_id: str) -> dict:
    """A person has checked a drafted question and its gold span."""
    from ..db import BenchQuestion, session_scope

    with session_scope() as s:
        q = s.get(BenchQuestion, question_id)
        if q is None:
            raise HTTPException(status_code=404, detail="No such question.")
        q.status = "verified"
    return {"id": question_id, "status": "verified"}


@router.get("/second")
def second_queue(annotator: str = Query(..., min_length=1)) -> dict:
    """Questions waiting for a second annotation by this person (first annotation hidden)."""
    return {"questions": store.second_queue(annotator.strip())}


@router.post("/questions/{question_id}/second")
def second(question_id: str, body: SecondAnnotation) -> dict:
    try:
        return store.annotate_second(question_id, body.annotator.strip(), body.gold.model_dump() if body.gold else None)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/stats")
def stats() -> dict:
    """Counts by slice / language / split, and inter-annotator agreement."""
    out = store.stats()
    try:
        out["manifest"] = man.summary(man.load())
    except FileNotFoundError:
        out["manifest"] = None
    return out


@router.get("/export", response_class=PlainTextResponse)
def export(include_drafts: bool = False) -> PlainTextResponse:
    return PlainTextResponse(store.jsonl(include_drafts), media_type="application/x-ndjson",
                             headers={"Content-Disposition": 'attachment; filename="sakshya-bench.jsonl"'})


@router.get("/manifest")
def manifest() -> dict:
    try:
        return man.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="backend/bench/manifest.json is missing.") from exc
