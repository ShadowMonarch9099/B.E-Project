"""Conversations about a document; answers stream as Server-Sent Events (plan section 6.8)."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..answer.service import NotFound, ask, message_view
from ..db import Conversation, Document, Message, get_session
from .documents import require_db

log = logging.getLogger("sakshya.api")
router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class NewConversation(BaseModel):
    doc_id: str | None = None  # None = ask across every indexed document


class NewMessage(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


def conversation_view(c: Conversation, messages: list[Message] | None = None) -> dict:
    out = {"id": c.id, "doc_id": c.doc_id, "title": c.title,
           "created_at": c.created_at.isoformat() if c.created_at else None}
    if messages is not None:
        out["messages"] = [message_view(m) for m in messages]
    return out


@router.post("", status_code=201, dependencies=[Depends(require_db)])
def create(body: NewConversation, s: Session = Depends(get_session)) -> dict:
    if body.doc_id is not None:
        doc = s.get(Document, body.doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="No such document.")
        if doc.status != "indexed":
            raise HTTPException(status_code=409, detail="The document is still being indexed.")
    c = Conversation(doc_id=body.doc_id)
    s.add(c)
    s.flush()
    return conversation_view(c, [])


@router.get("", dependencies=[Depends(require_db)])
def list_conversations(doc_id: str | None = Query(None), s: Session = Depends(get_session)) -> dict:
    stmt = select(Conversation).order_by(Conversation.created_at.desc())
    if doc_id is not None:
        stmt = stmt.where(Conversation.doc_id == doc_id)
    return {"conversations": [conversation_view(c) for c in s.scalars(stmt).all()]}


@router.get("/{conversation_id}", dependencies=[Depends(require_db)])
def get_conversation(conversation_id: str, s: Session = Depends(get_session)) -> dict:
    c = s.get(Conversation, conversation_id)
    if c is None:
        raise HTTPException(status_code=404, detail="No such conversation.")
    msgs = s.scalars(select(Message).where(Message.conversation_id == c.id).order_by(Message.id)).all()
    return conversation_view(c, msgs)


@router.delete("/{conversation_id}", status_code=204, dependencies=[Depends(require_db)])
def delete_conversation(conversation_id: str, s: Session = Depends(get_session)) -> None:
    c = s.get(Conversation, conversation_id)
    if c is not None:
        s.delete(c)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/{conversation_id}/messages", dependencies=[Depends(require_db)])
async def post_message(conversation_id: str, body: NewMessage):
    """Ask a question. The response is a text/event-stream (see app/answer/service.py)."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Ask a question.")

    async def events():
        try:
            async for event, data in ask(conversation_id, question):
                yield _sse(event, data)
        except NotFound:
            yield _sse("error", {"code": "not_found", "message": "No such conversation."})
        except Exception as exc:  # retrieval or storage failure: tell the browser, keep the server up
            log.exception("Answering failed")
            yield _sse("error", {"code": "internal", "message": f"Answering failed ({type(exc).__name__})."})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
