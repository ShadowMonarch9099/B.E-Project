"""One question, end to end (plan sections 6.4-6.7): retrieve -> evidence gate -> answer.

Yields (event, data) pairs that the API sends to the browser as Server-Sent Events:
  status     {"stage": "retrieving" | "answering"}
  retrieval  route, top pages and the gate decision
  delta      {"text": ...} streamed answer text
  answer     the stored assistant message (an answer, an abstention, or an error)
  done       {"message_id": ..., "query_log_id": ...}

Below the evidence gate's threshold the system abstains WITHOUT calling the answer model
and shows the nearest pages. Every question is written to query_logs with its features,
route, scores, gate decision, model usage and per-stage latency.
"""

import logging
import time
from collections.abc import AsyncIterator

from anyio import to_thread
from sqlalchemy import select

from ..db import Conversation, Message, QueryLog, session_scope
from ..ingest.normalize import script_of
from ..llm import get_provider
from ..llm.base import AnswerRequest, AnswerResult, Passage, ProviderError, Turn
from ..retrieval.chunks import ChunkInfo
from ..retrieval.hybrid import GATE_TAU, TOP_K, TOP_N, WEIGHTS, Retrieval, retrieve

log = logging.getLogger("sakshya.answer")

MAX_HISTORY_TURNS = 6
ABSTAIN_TEXT = "The documents do not seem to answer this. These are the nearest pages Sakshya found."
REFUSED_TEXT = "Sorry, Sakshya cannot answer this question."


class NotFound(Exception):
    pass


def passage_for(info: ChunkInfo) -> Passage:
    context = None
    if info.source == "ocr" and info.ocr_conf is not None:
        context = (f"Read from a scan with OCR (mean word confidence {info.ocr_conf:.0%}); "
                   "characters and digits may be misread.")
    return Passage(info.id, info.label(), info.text, context)


def citation_view(info: ChunkInfo, cited_text: str) -> dict:
    return {**info.as_dict(with_text=False), "cited_text": cited_text,
            "label": f"¶ {info.para_idx + 1}" if info.anchor == "paragraph" else f"p. {info.page_no}"}


def message_view(m: Message) -> dict:
    return {"id": m.id, "role": m.role, "content": m.content, "status": m.status, "payload": m.payload or {},
            "created_at": m.created_at.isoformat() if m.created_at else None}


def query_features(question: str) -> dict:
    """The cheap question features logged with every query (the router's training data)."""
    words = question.split()
    return {"length_chars": len(question), "length_words": len(words), "script": script_of(question),
            "has_digit": any(ch.isdigit() for ch in question)}


def _load(conversation_id: str) -> tuple[list[str] | None, list[Turn]]:
    with session_scope() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            raise NotFound()
        msgs = s.scalars(select(Message).where(Message.conversation_id == conversation_id)
                         .order_by(Message.id)).all()
        turns: list[Turn] = []
        pending_user: Message | None = None
        for m in msgs:
            if m.role == "user":
                pending_user = m
                continue
            usable = (m.status in ("complete", "cut_short") and (m.payload or {}).get("kind") == "answer")
            if pending_user is not None and usable and (pending_user.payload or {}).get("passages"):
                turns.append(Turn(pending_user.content,
                                  [Passage.from_dict(p) for p in pending_user.payload["passages"]], m.content))
            pending_user = None
        return ([conv.doc_id] if conv.doc_id else None), turns[-MAX_HISTORY_TURNS:]


def _add_message(conversation_id: str, role: str, content: str, status: str, payload: dict,
                 query_log_id: int | None = None) -> dict:
    with session_scope() as s:
        m = Message(conversation_id=conversation_id, role=role, content=content, status=status,
                    payload=payload, query_log_id=query_log_id)
        s.add(m)
        conv = s.get(Conversation, conversation_id)
        if conv is not None and role == "user" and not conv.title:
            conv.title = content[:120]
        s.flush()
        return message_view(m)


def _set_payload(message_id: int, payload: dict) -> None:
    with session_scope() as s:
        m = s.get(Message, message_id)
        if m is not None:
            m.payload = payload


def _write_log(**fields) -> int:
    with session_scope() as s:
        row = QueryLog(**fields)
        s.add(row)
        s.flush()
        return row.id


def _answer_payload(result: AnswerResult, r: Retrieval) -> dict:
    segments = []
    for seg in result.segments:
        cits = [citation_view(r.chunks[c.chunk_id], c.cited_text) for c in seg.citations if c.chunk_id in r.chunks]
        segments.append({"text": seg.text, "citations": cits})
    return {"kind": "answer", "segments": segments, "status": result.status, "model": result.model,
            "fallback_used": result.fallback_used, "refusal_category": result.refusal_category,
            "route": r.route, "gate": r.gate.as_dict()}


def _retrieval_view(r: Retrieval) -> dict:
    return {"route": r.route, "gate": r.gate.as_dict(), "latency_ms": r.latency_ms,
            "top": [{"chunk_id": c.chunk_id, "page_no": r.chunks[c.chunk_id].page_no,
                     "filename": r.chunks[c.chunk_id].filename, "rerank": round(c.rerank or 0, 4)}
                    for c in r.top]}


async def ask(conversation_id: str, question: str) -> AsyncIterator[tuple[str, dict]]:
    t0 = time.perf_counter()
    doc_ids, history = await to_thread.run_sync(_load, conversation_id)
    user = await to_thread.run_sync(_add_message, conversation_id, "user", question, "complete", {})
    yield "status", {"stage": "retrieving", "user_message": user}

    r = await to_thread.run_sync(retrieve, question, doc_ids)
    yield "retrieval", _retrieval_view(r)
    provider = get_provider()
    base_log = dict(conversation_id=conversation_id, doc_ids=doc_ids or [], question=question,
                    views={"as_typed": question, "normalized": r.normalized},
                    features=query_features(question), route=r.route,
                    route_reason="fixed route (rule-based router comes in phase 8)", route_scores={},
                    candidates=r.log(), reranked=[c.as_log() for c in r.top], gate=r.gate.as_dict(),
                    config={"top_k": TOP_K, "top_n": TOP_N, "tau": GATE_TAU, "weights": WEIGHTS,
                            "provider": provider.status()})

    if not r.gate.passed:
        payload = {"kind": "abstain", "reason": r.gate.reason, "gate": r.gate.as_dict(),
                   "nearest": r.nearest_pages(), "route": r.route}
        latency = {**r.latency_ms, "total": round((time.perf_counter() - t0) * 1000, 1)}
        log_id = await to_thread.run_sync(lambda: _write_log(**base_log, abstained=True, llm_called=False,
                                                             latency_ms=latency))
        msg = await to_thread.run_sync(_add_message, conversation_id, "assistant", ABSTAIN_TEXT, "abstained",
                                       {**payload, "query_log_id": log_id}, log_id)
        yield "answer", msg
        yield "done", {"message_id": msg["id"], "query_log_id": log_id}
        return

    passages = [passage_for(r.chunks[c.chunk_id]) for c in r.top]
    await to_thread.run_sync(_set_payload, user["id"], {"passages": [p.as_dict() for p in passages]})
    yield "status", {"stage": "answering"}
    request = AnswerRequest(question, passages, history)
    t_llm = time.perf_counter()
    first_token = None
    result: AnswerResult | None = None
    try:
        async for event in provider.stream(request):
            if event.kind == "delta":
                if first_token is None:
                    first_token = (time.perf_counter() - t_llm) * 1000
                yield "delta", {"text": event.text}
            elif event.kind == "result":
                result = event.result
    except ProviderError as exc:
        latency = {**r.latency_ms, "llm_total": round((time.perf_counter() - t_llm) * 1000, 1),
                   "total": round((time.perf_counter() - t0) * 1000, 1)}
        log_id = await to_thread.run_sync(lambda: _write_log(
            **base_log, abstained=False, llm_called=exc.code != "not_configured",
            llm={**provider.status(), "error_code": exc.code}, error=exc.message, latency_ms=latency))
        payload = {"kind": "error", "code": exc.code, "message": exc.message, "nearest": r.nearest_pages(),
                   "route": r.route, "gate": r.gate.as_dict(), "query_log_id": log_id}
        msg = await to_thread.run_sync(_add_message, conversation_id, "assistant", "", "error", payload, log_id)
        yield "answer", msg
        yield "done", {"message_id": msg["id"], "query_log_id": log_id}
        return

    assert result is not None
    latency = {**r.latency_ms, "llm_first_token": round(first_token or 0, 1),
               "llm_total": round((time.perf_counter() - t_llm) * 1000, 1),
               "total": round((time.perf_counter() - t0) * 1000, 1)}
    payload = _answer_payload(result, r)
    content = REFUSED_TEXT if result.status == "refused" else result.text
    status = "refused" if result.status == "refused" else result.status
    log_id = await to_thread.run_sync(lambda: _write_log(
        **base_log, abstained=False, llm_called=True,
        llm={**provider.status(), "served_by": result.model, "stop_reason": result.stop_reason,
             "fallback_used": result.fallback_used, "refusal_category": result.refusal_category,
             "usage": result.usage},
        latency_ms=latency))
    msg = await to_thread.run_sync(_add_message, conversation_id, "assistant", content, status,
                                   {**payload, "query_log_id": log_id}, log_id)
    yield "answer", msg
    yield "done", {"message_id": msg["id"], "query_log_id": log_id}
