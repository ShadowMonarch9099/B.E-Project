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
from ..retrieval.chunks import ChunkInfo, load_chunks
from .fidelity import apply_rewrite, check_and_repair, language_instruction
from .verify import Verification, verify
from ..retrieval.hybrid import GATE_TAU, TOP_K, TOP_N, WEIGHTS, Retrieval, retrieve

log = logging.getLogger("sakshya.answer")

MAX_HISTORY_TURNS = 6
ABSTAIN_TEXT = "The documents do not seem to answer this. These are the nearest pages Sakshya found."
REFUSED_TEXT = "Sorry, Sakshya cannot answer this question."
UNSUPPORTED_TEXT = ("The answer could not be supported by the cited passages, so Sakshya does not show it. "
                    "These are the nearest pages it found.")


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


def query_features(question: str, analysis=None) -> dict:
    """The cheap question features logged with every query (the router's training data)."""
    words = question.split()
    out = {"length_chars": len(question), "length_words": len(words), "script": script_of(question),
           "has_digit": any(ch.isdigit() for ch in question)}
    if analysis is not None:
        out.update(lang=analysis.lang, lang_confidence=analysis.lang_confidence, romanized=analysis.romanized,
                   code_mix=analysis.code_mix, has_number_or_id=bool(analysis.numbers),
                   views=sorted(analysis.views()))
    return out


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


ROUTING_FEATURES = ("lang", "script", "romanized", "code_mix", "has_number_or_id", "question_type",
                    "doc_main_lang", "doc_lang_mix", "scanned_share", "top_dense_ocr_conf", "shadow_ready")


def routing_view(r: Retrieval) -> dict | None:
    """What the "why this route" panel shows: the decision, its features and the probes."""
    d, a = r.decision, r.analysis
    if d is None:
        return None
    return {"route": d.route, "rule": d.rule, "reason": d.reason, "override": d.override,
            "features": {k: d.features.get(k) for k in ROUTING_FEATURES},
            "probes": {name: {"best": p.get("best"), "page": (p.get("top") or [{}])[0].get("page_no")}
                       for name, p in d.probes.items()},
            "views": a.views() if a else {}, "numbers": a.numbers if a else []}


def _answer_payload(result: AnswerResult, r: Retrieval, v: Verification, fid, chunks: dict) -> dict:
    """The verified answer: kept sentences (after any language/script repair) with citations."""
    sentences = []
    for i, sent in enumerate(v.kept):
        cits, seen = [], set()
        for c in sent.citations:
            if c.chunk_id in chunks and c.chunk_id not in seen:
                seen.add(c.chunk_id)
                cits.append(citation_view(chunks[c.chunk_id], c.cited_text))
        sentences.append({"text": fid.texts[i], "original": sent.text if fid.repair else None,
                          "devanagari": fid.devanagari[i] if fid.devanagari else None,
                          "entailment": sent.entailment, "citations": cits})
    return {"kind": "answer", "sentences": sentences,
            "removed": [{"text": x.text, "reason": x.reason} for x in v.removed],
            "fidelity": fid.as_log(), "verification": {"nli": v.nli, "kept": len(v.kept),
                                                        "removed": len(v.removed),
                                                        "numbers_fixed": [list(f) for x in v.kept
                                                                          for f in x.numbers_fixed]},
            "status": result.status, "model": result.model, "fallback_used": result.fallback_used,
            "refusal_category": result.refusal_category, "route": r.route, "gate": r.gate.as_dict(),
            "routing": routing_view(r)}


def _retrieval_view(r: Retrieval) -> dict:
    a = r.analysis
    return {"route": r.route, "gate": r.gate.as_dict(), "latency_ms": r.latency_ms, "routing": routing_view(r),
            "query": {"lang": a.lang, "romanized": a.romanized, "views": a.views(), "numbers": a.numbers}
            if a else None,
            "top": [{"chunk_id": c.chunk_id, "page_no": r.chunks[c.chunk_id].page_no,
                     "filename": r.chunks[c.chunk_id].filename, "rerank": round(c.rerank or 0, 4)}
                    for c in r.top]}


async def ask(conversation_id: str, question: str, route: str | None = None) -> AsyncIterator[tuple[str, dict]]:
    t0 = time.perf_counter()
    doc_ids, history = await to_thread.run_sync(_load, conversation_id)
    user = await to_thread.run_sync(_add_message, conversation_id, "user", question, "complete", {})
    yield "status", {"stage": "retrieving", "user_message": user}

    r = await to_thread.run_sync(lambda: retrieve(question, doc_ids, route=route))
    yield "retrieval", _retrieval_view(r)
    provider = get_provider()
    base_log = dict(conversation_id=conversation_id, doc_ids=doc_ids or [], question=question,
                    views=r.analysis.as_log() if r.analysis else {"as_typed": question},
                    features={**query_features(question, r.analysis), **(r.decision.features if r.decision else {})},
                    route=r.route,
                    route_reason=(f"[{r.decision.rule}] {r.decision.reason}" if r.decision else None),
                    route_scores=r.decision.probes if r.decision else {},
                    candidates=r.log(), reranked=[c.as_log() for c in r.top], gate=r.gate.as_dict(),
                    config={"top_k": TOP_K, "top_n": TOP_N, "tau": GATE_TAU, "weights": WEIGHTS,
                            "provider": provider.status()})

    if not r.gate.passed:
        payload = {"kind": "abstain", "reason": r.gate.reason, "gate": r.gate.as_dict(),
                   "nearest": r.nearest_pages(), "route": r.route, "routing": routing_view(r)}
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
    request = AnswerRequest(question, passages, history, answer_language=language_instruction(r.analysis))
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
                   "route": r.route, "gate": r.gate.as_dict(), "routing": routing_view(r), "query_log_id": log_id}
        msg = await to_thread.run_sync(_add_message, conversation_id, "assistant", "", "error", payload, log_id)
        yield "answer", msg
        yield "done", {"message_id": msg["id"], "query_log_id": log_id}
        return

    assert result is not None
    llm_ms = round((time.perf_counter() - t_llm) * 1000, 1)
    llm_log = {**provider.status(), "served_by": result.model, "stop_reason": result.stop_reason,
               "fallback_used": result.fallback_used, "refusal_category": result.refusal_category,
               "usage": result.usage}
    verification_log, fidelity_log = {}, {}
    if result.status == "refused":
        payload = {"kind": "answer", "sentences": [], "removed": [], "status": "refused", "model": result.model,
                   "fallback_used": result.fallback_used, "refusal_category": result.refusal_category,
                   "route": r.route, "gate": r.gate.as_dict(), "routing": routing_view(r)}
        content, status = REFUSED_TEXT, "refused"
    else:
        # Verify every sentence against its citations (N4), then the language and script (N5).
        t_v = time.perf_counter()
        texts = {p.chunk_id: p.text for t in history for p in t.passages}
        texts.update({p.chunk_id: p.text for p in passages})
        v = await to_thread.run_sync(lambda: verify(result.segments, texts))
        verification_log = v.as_log()
        latency_verify = (time.perf_counter() - t_v) * 1000
        if v.abstain:
            payload = {"kind": "abstain", "reason": v.reason, "unsupported": True, "gate": r.gate.as_dict(),
                       "nearest": r.nearest_pages(), "route": r.route, "routing": routing_view(r),
                       "removed": [{"text": x.text, "reason": x.reason} for x in v.removed]}
            content, status = UNSUPPORTED_TEXT, "abstained"
            fid = None
        else:
            t_f = time.perf_counter()
            fid = await to_thread.run_sync(lambda: check_and_repair([x.text for x in v.kept], r.analysis))
            if fid.needs_rewrite:
                instruction = language_instruction(r.analysis) or "Rewrite in the language of the question."
                try:
                    rewritten = await provider.rewrite(fid.texts, "Rewrite each sentence. " + instruction)
                except Exception as exc:  # a failed repair keeps the answer as written
                    log.warning("Rewrite failed: %s", exc)
                    rewritten = None
                fid = await to_thread.run_sync(lambda: apply_rewrite(fid, rewritten, r.analysis))
            fidelity_log = fid.as_log()
            cited = {c.chunk_id for x in v.kept for c in x.citations}
            chunks = {**r.chunks, **load_chunks([cid for cid in cited if cid not in r.chunks])}
            payload = _answer_payload(result, r, v, fid, chunks)
            content, status = " ".join(fid.texts), result.status
            latency_fidelity = (time.perf_counter() - t_f) * 1000
    latency = {**r.latency_ms, "llm_first_token": round(first_token or 0, 1), "llm_total": llm_ms,
               "total": round((time.perf_counter() - t0) * 1000, 1)}
    if verification_log:
        latency["verify"] = round(latency_verify, 1)
    if fidelity_log:
        latency["fidelity"] = round(latency_fidelity, 1)
    log_id = await to_thread.run_sync(lambda: _write_log(
        **base_log, abstained=status == "abstained", llm_called=True, llm=llm_log,
        verification=verification_log, fidelity=fidelity_log, latency_ms=latency))
    msg = await to_thread.run_sync(_add_message, conversation_id, "assistant", content, status,
                                   {**payload, "query_log_id": log_id}, log_id)
    yield "answer", msg
    yield "done", {"message_id": msg["id"], "query_log_id": log_id}
