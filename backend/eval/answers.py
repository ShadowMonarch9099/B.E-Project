"""Answer-level evaluation (plan section 9.1, Answers / Abstention / Language fidelity).

Each question goes through the app's real answer path (app/answer/service.py: retrieval, gate,
answer model, citation check, script fidelity) in a fresh conversation. This calls the answer
provider, so it can cost money: `estimate` runs first, and nothing is sent without --yes.

Measured automatically: page-level citation precision and recall, answer-level abstention
precision and recall, the share of answer sentences the verifier removed (no valid citation),
language and script match of the model's answer before repair (and whether repair fixed it),
latency per stage, tokens and the provider that served each answer. Human correctness (3-point
scale), fluency and the unsupported-claim rate need people: `answers` writes a rating sheet
(CSV) with one row per answered question for that.
"""

import asyncio
import csv
import os
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select

from app.db import Chunk, Conversation, QueryLog, session_scope
from app.llm import get_provider
from app.llm.prompts import SYSTEM_PROMPT_NUMBERED
from app.retrieval.hybrid import GATE_TAU, TOP_N

from . import data, metrics

# Prices in US$ per million tokens, for the estimate only. Set them from the provider's price
# page for the model in use; free tiers cost nothing but are rate-limited.
PRICE_IN = os.environ.get("SAKSHYA_PRICE_IN")
PRICE_OUT = os.environ.get("SAKSHYA_PRICE_OUT")
OUTPUT_TOKENS = 300  # a 1-4 sentence answer
REASONING_TOKENS = 800  # extra output of reasoning models such as gpt-oss (a guess, replaced by usage)
CHARS_PER_TOKEN = 3.0  # rough, for Devanagari and English mixed; the logged usage gives the real count


def estimate(questions: list[dict], docs: dict[str, dict], reasoning: bool = True) -> dict:
    """Tokens and cost of answering these questions once, from the documents' mean chunk size
    (the E5 tokenizer's count, a stand-in for the provider's tokenizer)."""
    doc_ids = {d["doc_id"] for d in docs.values() if d["manifest_id"]}
    with session_scope() as s:
        mean_chunk = s.scalar(select(func.avg(Chunk.token_count)).where(Chunk.doc_id.in_(doc_ids))) or 300
    per_input = int(len(SYSTEM_PROMPT_NUMBERED) / CHARS_PER_TOKEN + TOP_N * (float(mean_chunk) + 25) + 60)
    per_output = OUTPUT_TOKENS + (REASONING_TOKENS if reasoning else 0)
    n = len(questions)
    out = {"questions": n, "input_tokens_per_question": per_input, "output_tokens_per_question": per_output,
           "input_tokens": per_input * n, "output_tokens": per_output * n,
           "note": "Upper bound: questions that the evidence gate stops are not sent at all.",
           "provider": get_provider().status()}
    if PRICE_IN and PRICE_OUT:
        cost = (per_input * n * float(PRICE_IN) + per_output * n * float(PRICE_OUT)) / 1e6
        out.update(price_in=float(PRICE_IN), price_out=float(PRICE_OUT), cost_usd=round(cost, 4))
    else:
        out["cost_usd"] = None
        out["price_note"] = ("Set SAKSHYA_PRICE_IN and SAKSHYA_PRICE_OUT (US$ per million tokens, from the "
                             "provider's price page) to get a cost.")
    return out


async def _ask_one(question: str, doc_id: str | None, route: str | None) -> dict:
    from app.answer.service import ask

    with session_scope() as s:
        conv = Conversation(doc_id=doc_id, title="[eval] " + question[:100])
        s.add(conv)
        s.flush()
        conv_id = conv.id
    final, done = None, None
    async for kind, payload in ask(conv_id, question, route=route):
        if kind == "answer":
            final = payload
        elif kind == "done":
            done = payload
    log = {}
    if done and done.get("query_log_id"):
        with session_scope() as s:
            row = s.get(QueryLog, done["query_log_id"])
            if row is not None:
                log = {"llm": row.llm or {}, "latency_ms": row.latency_ms or {}, "verification": row.verification or {},
                       "fidelity": row.fidelity or {}, "abstained": row.abstained, "llm_called": row.llm_called}
    return {"message": final or {}, "log": log}


def score(q: dict, res: dict, docs: dict[str, dict]) -> dict:
    msg = res["message"]
    payload = msg.get("payload") or {}
    kind = payload.get("kind")
    gold = data.gold_pages(q, docs)
    cited = {(c["doc_id"], c["page_no"]) for s in payload.get("sentences") or [] for c in s.get("citations") or []}
    answered = kind == "answer" and payload.get("status") in ("complete", "cut_short")
    precision, recall = metrics.citation_pr(cited, gold) if answered else (None, None)
    v = res["log"].get("verification") or {}
    f = res["log"].get("fidelity") or {}
    return {"id": q["id"], "question": q["question"], "slice": q.get("slice"), "lang": q.get("lang"),
            "answerable": bool(q.get("answerable")), "kind": kind, "status": msg.get("status"),
            "answered": answered, "abstained": kind == "abstain", "error": payload.get("code") if kind == "error" else None,
            "answer": msg.get("content"), "cited_pages": sorted([list(p) for p in cited]),
            "gold": sorted([list(p) for p in gold]), "citation_precision": precision, "citation_recall": recall,
            "sentences": v.get("sentences"), "removed": (v.get("sentences") or 0) - (v.get("kept") or 0),
            "lang_match": f.get("lang_match"), "script_match": f.get("script_match"), "repair": f.get("repair"),
            "served_by": (res["log"].get("llm") or {}).get("served_by"),
            "fallback_used": (res["log"].get("llm") or {}).get("fallback_used"),
            "usage": (res["log"].get("llm") or {}).get("usage") or {}, "latency_ms": res["log"].get("latency_ms") or {}}


def run(questions: list[dict], scope: str = "document", route: str | None = None, pause_s: float = 0.0,
        progress=print) -> list[dict]:
    docs = data.documents()
    rows = []
    for i, q in enumerate(questions, start=1):
        base = docs.get(q.get("sha256") or "")
        if scope == "document" and base is None:
            continue
        res = asyncio.run(_ask_one(q["question"], base["doc_id"] if scope == "document" else None, route))
        rows.append(score(q, res, docs))
        if progress:
            r = rows[-1]
            progress(f"  {i}/{len(questions)} {r['kind']}/{r['status']} cited {r['cited_pages']} gold {r['gold']}")
        if pause_s:
            import time

            time.sleep(pause_s)  # free tiers limit tokens per minute
    return rows


def summarize(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["answered"]]
    ans_ok = [r for r in answered if r["answerable"]]
    sentences = sum(r["sentences"] or 0 for r in rows)
    usage_in = sum((r["usage"] or {}).get("input_tokens") or 0 for r in rows)
    usage_out = sum((r["usage"] or {}).get("output_tokens") or 0 for r in rows)
    by_lang = {}
    for lang in sorted({r["lang"] or "unknown" for r in answered}):
        sub = [r for r in answered if (r["lang"] or "unknown") == lang]
        by_lang[lang] = {"n": len(sub), "lang_match": metrics.mean(float(r["lang_match"]) for r in sub
                                                                   if r["lang_match"] is not None),
                         "script_match": metrics.mean(float(r["script_match"]) for r in sub
                                                      if r["script_match"] is not None)}
    return {
        "n": len(rows), "answered": len(answered), "abstained": sum(r["abstained"] for r in rows),
        "errors": dict(Counter(r["error"] for r in rows if r["error"])),
        "citation_precision": metrics.mean(r["citation_precision"] for r in ans_ok),
        "citation_recall": metrics.mean(r["citation_recall"] for r in ans_ok),
        "abstention": metrics.abstention_pr([not r["answered"] for r in rows if not r["error"]],
                                            [r["answerable"] for r in rows if not r["error"]]),
        "removed_sentence_share": (sum(r["removed"] for r in rows) / sentences) if sentences else None,
        "language_by_question_lang": by_lang,
        "repairs": dict(Counter(r["repair"] for r in answered if r["repair"])),
        "served_by": dict(Counter(r["served_by"] for r in rows if r["served_by"])),
        "fallbacks": sum(1 for r in rows if r["fallback_used"]),
        "tokens": {"input": usage_in, "output": usage_out,
                   "per_answer_input": usage_in / len(answered) if answered else None,
                   "per_answer_output": usage_out / len(answered) if answered else None},
        "latency_ms": metrics.latency_summary([r["latency_ms"] for r in rows]),
        "tau": GATE_TAU,
    }


def rating_sheet(rows: list[dict], path: Path) -> None:
    """One row per answered question for human raters (plan section 9.1)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "slice", "question", "answer", "cited_pages", "gold_pages",
                    "correctness (0 wrong, 1 partly, 2 correct)", "fluency (1-3)", "unsupported sentences (count)",
                    "notes"])
        for r in rows:
            if r["answered"]:
                w.writerow([r["id"], r["slice"], r["question"], r["answer"], r["cited_pages"], r["gold"], "", "", "", ""])
