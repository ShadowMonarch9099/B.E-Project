"""Answers from the Claude API (Anthropic), with citations (decision log, 2026-10-03).

- Official `anthropic` SDK, async client, streamed with `client.beta.messages.stream`.
- Credentials from the environment (ANTHROPIC_API_KEY, read by the SDK). The key is never
  logged and never sent to the browser.
- Model SAKSHYA_LLM_MODEL (default claude-opus-5-5); effort SAKSHYA_LLM_EFFORT (default
  medium) set explicitly through output_config. No `thinking` parameter is sent: Claude
  Opus 5.5 always thinks adaptively, and a request that disables thinking is rejected.
- Each retrieved chunk is its own `document` block with a custom-content source (one text
  block per chunk), a title naming the file and page, and citations enabled. A citation's
  document_index points back to our chunk id. Citations cannot be combined with structured
  outputs, so the answer call uses none.
- Prompt caching for follow-up questions: the stable system prompt comes first, each turn
  carries its own passages followed by its question, and earlier turns are replayed
  append-only, so top-level automatic caching reads the whole earlier conversation from cache.
- Server-side refusal fallbacks are on (`fallbacks: "default"`, beta
  server-side-fallback-2026-07-01): if the model's safeguards decline a request, the API
  retries it on the model Anthropic recommends for that refusal category.
- stop_reason is checked before the content is used: "refusal" -> a polite "cannot answer",
  "max_tokens" -> the answer is marked as cut short.
"""

import logging
import os
from collections.abc import AsyncIterator

import anthropic

from .base import (AnswerEvent, AnswerRequest, AnswerResult, Citation, Passage, ProviderError,
                   Segment)
from .prompts import SYSTEM_PROMPT

log = logging.getLogger("sakshya.llm")

MODEL = os.environ.get("SAKSHYA_LLM_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("SAKSHYA_LLM_EFFORT", "medium")
MAX_TOKENS = int(os.environ.get("SAKSHYA_LLM_MAX_TOKENS", "16000"))  # thinking counts toward it too
FALLBACK_BETA = "server-side-fallback-2026-07-01"
TIMEOUT_S = 120.0

SETUP_HINT = ("Answers need a Claude API key. Add ANTHROPIC_API_KEY=<your key> to backend/.env "
              "(see backend/.env.example) and restart the backend. Search, abstention and the "
              "document viewer work without it.")


def key_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def document_block(p: Passage) -> dict:
    block = {
        "type": "document",
        "source": {"type": "content", "content": [{"type": "text", "text": p.text}]},
        "title": p.title,
        "citations": {"enabled": True},
    }
    if p.context:
        block["context"] = p.context
    return block


def build_messages(request: AnswerRequest) -> tuple[list[dict], list[int]]:
    """The messages, and the chunk id of every document block in request order."""
    messages: list[dict] = []
    order: list[int] = []
    for turn in request.history:
        messages.append({"role": "user", "content": [*(document_block(p) for p in turn.passages),
                                                     {"type": "text", "text": turn.question}]})
        order.extend(p.chunk_id for p in turn.passages)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": turn.answer}]})
    last = [*(document_block(p) for p in request.passages), {"type": "text", "text": request.question}]
    if request.answer_language:
        last.append({"type": "text", "text": request.answer_language})
    messages.append({"role": "user", "content": last})
    order.extend(p.chunk_id for p in request.passages)
    return messages, order


def _all_passages(request: AnswerRequest) -> dict[int, Passage]:
    out = {p.chunk_id: p for t in request.history for p in t.passages}
    out.update({p.chunk_id: p for p in request.passages})
    return out


def _citation_chunk(c, order: list[int], passages: dict[int, Passage]) -> int | None:
    """Map a citation to our chunk id: by document_index, checked against the cited text."""
    idx = getattr(c, "document_index", None)
    cited = (getattr(c, "cited_text", "") or "").strip()
    if idx is not None and 0 <= idx < len(order):
        cid = order[idx]
        if not cited or cited[:40] in passages[cid].text:
            return cid
    for cid, p in passages.items():  # fall back to the passage that contains the cited text
        if cited and cited[:40] in p.text:
            return cid
    return order[idx] if idx is not None and 0 <= idx < len(order) else None


def parse_message(message, request: AnswerRequest, order: list[int]) -> AnswerResult:
    """Turn the final API message into segments with citations, after checking stop_reason."""
    passages = _all_passages(request)
    usage = message.usage
    usage_d = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None) or 0,
    }
    fallback_used = any(getattr(b, "type", None) == "fallback" for b in message.content) or any(
        getattr(it, "type", None) == "fallback_message" for it in (getattr(usage, "iterations", None) or []))

    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        # Discard any partial output: a declined answer is not an answer.
        return AnswerResult([], "refused", "refusal", message.model,
                            refusal_category=getattr(details, "category", None) if details else None,
                            fallback_used=fallback_used, usage=usage_d)

    segments: list[Segment] = []
    for block in message.content:
        if block.type != "text":
            continue  # thinking (empty under the default display) and fallback markers
        cits = []
        for c in block.citations or []:
            cid = _citation_chunk(c, order, passages)
            if cid is not None:
                cits.append(Citation(cid, getattr(c, "cited_text", "") or ""))
        segments.append(Segment(block.text, cits))
    status = "cut_short" if message.stop_reason == "max_tokens" else "complete"
    return AnswerResult(segments, status, message.stop_reason, message.model, fallback_used=fallback_used,
                        usage=usage_d)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = MODEL, effort: str = EFFORT, max_tokens: int = MAX_TOKENS):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self._client: anthropic.AsyncAnthropic | None = None

    def status(self) -> dict:
        return {"provider": "Claude (Anthropic API)", "configured": key_configured(), "model": self.model,
                "effort": self.effort, "fallbacks": "default"}

    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(timeout=TIMEOUT_S, max_retries=2)
        return self._client

    async def rewrite(self, sentences: list[str], instruction: str) -> tuple[list[str], str] | None:
        return None  # not implemented for Claude: a wrong-language answer is shown as written, with a note

    async def stream(self, request: AnswerRequest) -> AsyncIterator[AnswerEvent]:
        if not key_configured():
            raise ProviderError("not_configured", SETUP_HINT)
        messages, order = build_messages(request)
        try:
            async with self.client().beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                async for event in stream:
                    if event.type == "text" and event.text:
                        yield AnswerEvent("delta", text=event.text)
                final = await stream.get_final_message()
        except anthropic.CredentialsError as exc:
            raise ProviderError("not_configured", SETUP_HINT) from exc
        except anthropic.AuthenticationError as exc:
            raise ProviderError("auth", "The Claude API rejected the API key. Check ANTHROPIC_API_KEY in "
                                        "backend/.env and restart the backend.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise ProviderError("permission", "This API key is not allowed to use "
                                              f"{self.model}. Check the key's workspace and model access.") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("rate_limited", "Claude is busy right now (rate limit reached). "
                                                "Please retry in a minute.") from exc
        except anthropic.OverloadedError as exc:
            raise ProviderError("busy", "Claude is overloaded right now. Please retry in a minute.") from exc
        except anthropic.APIStatusError as exc:
            log.warning("Claude API error %s (request %s)", exc.status_code, getattr(exc, "request_id", None))
            raise ProviderError("api_error", f"The Claude API returned an error ({exc.status_code}): "
                                             f"{exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("connection", "Could not reach the Claude API. Check the internet "
                                              "connection and retry.") from exc
        yield AnswerEvent("result", result=parse_message(final, request, order))
