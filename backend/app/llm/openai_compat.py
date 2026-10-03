"""Answers from OpenAI-compatible chat APIs: Groq, OpenRouter and NVIDIA NIM (decision log,
2026-10-03, second entry).

- The official `openai` Python SDK (AsyncOpenAI) pointed at each provider's base URL; answers
  are streamed (`stream=True`) and token usage is requested in the last chunk.
- Keys come from the environment: GROQ_API_KEY, OPENROUTER_API_KEY, NVIDIA_API_KEY (put them in
  backend/.env, which is gitignored). They are never logged and never sent to the browser.
- Models are set per provider (SAKSHYA_GROQ_MODEL, SAKSHYA_OPENROUTER_MODEL, SAKSHYA_NVIDIA_MODEL);
  `python -m app.llm models` lists what each key can use.
- These APIs have no citations feature, so the passages are numbered in the user message and the
  model ends each sentence with the numbers it relies on ("... 3 lakh rupees [2]."). The markers
  are parsed back into per-sentence citations; the verifier (app/answer/verify.py) then removes
  any sentence that cites nothing, exactly as for the Claude provider.
- The chain (SAKSHYA_LLM_ORDER, default groq,openrouter,nvidia) tries each provider that has a key;
  if one fails before it has streamed any text (rate limit, outage, bad model name, no credit),
  the next one answers and the answer says which provider served it.
- finish_reason is checked: "length" -> cut short, "content_filter" -> a polite "cannot answer".
- Earlier turns are replayed append-only, so providers with automatic prefix caching can reuse them.
- Reasoning models (e.g. gpt-oss) return their reasoning in a separate field, which is ignored;
  a <think>...</think> block inside the content is stripped too.
"""

import logging
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import openai

from ..ingest.chunking import sentence_spans
from .base import AnswerEvent, AnswerRequest, AnswerResult, Citation, Passage, ProviderError, Segment
from .prompts import SYSTEM_PROMPT_NUMBERED

log = logging.getLogger("sakshya.llm")

MAX_TOKENS = int(os.environ.get("SAKSHYA_LLM_MAX_TOKENS", "4096"))  # reasoning tokens count too
TEMPERATURE = float(os.environ.get("SAKSHYA_LLM_TEMPERATURE", "0.2"))
REASONING_EFFORT = os.environ.get("SAKSHYA_LLM_REASONING_EFFORT") or None  # e.g. "low" for gpt-oss
# One attempt per provider: the chain is the retry. A provider that hangs (seen on NVIDIA NIM for
# models it lists but does not serve) then delays the next one by TIMEOUT_S, not twice that.
TIMEOUT_S = float(os.environ.get("SAKSHYA_LLM_TIMEOUT", "45"))
RETRIES = int(os.environ.get("SAKSHYA_LLM_RETRIES", "0"))


@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    base_url: str
    key_env: str
    model_env: str
    default_model: str
    headers: tuple = ()


PRESETS = {p.name: p for p in [
    Preset("groq", "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "SAKSHYA_GROQ_MODEL",
           "openai/gpt-oss-120b"),
    Preset("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
           "SAKSHYA_OPENROUTER_MODEL", "openai/gpt-oss-120b", (("X-Title", "Sakshya"),)),
    Preset("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
           "SAKSHYA_NVIDIA_MODEL", "openai/gpt-oss-20b"),  # NIM does not serve gpt-oss-120b (2026-10-04)
]}
DEFAULT_ORDER = "groq,openrouter,nvidia"

SETUP_HINT = ("Answers need an API key for Groq, OpenRouter or NVIDIA NIM. Add GROQ_API_KEY, "
              "OPENROUTER_API_KEY and/or NVIDIA_API_KEY to backend/.env (see backend/.env.example) and "
              "restart the backend. Search, abstention and the document viewer work without a key.")

# "[2]", "[1][4]", "[1, 3]", "[2-3]", "[passage 2]"; leading spaces go with the marker.
_MARKER = re.compile(r"\s*(?:\[\s*(?:passages?|source|p\.?)?\s*\d+(?:\s*(?:,|;|–|-|and)\s*\d+)*\s*\])+",
                     re.IGNORECASE)
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def passage_block(n: int, p: Passage) -> str:
    head = f"[{n}] ({p.title})"
    if p.context:
        head += f" Note: {p.context}"
    return f"{head}\n{p.text}"


def build_messages(request: AnswerRequest) -> tuple[list[dict], dict[int, int]]:
    """The chat messages, and passage number -> chunk id. Passages are numbered across the
    whole conversation (turn 1: 1-6, turn 2: 7-12, ...) so earlier turns never change."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT_NUMBERED}]
    numbers: dict[int, int] = {}

    def user_turn(question: str, passages: list[Passage]) -> dict:
        blocks = []
        for p in passages:
            n = len(numbers) + 1
            numbers[n] = p.chunk_id
            blocks.append(passage_block(n, p))
        return {"role": "user", "content": "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"}

    for turn in request.history:
        messages.append(user_turn(turn.question, turn.passages))
        messages.append({"role": "assistant", "content": turn.answer})
    last = user_turn(request.question, request.passages)
    if request.answer_language:
        last["content"] += "\n\n" + request.answer_language
    messages.append(last)
    return messages, numbers


def _expand(marker: str) -> list[int]:
    out: list[int] = []
    for group in re.findall(r"\[([^\]]*)\]", marker):
        for a, b in re.findall(r"(\d+)\s*(?:[–-]\s*(\d+))?", group):
            lo, hi = int(a), int(b or a)
            out.extend(range(lo, min(hi, lo + 20) + 1) if hi >= lo else [lo])
    return out


def _best_sentence(passage: str, sentence: str) -> str:
    """The passage sentence that shares most words with the answer sentence (shown as the cited text)."""
    want = set(w.lower() for w in _WORDS.findall(sentence))
    best, best_n = passage[:200], -1
    for s, e in sentence_spans(passage):
        n = len(want & set(w.lower() for w in _WORDS.findall(passage[s:e])))
        if n > best_n:
            best, best_n = passage[s:e], n
    return best


def strip_reasoning(text: str) -> str:
    return _THINK.sub("", text).lstrip()


def parse_cited(text: str, numbers: dict[int, int], passages: dict[int, Passage]) -> list[Segment]:
    """Answer text with [n] markers -> one segment per sentence, carrying the citations of the
    markers inside it (or right after its full stop). Markers are removed from the text."""
    clean, marks, pos = "", [], 0
    for m in _MARKER.finditer(text):
        clean += text[pos:m.start()]
        marks.append((len(clean), _expand(m.group())))
        pos = m.end()
    clean += text[pos:]
    spans, offset = [], 0
    for line in clean.split("\n"):
        spans.extend(sentence_spans(line, offset))
        offset += len(line) + 1
    if not spans:
        return [Segment(clean, [])] if clean.strip() else []
    cited: list[list[int]] = [[] for _ in spans]
    for at, nums in marks:
        i = max([k for k, (s, _) in enumerate(spans) if s <= at] or [0])
        cited[i].extend(n for n in nums if n in numbers and n not in cited[i])
    segments, prev = [], 0
    for k, (s, e) in enumerate(spans):
        end = spans[k + 1][0] if k + 1 < len(spans) else len(clean)
        body = clean[prev:end]
        sentence = clean[s:e]
        cits = [Citation(numbers[n], _best_sentence(passages[numbers[n]].text, sentence)) for n in cited[k]
                if numbers[n] in passages]
        segments.append(Segment(body, cits))
        prev = end
    return segments


REWRITE_PROMPT = ("You rewrite sentences as instructed. Keep every number, name, date and ID exactly as "
                  "written. Add nothing and leave nothing out. Reply with exactly one numbered line per "
                  "sentence, in the same order (1. ..., 2. ...), and nothing else.")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s*(.*\S)\s*$")


def parse_numbered(text: str, n: int) -> list[str] | None:
    """Lines "1. ..." to "n. ..." of a reply, in order; None unless all n are there."""
    lines: dict[int, str] = {}
    for line in strip_reasoning(text).splitlines():
        m = _NUMBERED.match(line)
        if m:
            lines.setdefault(int(m.group(1)), m.group(2))
    return [lines[i] for i in range(1, n + 1)] if all(i in lines for i in range(1, n + 1)) else None


class OpenAICompatProvider:
    def __init__(self, preset: Preset, model: str | None = None) -> None:
        self.preset = preset
        self.name = preset.name
        self.model = model or os.environ.get(preset.model_env) or preset.default_model
        self._client: openai.AsyncOpenAI | None = None

    def configured(self) -> bool:
        return bool(os.environ.get(self.preset.key_env))

    def status(self) -> dict:
        return {"provider": self.preset.label, "configured": self.configured(), "model": self.model}

    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=os.environ.get(self.preset.key_env), base_url=self.preset.base_url,
                                              timeout=TIMEOUT_S, max_retries=RETRIES,
                                              default_headers=dict(self.preset.headers) or None)
        return self._client

    def _error(self, exc: Exception) -> ProviderError:
        who = f"{self.preset.label} ({self.model})"
        if isinstance(exc, openai.AuthenticationError):
            return ProviderError("auth", f"{self.preset.label} rejected the API key. Check {self.preset.key_env} "
                                         "in backend/.env and restart the backend.")
        if isinstance(exc, openai.PermissionDeniedError):
            return ProviderError("permission", f"This {self.preset.label} key may not use {self.model}.")
        if isinstance(exc, openai.NotFoundError):
            return ProviderError("not_found", f"{self.preset.label} does not offer the model {self.model}. Set "
                                              f"{self.preset.model_env} (see `python -m app.llm models`).")
        if isinstance(exc, openai.RateLimitError):
            return ProviderError("rate_limited", f"{who} is rate-limited right now. Please retry in a minute.")
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code == 402:
                return ProviderError("payment", f"{self.preset.label} reports no credit left for {self.model}.")
            if exc.status_code >= 500:
                return ProviderError("busy", f"{who} is unavailable right now ({exc.status_code}).")
            return ProviderError("api_error", f"{who} returned an error ({exc.status_code}).")
        if isinstance(exc, openai.APIConnectionError):
            return ProviderError("connection", f"Could not reach {self.preset.label}. Check the internet connection.")
        return ProviderError("api_error", f"{who} failed: {type(exc).__name__}")

    async def rewrite(self, sentences: list[str], instruction: str) -> tuple[list[str], str] | None:
        return await _rewrite(self, sentences, instruction) if self.configured() else None

    async def _create(self, messages: list[dict], with_usage: bool):
        kwargs = dict(model=self.model, messages=messages, stream=True, max_tokens=MAX_TOKENS,
                      temperature=TEMPERATURE)
        if with_usage:
            kwargs["stream_options"] = {"include_usage": True}
        if REASONING_EFFORT:
            kwargs["reasoning_effort"] = REASONING_EFFORT
        return await self.client().chat.completions.create(**kwargs)

    async def stream(self, request: AnswerRequest) -> AsyncIterator[AnswerEvent]:
        if not self.configured():
            raise ProviderError("not_configured", SETUP_HINT)
        messages, numbers = build_messages(request)
        passages = {p.chunk_id: p for t in request.history for p in t.passages}
        passages.update({p.chunk_id: p for p in request.passages})
        text, finish, usage, served_model = "", None, {}, self.model
        in_think = False
        try:
            try:
                stream = await self._create(messages, with_usage=True)
            except openai.BadRequestError as exc:
                if "stream_options" not in str(exc):
                    raise
                stream = await self._create(messages, with_usage=False)  # a provider without usage-in-stream
            async for chunk in stream:
                served_model = chunk.model or served_model
                if chunk.usage:
                    u = chunk.usage
                    usage = {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}
                    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
                    if cached:
                        usage["cache_read_input_tokens"] = cached
                for choice in chunk.choices:
                    piece = choice.delta.content or ""
                    if choice.finish_reason:
                        finish = choice.finish_reason
                    if not piece:
                        continue
                    text += piece
                    if "<think>" in piece:
                        in_think = True
                    if in_think:
                        in_think = "</think>" not in piece
                        continue
                    yield AnswerEvent("delta", text=piece)
        except openai.OpenAIError as exc:
            log.warning("%s error: %s", self.preset.label, type(exc).__name__)
            raise self._error(exc) from exc
        model_label = f"{served_model} via {self.preset.label}"
        if finish == "content_filter":
            result = AnswerResult([], "refused", finish, model_label, refusal_category="content_filter",
                                         usage=usage)
        else:
            segments = parse_cited(strip_reasoning(text), numbers, passages)
            result = AnswerResult(segments, "cut_short" if finish == "length" else "complete", finish,
                                         model_label, usage=usage)
        yield AnswerEvent("result", result=result)


async def _rewrite(provider: "OpenAICompatProvider", sentences: list[str],
                   instruction: str) -> tuple[list[str], str] | None:
    body = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    try:
        r = await provider.client().chat.completions.create(
            model=provider.model, max_tokens=MAX_TOKENS, temperature=0.0,
            messages=[{"role": "system", "content": REWRITE_PROMPT},
                      {"role": "user", "content": f"{instruction}\n\n{body}"}])
    except openai.OpenAIError as exc:
        log.warning("%s rewrite failed: %s", provider.preset.label, type(exc).__name__)
        return None
    out = parse_numbered(r.choices[0].message.content or "", len(sentences)) if r.choices else None
    return (out, f"{r.model or provider.model} via {provider.preset.label}") if out else None


FALLBACK_CODES = {"rate_limited", "busy", "connection", "api_error", "not_found", "payment", "auth", "permission"}


class ChainProvider:
    """Each configured provider in turn; the next one answers when one fails before streaming."""

    name = "chain"

    def __init__(self, order: str | None = None) -> None:
        names = [n.strip() for n in (order or os.environ.get("SAKSHYA_LLM_ORDER", DEFAULT_ORDER)).split(",")
                 if n.strip()]
        unknown = [n for n in names if n not in PRESETS]
        if unknown:
            raise ValueError(f"Unknown provider(s) in SAKSHYA_LLM_ORDER: {', '.join(unknown)}")
        self.providers = [OpenAICompatProvider(PRESETS[n]) for n in names]

    def configured(self) -> list[OpenAICompatProvider]:
        return [p for p in self.providers if p.configured()]

    def status(self) -> dict:
        ready = self.configured()
        first = ready[0] if ready else (self.providers[0] if self.providers else None)
        return {"provider": first.preset.label if first else None, "configured": bool(ready),
                "model": first.model if first else None,
                "chain": [p.status() for p in self.providers]}

    async def rewrite(self, sentences: list[str], instruction: str) -> tuple[list[str], str] | None:
        for provider in self.configured():
            out = await provider.rewrite(sentences, instruction)
            if out:
                return out
        return None

    async def stream(self, request: AnswerRequest) -> AsyncIterator[AnswerEvent]:
        ready = self.configured()
        if not ready:
            raise ProviderError("not_configured", SETUP_HINT)
        failures: list[str] = []
        for i, provider in enumerate(ready):
            streamed = False
            try:
                async for event in provider.stream(request):
                    if event.kind == "delta":
                        streamed = True
                    elif event.kind == "result" and event.result is not None:
                        event.result.fallback_used = i > 0
                        if failures:
                            event.result.usage = {**event.result.usage, "failed_over": failures}
                    yield event
                return
            except ProviderError as exc:
                failures.append(f"{provider.preset.label}: {exc.code}")
                if streamed or exc.code not in FALLBACK_CODES or i == len(ready) - 1:
                    if len(failures) > 1:
                        raise ProviderError(exc.code, exc.message + " Also tried: "
                                            + "; ".join(failures[:-1]) + ".") from exc
                    raise
                log.info("%s failed (%s); trying the next provider", provider.preset.label, exc.code)
