"""The answer-model interface (plan section 6.7 step 2).

Retrieval, the evidence gate and the UI talk only to `AnswerProvider`. The Claude API is
the provider for now (decision log, 2026-10-03); the plan's local 4-bit model can be added
later as a second provider without touching retrieval or the frontend.

Only the top passages for each question are sent to the provider, never a whole document.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Passage:
    """One retrieved chunk, as the answer model sees it."""

    chunk_id: int
    title: str  # names the file and page, e.g. "GR_2023.pdf · p. 4"
    text: str
    context: str | None = None  # e.g. "Read with OCR (confidence 64%); figures may be misread."

    def as_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "title": self.title, "text": self.text, "context": self.context}

    @classmethod
    def from_dict(cls, d: dict) -> "Passage":
        return cls(d["chunk_id"], d["title"], d["text"], d.get("context"))


@dataclass
class Turn:
    """An earlier question and its answer, replayed for follow-up questions."""

    question: str
    passages: list[Passage]
    answer: str


@dataclass
class AnswerRequest:
    question: str
    passages: list[Passage]
    history: list[Turn] = field(default_factory=list)
    # e.g. "Answer in English." Stated with every question: gpt-oss answered an English question
    # about Marathi passages in Marathi when the language rule was only in the system prompt.
    answer_language: str | None = None


@dataclass
class Citation:
    chunk_id: int
    cited_text: str


@dataclass
class Segment:
    """A run of answer text and the passages it cites (the API's text blocks)."""

    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class AnswerResult:
    segments: list[Segment]
    # complete / cut_short (hit max_tokens) / refused (declined by the model or its safeguards)
    status: str
    stop_reason: str | None
    model: str
    refusal_category: str | None = None
    fallback_used: bool = False
    usage: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)


@dataclass
class AnswerEvent:
    kind: str  # "delta" (streamed text) or "result" (the finished answer)
    text: str = ""
    result: AnswerResult | None = None


class ProviderError(Exception):
    """A failure the user should see, with a stable code for the UI."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # not_configured / auth / permission / rate_limited / busy / api_error / connection
        self.message = message


class AnswerProvider(Protocol):
    name: str

    def status(self) -> dict:
        """{"configured": bool, "model": str, ...}; must not reveal secrets."""

    def stream(self, request: AnswerRequest) -> AsyncIterator[AnswerEvent]:
        """Text deltas as they arrive, then one "result" event. Raises ProviderError."""

    async def rewrite(self, sentences: list[str], instruction: str) -> tuple[list[str], str] | None:
        """The sentences rewritten as instructed (e.g. into English), one for one, and the model
        that did it; None if this provider cannot. Used to repair an answer in the wrong language."""
