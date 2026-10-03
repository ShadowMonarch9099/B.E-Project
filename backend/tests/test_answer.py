"""Phase 4: retrieval -> evidence gate -> cited answer over SSE, with query logs.

The answer model is a scripted fake (no key, no cost). The Anthropic provider's request
building, response parsing and error mapping are tested against the SDK's own types; one
live smoke test runs only when ANTHROPIC_API_KEY is set.
"""

import asyncio
import json
import os

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage
from fastapi.testclient import TestClient

from app.db import QueryLog, session_scope
from app.llm import get_provider
from app.llm import anthropic_provider as ap
from app.llm.base import AnswerRequest, AnswerResult, Citation, Passage, ProviderError, Segment, Turn
from app.main import app
from tests.helpers import native_pdf

PAGES = [
    "Introduction to the scheme for rural housing in Maharashtra districts. " * 5,
    "The annual income of the beneficiary family must not exceed three lakh rupees. "
    "The applicant must attach an income certificate issued by the Tahsildar. " * 3,
]


def sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def doc_id(client):
    r = client.post("/api/documents", files={"file": ("gharkul.pdf", native_pdf(PAGES), "application/pdf")})
    return r.json()["document"]["id"]


@pytest.fixture()
def fake():
    provider = get_provider()
    provider.script.clear()
    provider.requests.clear()
    return provider


def ask(client, conv_id: str, question: str) -> list[tuple[str, dict]]:
    r = client.post(f"/api/conversations/{conv_id}/messages", json={"question": question})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    return sse(r.text)


# ---------- the Anthropic provider, offline ----------

P1 = Passage(11, "gharkul.pdf · p. 2", "The annual income must not exceed three lakh rupees.", None)
P2 = Passage(12, "gharkul.pdf · p. 3", "Attach the income certificate.", "Read from a scan with OCR (64%).")


def test_each_chunk_is_its_own_cited_document_block():
    req = AnswerRequest("What is the income limit?", [P2],
                        history=[Turn("Earlier question?", [P1], "Earlier answer.")])
    messages, order = ap.build_messages(req)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    doc = messages[0]["content"][0]
    assert doc == {"type": "document", "title": "gharkul.pdf · p. 2", "citations": {"enabled": True},
                   "source": {"type": "content", "content": [{"type": "text", "text": P1.text}]}}
    assert messages[0]["content"][-1] == {"type": "text", "text": "Earlier question?"}
    assert messages[2]["content"][0]["context"].startswith("Read from a scan")
    assert messages[2]["content"][-1]["text"] == "What is the income limit?"
    assert order == [11, 12]  # document_index -> chunk id, across the whole request


def message(content: list, stop_reason: str = "end_turn", **extra) -> BetaMessage:
    return BetaMessage.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 900, "output_tokens": 40, "cache_read_input_tokens": 500,
                  "cache_creation_input_tokens": 0}, **extra})


def cite(index: int, text: str) -> dict:
    return {"type": "content_block_location", "cited_text": text, "document_index": index,
            "document_title": "t", "start_block_index": 0, "end_block_index": 1}


def test_citations_map_back_to_chunks():
    req = AnswerRequest("q", [P2], history=[Turn("q0", [P1], "a0")])
    _, order = ap.build_messages(req)
    msg = message([
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "The limit is three lakh rupees", "citations": [cite(0, P1.text)]},
        {"type": "text", "text": ". ", "citations": None},
        {"type": "text", "text": "Attach the certificate.", "citations": [cite(1, "Attach the income certificate.")]},
    ])
    r = ap.parse_message(msg, req, order)
    assert r.status == "complete" and r.text == "The limit is three lakh rupees. Attach the certificate."
    assert [[c.chunk_id for c in s.citations] for s in r.segments] == [[11], [], [12]]
    assert r.usage["cache_read_input_tokens"] == 500


def test_refusal_and_max_tokens_are_checked_before_the_content():
    req = AnswerRequest("q", [P1])
    refused = message([{"type": "text", "text": "partial", "citations": None}], "refusal",
                      stop_details={"type": "refusal", "category": "bio", "explanation": None})
    r = ap.parse_message(refused, req, [11])
    assert r.status == "refused" and r.segments == [] and r.refusal_category == "bio"
    cut = ap.parse_message(message([{"type": "text", "text": "The limit", "citations": None}], "max_tokens"),
                           req, [11])
    assert cut.status == "cut_short" and cut.text == "The limit"


def _raising(exc):
    class Stream:
        async def __aenter__(self):
            raise exc

        async def __aexit__(self, *a):
            return False

    class Messages:
        def stream(self, **kwargs):
            return Stream()

    class Beta:
        messages = Messages()

    class Client:
        beta = Beta()

    return Client()


REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize("exc,code", [
    (anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=REQ), body=None), "auth"),
    (anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=REQ), body=None), "rate_limited"),
    (anthropic.InternalServerError("boom", response=httpx2.Response(500, request=REQ), body=None), "api_error"),
    (anthropic.APIConnectionError(request=REQ), "connection"),
])
def test_api_errors_become_readable_provider_errors(monkeypatch, exc, code):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    provider = ap.AnthropicProvider()
    monkeypatch.setattr(provider, "client", lambda: _raising(exc))

    async def run():
        async for _ in provider.stream(AnswerRequest("q", [P1])):
            pass

    with pytest.raises(ProviderError) as info:
        asyncio.run(run())
    assert info.value.code == code and "sk-ant" not in info.value.message


def test_missing_key_gives_a_setup_message_without_calling_the_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    provider = ap.AnthropicProvider()
    monkeypatch.setattr(provider, "client", lambda: pytest.fail("the API must not be called"))

    async def run():
        async for _ in provider.stream(AnswerRequest("q", [P1])):
            pass

    with pytest.raises(ProviderError) as info:
        asyncio.run(run())
    assert info.value.code == "not_configured" and "ANTHROPIC_API_KEY" in info.value.message
    assert provider.status()["configured"] is False


# ---------- end to end through the API ----------

def test_answer_streams_with_page_citations_and_is_logged(client, doc_id, fake):
    conv = client.post("/api/conversations", json={"doc_id": doc_id}).json()
    events = ask(client, conv["id"], "What is the annual income limit for the beneficiary family?")
    names = [e for e, _ in events]
    assert names[0] == "status" and "retrieval" in names and "delta" in names and names[-1] == "done"
    answer = dict(events)["answer"]
    assert answer["status"] == "complete" and answer["payload"]["kind"] == "answer"
    cit = answer["payload"]["sentences"][0]["citations"][0]
    assert cit["page_no"] == 2 and cit["label"] == "p. 2" and cit["filename"] == "gharkul.pdf" and cit["bbox"]
    # only the top passages were sent, never the whole document
    assert 0 < len(fake.requests[-1].passages) <= 6
    assert fake.requests[-1].answer_language == "Answer in English."  # stated with every question
    with session_scope() as s:
        row = s.get(QueryLog, dict(events)["done"]["query_log_id"])
        assert row.llm_called and not row.abstained and row.gate["passed"]
        assert row.route == "hybrid" and row.candidates["candidates"] and row.latency_ms["total"] > 0


def test_off_topic_question_abstains_without_calling_the_model(client, doc_id, fake, monkeypatch):
    import app.retrieval.hybrid as hybrid

    monkeypatch.setattr(hybrid, "SMALL_DOC_CHUNKS", 0)  # as for a document larger than the 6 passages sent
    conv = client.post("/api/conversations", json={"doc_id": doc_id}).json()
    events = ask(client, conv["id"], "Who won the cricket world cup in 2011?")
    answer = dict(events)["answer"]
    assert answer["status"] == "abstained" and answer["payload"]["kind"] == "abstain"
    assert answer["payload"]["nearest"] and "page_no" in answer["payload"]["nearest"][0]
    assert fake.requests == []  # the gate stopped it before the model
    with session_scope() as s:
        row = s.get(QueryLog, dict(events)["done"]["query_log_id"])
        assert row.abstained and not row.llm_called and not row.gate["passed"]


def test_without_a_key_the_answer_explains_the_setup(client, doc_id, fake):
    fake.script.append(ProviderError("not_configured", ap.SETUP_HINT))
    conv = client.post("/api/conversations", json={"doc_id": doc_id}).json()
    answer = dict(ask(client, conv["id"], "What is the annual income limit?"))["answer"]
    assert answer["status"] == "error" and answer["payload"]["code"] == "not_configured"
    assert "ANTHROPIC_API_KEY" in answer["payload"]["message"] and answer["payload"]["nearest"]


def test_follow_up_questions_carry_the_earlier_turn(client, doc_id, fake):
    conv = client.post("/api/conversations", json={"doc_id": doc_id}).json()
    ask(client, conv["id"], "What is the annual income limit for the beneficiary family?")
    ask(client, conv["id"], "Which income certificate must the applicant attach?")
    second = fake.requests[-1]
    assert len(second.history) == 1 and second.history[0].question.startswith("What is the annual income")
    assert second.history[0].passages and second.history[0].answer
    stored = client.get(f"/api/conversations/{conv['id']}").json()["messages"]
    assert [m["role"] for m in stored] == ["user", "assistant", "user", "assistant"]


def test_refusal_is_shown_politely(client, doc_id, fake):
    fake.script.append(AnswerResult([], "refused", "refusal", "claude-opus-5-5", refusal_category="cyber"))
    conv = client.post("/api/conversations", json={"doc_id": doc_id}).json()
    answer = dict(ask(client, conv["id"], "What is the annual income limit?"))["answer"]
    assert answer["status"] == "refused" and answer["content"].startswith("Sorry")


def test_cut_short_answer_is_marked(client, doc_id, fake):
    fake.script.append(AnswerResult([Segment("The limit is", [Citation(0, "")])], "cut_short", "max_tokens", "m"))
    conv = client.post("/api/conversations", json={"doc_id": doc_id}).json()
    answer = dict(ask(client, conv["id"], "What is the annual income limit?"))["answer"]
    assert answer["status"] == "cut_short"


# ---------- live smoke test (costs about one cent) ----------

@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set (live API test)")
def test_live_claude_answer_has_citations():
    provider = ap.AnthropicProvider()
    passage = Passage(1, "gharkul.pdf · p. 2", PAGES[1])

    async def run():
        result = None
        async for event in provider.stream(AnswerRequest("What is the annual income limit?", [passage])):
            if event.kind == "result":
                result = event.result
        return result

    result = asyncio.run(run())
    assert result.status == "complete"
    assert any(c.chunk_id == 1 for s in result.segments for c in s.citations)
    assert "three lakh" in result.text.lower() or "3" in result.text


def test_a_small_document_goes_to_the_model_whole(client, doc_id, fake):
    """All of a small document fits in the passages sent, so the gate does not stop the question;
    the model decides, and the citation check still applies."""
    conv = client.post("/api/conversations", json={"doc_id": doc_id}).json()
    events = dict(ask(client, conv["id"], "Who won the cricket world cup in 2011?"))
    assert fake.requests and fake.requests[-1].question.startswith("Who won")
    assert "whole document" in events["retrieval"]["gate"]["reason"]
