"""The OpenAI-compatible providers (Groq, OpenRouter, NVIDIA NIM): [n] citation markers, the
fallback chain, finish reasons and provider selection. No network and no key: the client is
replaced by a scripted stream. The live smoke test runs only with a key and SAKSHYA_LIVE_LLM=1."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from app.answer.verify import verify
from app.llm import get_provider
from app.llm import openai_compat as oc
from app.llm.base import AnswerRequest, Passage, ProviderError, Turn

P1 = Passage(11, "gharkul.pdf · p. 1", "The scheme gives a house to families without one. It started in 2016.")
P2 = Passage(12, "gharkul.pdf · p. 2", "The annual income limit is three lakh rupees. Apply at the Gram Panchayat.")


def test_markers_become_per_sentence_citations():
    numbers = {1: 11, 2: 12}
    passages = {11: P1, 12: P2}
    segs = oc.parse_cited("The income limit is three lakh rupees [2]. The scheme started in 2016 [1][2].",
                          numbers, passages)
    assert [s.text.strip() for s in segs] == ["The income limit is three lakh rupees.", "The scheme started in 2016."]
    assert [c.chunk_id for c in segs[0].citations] == [12]
    assert [c.chunk_id for c in segs[1].citations] == [11, 12]
    assert segs[0].citations[0].cited_text.startswith("The annual income limit is three lakh rupees")
    # a marker after the full stop, a range, "passage n", and a number that was never given
    segs = oc.parse_cited("Apply at the Gram Panchayat. [2] It began in 2016 [passage 1]. Houses for all [1-2]. "
                          "Also a bonus [9].", numbers, passages)
    assert [[c.chunk_id for c in s.citations] for s in segs] == [[12], [11], [11, 12], []]


def test_uncited_sentences_are_removed_by_the_verifier():
    segs = oc.parse_cited("Here is what the documents say. The annual income limit is three lakh rupees [2].",
                          {1: 11, 2: 12}, {11: P1, 12: P2})
    v = verify(segs, {11: P1.text, 12: P2.text}, use_nli=False)
    assert [s.text for s in v.kept] == ["The annual income limit is three lakh rupees."]
    assert v.removed[0].reason == "no_citation"


def test_passages_are_numbered_across_turns():
    req = AnswerRequest("And where do I apply?", [P2], [Turn("What is the limit?", [P1, P2], "Three lakh.")])
    messages, numbers = oc.build_messages(req)
    assert numbers == {1: 11, 2: 12, 3: 12}
    assert messages[0]["role"] == "system" and "[n]" not in messages[0]["content"]
    assert "[1] (gharkul.pdf · p. 1)" in messages[1]["content"] and messages[2]["content"] == "Three lakh."
    assert "[3] (gharkul.pdf · p. 2)" in messages[3]["content"] and messages[3]["content"].endswith(
        "Question: And where do I apply?")


def chunk(content=None, finish=None, usage=None, model="openai/gpt-oss-120b"):
    choices = [SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)] if (
        content is not None or finish) else []
    return SimpleNamespace(model=model, usage=usage, choices=choices)


def scripted(provider, pieces, finish="stop"):
    async def gen():
        for p in pieces:
            yield chunk(p)
        yield chunk(finish=finish)
        yield chunk(usage=SimpleNamespace(prompt_tokens=900, completion_tokens=40, prompt_tokens_details=None))

    async def create(messages, with_usage):
        provider.sent = messages
        return gen()

    provider._create = create


def run(provider, request):
    async def go():
        deltas, result = [], None
        async for e in provider.stream(request):
            if e.kind == "delta":
                deltas.append(e.text)
            else:
                result = e.result
        return deltas, result

    return asyncio.run(go())


@pytest.fixture
def keys(monkeypatch):
    for p in oc.PRESETS.values():
        monkeypatch.setenv(p.key_env, "test-key-not-real")
    yield
    get_provider.cache_clear()


def test_a_streamed_answer_with_usage_and_think_block(keys):
    p = oc.OpenAICompatProvider(oc.PRESETS["groq"])
    scripted(p, ["<think>the user wants the limit</think>", "The annual income limit is ", "three lakh rupees [1]."])
    deltas, result = run(p, AnswerRequest("What is the annual income limit?", [P2]))
    assert "think" not in "".join(deltas)
    assert result.status == "complete" and result.model == "openai/gpt-oss-120b via Groq"
    assert result.text.strip() == "The annual income limit is three lakh rupees."
    assert result.segments[0].citations[0].chunk_id == 12
    assert result.usage == {"input_tokens": 900, "output_tokens": 40}


def test_length_and_content_filter_finish_reasons(keys):
    p = oc.OpenAICompatProvider(oc.PRESETS["nvidia"])
    scripted(p, ["The annual income limit is [1"], finish="length")
    assert run(p, AnswerRequest("q", [P2]))[1].status == "cut_short"
    scripted(p, ["I can"], finish="content_filter")
    result = run(p, AnswerRequest("q", [P2]))[1]
    assert result.status == "refused" and result.segments == []


def test_the_chain_falls_back_when_a_provider_fails_before_streaming(keys):
    chain = oc.ChainProvider("groq,openrouter")
    groq, router = chain.providers

    async def rate_limited(messages, with_usage):
        raise oc.openai.RateLimitError("slow down", response=_response(429), body=None)

    groq._create = rate_limited
    scripted(router, ["The annual income limit is three lakh rupees [1]."])
    _, result = run(chain, AnswerRequest("q", [P2]))
    assert result.fallback_used and result.model.endswith("via OpenRouter")
    assert result.usage["failed_over"] == ["Groq: rate_limited"]


def test_the_chain_reports_every_failure(keys):
    chain = oc.ChainProvider("groq,nvidia")

    async def missing_model(messages, with_usage):
        raise oc.openai.NotFoundError("no such model", response=_response(404), body=None)

    for p in chain.providers:
        p._create = missing_model
    with pytest.raises(ProviderError) as info:
        run(chain, AnswerRequest("q", [P2]))
    assert info.value.code == "not_found" and "Also tried: Groq: not_found" in info.value.message
    assert "test-key-not-real" not in info.value.message


def test_without_keys_the_chain_explains_the_setup(monkeypatch):
    for p in oc.PRESETS.values():
        monkeypatch.delenv(p.key_env, raising=False)
    with pytest.raises(ProviderError) as info:
        run(oc.ChainProvider(), AnswerRequest("q", [P2]))
    assert info.value.code == "not_configured" and "GROQ_API_KEY" in info.value.message


def test_auto_selection(monkeypatch):
    for p in oc.PRESETS.values():
        monkeypatch.delenv(p.key_env, raising=False)
    monkeypatch.setenv("SAKSHYA_LLM_PROVIDER", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    get_provider.cache_clear()
    assert get_provider().name == "anthropic"
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    get_provider.cache_clear()
    provider = get_provider()
    assert provider.name == "chain" and provider.status()["provider"] == "NVIDIA NIM"
    assert "test-key-not-real" not in str(provider.status())
    get_provider.cache_clear()


def _response(code):
    import httpx

    return httpx.Response(code, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))


# ---------- live smoke test: needs a key and SAKSHYA_LIVE_LLM=1 (free tiers cost nothing) ----------

LIVE = os.environ.get("SAKSHYA_LIVE_LLM") == "1" and any(os.environ.get(p.key_env) for p in oc.PRESETS.values())


@pytest.mark.skipif(not LIVE, reason="set SAKSHYA_LIVE_LLM=1 and a provider key to call the real API")
def test_live_answer_cites_its_passage():
    chain = oc.ChainProvider()
    _, result = run(chain, AnswerRequest("What is the annual income limit?", [P1, P2]))
    assert result.status == "complete"
    assert any(c.chunk_id == 12 for s in result.segments for c in s.citations)
    assert "three lakh" in result.text.lower() or "3" in result.text
