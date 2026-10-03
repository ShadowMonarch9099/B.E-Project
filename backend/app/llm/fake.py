"""A scripted provider for the tests: no network, no key, no cost.

By default it answers with the first sentence of the top passage and cites that passage.
Tests can set `script` to a list of AnswerResult / ProviderError to return in turn, and
read `requests` to see what would have been sent.
"""

from collections.abc import AsyncIterator

from .base import AnswerEvent, AnswerRequest, AnswerResult, Citation, ProviderError, Segment


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.script: list = []
        self.requests: list[AnswerRequest] = []

    def status(self) -> dict:
        return {"provider": "fake (tests)", "configured": True, "model": "fake", "effort": None}

    async def stream(self, request: AnswerRequest) -> AsyncIterator[AnswerEvent]:
        self.requests.append(request)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, ProviderError):
                raise item
            result = item
        else:
            top = request.passages[0]
            first = top.text.split(". ")[0].strip().rstrip(".") + "."
            result = AnswerResult([Segment(first, [Citation(top.chunk_id, first)])], "complete", "end_turn", "fake")
        for seg in result.segments:
            for word in seg.text.split(" "):
                yield AnswerEvent("delta", text=word + " ")
        yield AnswerEvent("result", result=result)
