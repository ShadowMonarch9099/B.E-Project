"""Answer providers. SAKSHYA_LLM_PROVIDER picks one: "anthropic" (default). A local model
(plan section 7.3) can be registered here as a second provider."""

import os
from functools import lru_cache

from .base import AnswerProvider


@lru_cache(maxsize=1)
def get_provider() -> AnswerProvider:
    name = os.environ.get("SAKSHYA_LLM_PROVIDER", "anthropic")
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if name == "fake":  # tests only
        from .fake import FakeProvider

        return FakeProvider()
    raise ValueError(f"Unknown SAKSHYA_LLM_PROVIDER: {name}")
