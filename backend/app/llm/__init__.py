"""Answer providers. SAKSHYA_LLM_PROVIDER picks one:

- "auto" (default): the OpenAI-compatible chain (Groq, OpenRouter, NVIDIA NIM, in the order
  SAKSHYA_LLM_ORDER) when any of their keys is set; otherwise Claude if ANTHROPIC_API_KEY is set;
  otherwise the chain, which explains how to add a key.
- "chain", or one of "groq" / "openrouter" / "nvidia" alone;
- "anthropic": the Claude API with its citations feature;
- "fake": the scripted provider used by the tests.
A local model (plan section 7.3) can be registered here as another provider.
"""

import os
from functools import lru_cache

from .base import AnswerProvider


@lru_cache(maxsize=1)
def get_provider() -> AnswerProvider:
    name = os.environ.get("SAKSHYA_LLM_PROVIDER", "auto")
    if name == "auto":
        from .anthropic_provider import key_configured
        from .openai_compat import PRESETS

        if not any(os.environ.get(p.key_env) for p in PRESETS.values()) and key_configured():
            name = "anthropic"
        else:
            name = "chain"
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if name == "chain":
        from .openai_compat import ChainProvider

        return ChainProvider()
    if name in ("groq", "openrouter", "nvidia"):
        from .openai_compat import ChainProvider

        return ChainProvider(name)
    if name == "fake":  # tests only
        from .fake import FakeProvider

        return FakeProvider()
    raise ValueError(f"Unknown SAKSHYA_LLM_PROVIDER: {name}")
