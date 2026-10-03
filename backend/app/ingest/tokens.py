"""Token counts for chunk sizes, in the dense model's own tokens (XLM-R SentencePiece).

Chunks must stay under multilingual-E5's 512-token limit, and Devanagari text needs more
tokens per word than English, so word counts are not enough. When the tokenizer cannot be
loaded (offline, first run), an estimate is used instead; it errs on the high side.
"""

import logging
import math
import os
from functools import lru_cache

log = logging.getLogger("sakshya.tokens")

EMBED_MODEL = os.environ.get("SAKSHYA_EMBED_MODEL", "intfloat/multilingual-e5-base")


@lru_cache(maxsize=1)
def _tokenizer():
    if os.environ.get("SAKSHYA_TEST_MODE") == "1" and os.environ.get("SAKSHYA_TEST_TOKENIZER") != "1":
        return None  # tests use the deterministic estimate
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(EMBED_MODEL)
    except Exception as exc:  # offline and not cached yet
        log.warning("Tokenizer for %s not available (%s); estimating token counts", EMBED_MODEL, exc)
        return None


def estimate(text: str) -> int:
    """About 1.3 tokens per English word; Devanagari words split into more pieces."""
    total = 0
    for word in text.split():
        deva = sum(1 for c in word if "ऀ" <= c <= "ॿ")
        total += math.ceil(len(word) / 2.5) if deva else max(1, math.ceil(len(word) / 4.5))
    return total


def count(text: str) -> int:
    tok = _tokenizer()
    if tok is None:
        return estimate(text)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def count_many(texts: list[str]) -> list[int]:
    tok = _tokenizer()
    if tok is None:
        return [estimate(t) for t in texts]
    if not texts:
        return []
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def tokenizer_name() -> str:
    return EMBED_MODEL if _tokenizer() is not None else "estimate"
