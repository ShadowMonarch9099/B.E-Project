"""Cross-encoder reranking (plan section 6.4), with a pretrained model used as-is.

cross-encoder/mmarco-mMiniLMv2-L12-H384-v1: a multilingual MiniLM trained on mMARCO
(MS MARCO translated into 14 languages, including Hindi). It reads the question and a
passage together and returns one relevance logit; we report sigmoid(logit) in [0, 1] so
the evidence gate's threshold is easy to read. No fine-tuning (decision log, 2026-10-03).

SAKSHYA_RERANKER=overlap swaps in a word-overlap stand-in for the unit tests.
"""

import logging
import math
import os
import re
import threading
from functools import lru_cache

log = logging.getLogger("sakshya.reranker")

RERANK_MODEL = os.environ.get("SAKSHYA_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
HF_FILES = ["*.json", "model.safetensors", "sentencepiece.bpe.model"]
BATCH = 16
MAX_LENGTH = 512


class CrossEncoderReranker:
    def __init__(self, name: str = RERANK_MODEL):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = snapshot_download(name, allow_patterns=HF_FILES)
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self._torch = torch
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, as sigmoid(logit) in [0, 1]."""
        torch = self._torch
        scores: list[float] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(passages), BATCH):
                batch = passages[i:i + BATCH]
                enc = self.tokenizer([query] * len(batch), batch, padding=True, truncation="only_second",
                                     max_length=MAX_LENGTH, return_tensors="pt")
                logits = self.model(**enc).logits.view(-1)
                scores.extend(torch.sigmoid(logits).tolist())
        return scores


class OverlapReranker:
    """Test stand-in: the share of the query's words that appear in the passage."""

    name = "word-overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3 or w.isdigit()}  # skip "the", "in"
        out = []
        for p in passages:
            have = set(re.findall(r"\w+", p.lower()))
            share = len(words & have) / len(words) if words else 0.0
            out.append(1 / (1 + math.exp(-10 * (share - 0.5))))
        return out


@lru_cache(maxsize=1)
def get_reranker():
    if os.environ.get("SAKSHYA_RERANKER", "cross-encoder") == "overlap":
        return OverlapReranker()
    log.info("Loading %s", RERANK_MODEL)
    return CrossEncoderReranker(RERANK_MODEL)


def status() -> dict:
    loaded = get_reranker.cache_info().currsize > 0
    name = get_reranker().name if loaded else (
        "word-overlap" if os.environ.get("SAKSHYA_RERANKER") == "overlap" else RERANK_MODEL)
    return {"model": name, "loaded": loaded}
