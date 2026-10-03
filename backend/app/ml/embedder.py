"""Dense embeddings with multilingual-E5 (plan section 6.2 step 8), used as-is.

E5 expects "passage: " before indexed text and "query: " before questions; vectors are
mean-pooled over the tokens and L2-normalised, so cosine similarity is a dot product.
SAKSHYA_EMBED_MODEL picks the size (intfloat/multilingual-e5-base, or -large: a budget
variable in the plan). Only the files needed for PyTorch are downloaded.

SAKSHYA_EMBEDDER=hash swaps in a tiny deterministic embedder (hashed character
trigrams). The unit tests use it so they do not download a 1 GB model; it is not a
retrieval model and is never used by default.
"""

import hashlib
import logging
import os
import threading
from functools import lru_cache

import numpy as np

log = logging.getLogger("sakshya.embedder")

EMBED_MODEL = os.environ.get("SAKSHYA_EMBED_MODEL", "intfloat/multilingual-e5-base")
BATCH = int(os.environ.get("SAKSHYA_EMBED_BATCH", "16"))
MAX_LENGTH = 512
HF_FILES = ["*.json", "model.safetensors", "sentencepiece.bpe.model"]


class E5Embedder:
    def __init__(self, name: str = EMBED_MODEL):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        path = snapshot_download(name, allow_patterns=HF_FILES)
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).eval()
        self.dim = int(self.model.config.hidden_size)
        self._torch = torch
        self._lock = threading.Lock()  # one forward pass at a time; torch already uses all cores

    def _embed(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(texts), BATCH):
                enc = self.tokenizer(texts[i:i + BATCH], padding=True, truncation=True,
                                     max_length=MAX_LENGTH, return_tensors="pt")
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([f"passage: {t}" for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([f"query: {t}" for t in texts])


class HashEmbedder:
    """Test stand-in: hashed character trigrams, 256 dimensions."""

    name = "hash-trigram-256"
    dim = 256

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            h = int.from_bytes(hashlib.blake2b(t[i:i + 3].encode(), digest_size=4).digest(), "little")
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    embed_queries = embed_passages


@lru_cache(maxsize=1)
def get_embedder():
    kind = os.environ.get("SAKSHYA_EMBEDDER", "e5")
    if kind == "hash":
        return HashEmbedder()
    log.info("Loading %s", EMBED_MODEL)
    return E5Embedder(EMBED_MODEL)


def status() -> dict:
    """Whether the embedding model is loaded, without loading it."""
    loaded = get_embedder.cache_info().currsize > 0
    name = get_embedder().name if loaded else (
        "hash-trigram-256" if os.environ.get("SAKSHYA_EMBEDDER") == "hash" else EMBED_MODEL)
    return {"model": name, "loaded": loaded}
