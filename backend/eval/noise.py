"""OCR-noise stress test (plan section 9.1, Robustness): Recall@20 when the indexed text has a
synthetic character error rate (CER) of 5-30%.

Clean (native) pages are corrupted the way OCR corrupts Devanagari and Latin text: look-alike
substitutions (ि/ी, ब/व, rn/m, l/1, ...), dropped and inserted characters. The questions are
not corrupted. For each noise level the chunks are indexed again in memory, without touching
the app's database: E5 vectors of the noisy text, BM25 over its words and over its character
3-5-grams. Each question searches the chunks of its own document, and Recall@20 is computed
at page level for dense alone, words alone, character n-grams alone, the hybrid fusion, and
the hybrid fusion with OCR-aware weights (each chunk given an OCR confidence of 1 - CER).
"""

import random
import time

import numpy as np

from app.retrieval.hybrid import WEIGHTS, fuse

from . import metrics

CONFUSABLE = {
    # Devanagari vowel signs and consonants that OCR confuses
    "ि": "ी", "ी": "ि", "ु": "ू", "ू": "ु", "े": "ै", "ै": "े", "ो": "ौ", "ौ": "ो", "ं": "ँ", "ँ": "ं",
    "ब": "व", "व": "ब", "म": "भ", "भ": "म", "घ": "ध", "ध": "घ", "प": "ष", "ष": "प", "ड": "ङ", "ट": "ठ",
    "ठ": "ट", "द": "ढ", "र": "२", "ल": "ळ", "ळ": "ल", "न": "त", "त": "न", "्": "", "।": "|",
    # Latin
    "l": "1", "1": "l", "I": "l", "O": "0", "0": "O", "o": "c", "e": "c", "c": "e", "n": "r", "u": "v",
    "a": "o", "S": "5", "5": "S", "B": "8", "8": "B", ",": ".", ".": ",",
}
SUB, DEL = 0.6, 0.2  # the rest (0.2) are insertions


def add_noise(text: str, cer: float, rng: random.Random) -> str:
    """Corrupt about `cer` of the non-space characters: 60% look-alike substitutions,
    20% deletions, 20% insertions (a duplicated or stray character)."""
    if cer <= 0:
        return text
    out = []
    for ch in text:
        if ch.isspace() or rng.random() >= cer:
            out.append(ch)
            continue
        x = rng.random()
        if x < SUB:
            out.append(CONFUSABLE.get(ch) if ch in CONFUSABLE else rng.choice(text.replace(" ", "") or ch))
        elif x < SUB + DEL:
            continue
        else:
            out.append(ch + rng.choice(("।", ch, "'", "-")))
    return "".join(out)


def char_error_rate(clean: str, noisy: str) -> float:
    """Levenshtein distance / length of the clean text (to check the noise level)."""
    a, b = clean, noisy
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


class _MemoryIndex:
    """Dense + BM25 word + BM25 character indexes over a list of chunk texts."""

    def __init__(self, texts: list[str], embed) -> None:
        import bm25s

        from app.index.lexical import char_ngrams, word_tokens

        self._words, self._chars = word_tokens, char_ngrams
        tokens = [word_tokens(t) for t in texts]
        self.word = bm25s.BM25()
        self.word.index([t or ["∅"] for t in tokens], show_progress=False)
        self.char = bm25s.BM25()
        self.char.index([char_ngrams(t) or ["∅"] for t in tokens], show_progress=False)
        self.vecs = embed(texts)

    def search(self, views: list[str], query_vecs: np.ndarray, mask: np.ndarray, k: int) -> dict:
        idx = np.flatnonzero(mask)
        dense = (self.vecs[idx] @ query_vecs.T).max(axis=1)
        out = {"dense": [(int(idx[i]), float(dense[i])) for i in np.argsort(-dense)[:k]]}
        words = self._words(" ".join(views))
        for name, index, q in (("word", self.word, words), ("char", self.char, self._chars(words))):
            if not q:
                out[name] = []
                continue
            scores = index.get_scores(q, weight_mask=mask.astype(np.float32))
            top = np.argsort(-scores)[:k]
            out[name] = [(int(i), float(scores[i])) for i in top if scores[i] > 0]
        return out


def stress(questions: list[dict], chunks: list[dict], levels: list[float], analyses: dict, k: int = 40,
           seed: int = 7, progress=print) -> dict:
    """questions: {"id", "question", "doc_id", "gold": {(doc_id, page)}}; chunks: {"doc_id",
    "page_no", "text"} of the questions' (native) documents; analyses: question -> QueryAnalysis."""
    from app.ml.embedder import get_embedder

    emb = get_embedder()
    doc_of = np.array([c["doc_id"] for c in chunks], dtype=object)
    pages = [(c["doc_id"], c["page_no"]) for c in chunks]
    qvecs = {q["question"]: emb.embed_queries(list(analyses[q["question"]].views().values())) for q in questions}
    results = {}
    for level in levels:
        t0 = time.perf_counter()
        rng = random.Random(seed)
        noisy = [add_noise(c["text"], level, rng) for c in chunks]
        sample = [char_error_rate(c["text"][:400], n[:400]) for c, n in zip(chunks[:50], noisy[:50])]
        index = _MemoryIndex(noisy, emb.embed_passages)
        conf = {i: (1.0 - level if level > 0 else None) for i in range(len(chunks))}
        per = {name: [] for name in ("dense", "word", "char", "hybrid", "hybrid+ocr-aware")}
        for q in questions:
            mask = doc_of == q["doc_id"]
            views = list(analyses[q["question"]].views().values())
            lists = index.search(views, qvecs[q["question"]], mask, k)
            ranked = {name: [pages[i] for i, _ in hits] for name, hits in lists.items()}
            ranked["hybrid"] = [pages[c.chunk_id] for c in fuse(lists, WEIGHTS)[:k]]
            ranked["hybrid+ocr-aware"] = [pages[c.chunk_id] for c in fuse(lists, WEIGHTS, conf)[:k]]
            for name in per:
                per[name].append(metrics.recall_at(ranked[name], q["gold"], 20))
        results[str(level)] = {"target_cer": level, "measured_cer": round(metrics.mean(sample) or 0.0, 4),
                               "recall@20": {name: metrics.mean(v) for name, v in per.items()},
                               "per_question": per, "seconds": round(time.perf_counter() - t0, 1)}
        if progress:
            progress(f"  CER {level:.0%}: " + ", ".join(f"{n} {metrics.mean(v):.3f}" for n, v in per.items())
                     + f" ({results[str(level)]['seconds']}s)")
    return results
