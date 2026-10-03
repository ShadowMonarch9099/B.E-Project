"""Retrieval routes, fusion, reranking and the evidence gate (plan sections 6.4 and 6.7 step 1).

hybrid:  E5 dense + BM25 words + BM25 character n-grams, fused with weighted reciprocal
         rank fusion (RRF), top 40 -> cross-encoder -> top 6.
direct:  E5 dense only, top 40 -> cross-encoder -> top 6.

The evidence gate abstains, without calling the answer model, when the best reranker
score is below tau (SAKSHYA_GATE_TAU). tau is a hand-set threshold for now: the plan's
calibrated gate (section 6.7) is future work, and tau will be re-chosen from the
risk-coverage curve on the Sakshya-Bench dev split (decision log, 2026-10-03).
"""

import os
import time
from dataclasses import dataclass, field

from ..index import dense
from ..index.lexical import get_lexical_index
from ..ingest.normalize import normalize
from ..ml.embedder import get_embedder
from ..ml.reranker import get_reranker
from .chunks import ChunkInfo, load_chunks

RRF_K = 60
TOP_K = int(os.environ.get("SAKSHYA_TOP_K", "40"))
TOP_N = int(os.environ.get("SAKSHYA_TOP_N", "6"))
WEIGHTS = {"dense": 1.0, "word": 1.0, "char": 0.6}
# Starting value, measured with tools/gate_probe.py on 24 hand-written questions over two of
# the team's sample GRs (2026-10-03): the 14 answerable questions' best reranker scores were
# 0.077-1.000 and the 10 unanswerable ones' 0.002-0.063. A small sample with a thin margin;
# to be re-chosen from the risk-coverage curve on the Sakshya-Bench dev split.
GATE_TAU = float(os.environ.get("SAKSHYA_GATE_TAU", "0.07"))
NEAREST_PAGES = 3


@dataclass
class Candidate:
    chunk_id: int
    fused: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # 1-based rank in each list it appears in
    scores: dict[str, float] = field(default_factory=dict)
    rerank: float | None = None

    def as_log(self) -> dict:
        return {"chunk_id": self.chunk_id, "fused": round(self.fused, 5), "ranks": self.ranks,
                "scores": {k: round(v, 4) for k, v in self.scores.items()},
                "rerank": None if self.rerank is None else round(self.rerank, 4)}


@dataclass
class GateDecision:
    passed: bool
    score: float
    tau: float
    reason: str

    def as_dict(self) -> dict:
        return {"passed": self.passed, "score": round(self.score, 4), "tau": self.tau, "reason": self.reason}


@dataclass
class Retrieval:
    question: str
    normalized: str
    route: str
    lists: dict[str, list[tuple[int, float]]]  # per-source hits before fusion
    candidates: list[Candidate]  # fused top-k, reranked order
    top: list[Candidate]  # top-n after reranking
    gate: GateDecision
    chunks: dict[int, ChunkInfo]
    latency_ms: dict[str, float]

    def nearest_pages(self, n: int = NEAREST_PAGES) -> list[dict]:
        """The best distinct pages, for an abstention."""
        seen, out = set(), []
        for c in self.candidates:
            info = self.chunks.get(c.chunk_id)
            if info is None or (info.doc_id, info.page_no) in seen:
                continue
            seen.add((info.doc_id, info.page_no))
            out.append({**info.as_dict(with_text=False), "score": round(c.rerank or 0.0, 4),
                        "snippet": info.text[:240]})
            if len(out) == n:
                break
        return out

    def log(self) -> dict:
        return {"lists": {k: [[cid, round(s, 4)] for cid, s in v] for k, v in self.lists.items()},
                "candidates": [c.as_log() for c in self.candidates]}


def fuse(lists: dict[str, list[tuple[int, float]]], weights: dict[str, float]) -> list[Candidate]:
    """Weighted reciprocal rank fusion: sum over lists of weight / (60 + rank)."""
    out: dict[int, Candidate] = {}
    for name, hits in lists.items():
        w = weights.get(name, 0.0)
        for rank, (cid, score) in enumerate(hits, start=1):
            c = out.setdefault(cid, Candidate(cid))
            c.ranks[name] = rank
            c.scores[name] = score
            c.fused += w / (RRF_K + rank)
    return sorted(out.values(), key=lambda c: -c.fused)


def evidence_gate(top: list[Candidate], tau: float = GATE_TAU) -> GateDecision:
    best = max((c.rerank or 0.0 for c in top), default=0.0)
    if not top:
        return GateDecision(False, 0.0, tau, "nothing was retrieved")
    if best < tau:
        return GateDecision(False, best, tau, f"best passage scores {best:.2f}, below the threshold {tau:.2f}")
    return GateDecision(True, best, tau, f"best passage scores {best:.2f} (threshold {tau:.2f})")


def first_stage(question: str, doc_ids: list[str] | None, route: str = "hybrid",
                k: int = TOP_K, timings: dict | None = None) -> dict[str, list[tuple[int, float]]]:
    """The candidate lists a route draws from, before fusion."""
    timings = {} if timings is None else timings
    norm = normalize(question)
    t = time.perf_counter()
    qvec = get_embedder().embed_queries([norm])[0]
    timings["embed"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    lists = {"dense": dense.search(qvec, doc_ids, k)}
    timings["dense"] = (time.perf_counter() - t) * 1000
    if route == "hybrid":
        t = time.perf_counter()
        lexical = get_lexical_index().search(norm, doc_ids, k)
        lists["word"] = [(h.chunk_id, h.score) for h in lexical["word"]]
        lists["char"] = [(h.chunk_id, h.score) for h in lexical["char"]]
        timings["lexical"] = (time.perf_counter() - t) * 1000
    return lists


def retrieve(question: str, doc_ids: list[str] | None, route: str = "hybrid", k: int = TOP_K,
             n: int = TOP_N, tau: float = GATE_TAU, weights: dict[str, float] | None = None) -> Retrieval:
    if route not in {"hybrid", "direct"}:
        raise ValueError(f"unknown route {route}")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    lists = first_stage(question, doc_ids, route, k, timings)
    candidates = fuse(lists, weights or WEIGHTS)[:k]
    chunks = load_chunks([c.chunk_id for c in candidates])
    candidates = [c for c in candidates if c.chunk_id in chunks]

    t = time.perf_counter()
    if candidates:
        scores = get_reranker().score(question, [chunks[c.chunk_id].text for c in candidates])
        for c, s in zip(candidates, scores):
            c.rerank = s
        candidates.sort(key=lambda c: -(c.rerank or 0.0))
    timings["rerank"] = (time.perf_counter() - t) * 1000
    top = candidates[:n]
    gate = evidence_gate(top, tau)
    timings["retrieval_total"] = (time.perf_counter() - t0) * 1000
    return Retrieval(question, normalize(question), route, lists, candidates, top, gate, chunks,
                     {key: round(v, 1) for key, v in timings.items()})
