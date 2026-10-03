"""Retrieval routes, fusion, reranking and the evidence gate (plan sections 6.4 and 6.7 step 1).

direct:    E5 dense search of the chunks as written, with the question's views.
translate: the question in English (as typed, or its translation) against the English shadow
           index (each chunk translated to English and embedded with E5).
hybrid:    E5 dense + BM25 words + BM25 character n-grams (+ exact number/ID matches),
           fused with weighted reciprocal rank fusion (RRF).
Every route: top 40 -> the same cross-encoder, on the chunks as written -> top 6, so the
routes differ only in the candidates they find (plan section 6.4; RQ1 compares them).

Unless a route is forced, the rule-based router (app/retrieval/router.py) chooses one. All
first-stage lists are computed once; each route's top three candidates are reranked (the
"probe"), which gives the router comparable scores across routes and goes to query_logs.

OCR-aware fusion (N3, section 6.6): in the hybrid route, a chunk from a low-confidence scan
gets more weight from the character n-gram list and less from the word list, because OCR
errors break whole words but leave most of their character n-grams intact.

Every route searches with all of the question's views (plan section 6.3, app/retrieval/
query.py): as typed, the Devanagari transliteration of a romanized question, and the English
translation of a Hindi/Marathi one. Dense search runs once per view and keeps each chunk's
best score; BM25 searches the views together; the reranker scores each passage against the
question as typed and its Devanagari view and keeps the higher score.

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
from ..ml.embedder import get_embedder
from ..ml.reranker import get_reranker
from .chunks import ChunkInfo, load_chunks
from .numbers import exact_matches
from .query import AnalysisOptions, QueryAnalysis, analyse_query
from .router import Decision, choose, features

RRF_K = 60
TOP_K = int(os.environ.get("SAKSHYA_TOP_K", "40"))
TOP_N = int(os.environ.get("SAKSHYA_TOP_N", "6"))
WEIGHTS = {"dense": 1.0, "word": 1.0, "char": 0.6, "exact": 0.8, "translate": 1.0}
# Measured with tools/gate_probe.py on 24 hand-written questions over two of the team's
# sample GRs. With the as-typed question only (phase 4): answerable 0.077-1.000, unanswerable
# 0.002-0.063, so tau was 0.07. With the transliterated and translated views (phase 6), the
# romanized answerable questions rose (0.077 -> 0.992) but so did some romanized off-topic ones
# ("RUSA 3.0 ...", a near-miss on a GR about RUSA 1.0/2.0: 0.063 -> 0.401): answerable
# 0.171-1.000, unanswerable 0.002-0.401. tau = 0.15 answers 14/14 and abstains on 9/10.
# A small sample; to be re-chosen from the risk-coverage curve on the Sakshya-Bench dev split.
GATE_TAU = float(os.environ.get("SAKSHYA_GATE_TAU", "0.15"))
# A document small enough that all of it goes to the answer model (at most TOP_N passages) is
# not gated: the reranker scores broad questions low ("are there any technical skills in this
# resume?" scored 0.012 against the passage listing them), and with the whole document in view
# the model can say it does not cover the question; the citation check still applies.
SMALL_DOC_CHUNKS = int(os.environ.get("SAKSHYA_SMALL_DOC_CHUNKS", str(TOP_N)))
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
    analysis: QueryAnalysis | None = None
    decision: "Decision | None" = None

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


OCR_LOW = 0.85  # below this page OCR confidence, fusion shifts toward character n-grams
OCR_AWARE = True


def ocr_weight(name: str, base: float, conf: float | None) -> float:
    """The weight of one list for one chunk, given the chunk's OCR confidence (None = native)."""
    if conf is None or conf >= OCR_LOW:
        return base
    damage = (OCR_LOW - conf) / OCR_LOW  # 0 at the threshold, up to 1 for an unreadable page
    if name == "char":
        return base * (1 + 1.5 * damage)
    if name == "word":
        return base * (1 - 0.5 * damage)
    return base


def fuse(lists: dict[str, list[tuple[int, float]]], weights: dict[str, float],
         ocr: dict[int, float | None] | None = None) -> list[Candidate]:
    """Weighted reciprocal rank fusion: sum over lists of weight / (60 + rank). With `ocr`
    (chunk id -> OCR confidence), the word and character weights follow each chunk's OCR."""
    out: dict[int, Candidate] = {}
    for name, hits in lists.items():
        base = weights.get(name, 0.0)
        for rank, (cid, score) in enumerate(hits, start=1):
            c = out.setdefault(cid, Candidate(cid))
            c.ranks[name] = rank
            c.scores[name] = score
            w = ocr_weight(name, base, ocr.get(cid)) if ocr else base
            c.fused += w / (RRF_K + rank)
    return sorted(out.values(), key=lambda c: -c.fused)


def evidence_gate(top: list[Candidate], tau: float = GATE_TAU) -> GateDecision:
    best = max((c.rerank or 0.0 for c in top), default=0.0)
    if not top:
        return GateDecision(False, 0.0, tau, "nothing was retrieved")
    if best < tau:
        return GateDecision(False, best, tau, f"best passage scores {best:.2f}, below the threshold {tau:.2f}")
    return GateDecision(True, best, tau, f"best passage scores {best:.2f} (threshold {tau:.2f})")


def dense_lists(analysis: QueryAnalysis, doc_ids: list[str] | None, k: int, view: str = "passage",
                timings: dict | None = None) -> list[tuple[int, float]]:
    """Dense search with every view of the question; each chunk keeps its best score."""
    timings = {} if timings is None else timings
    texts = list(analysis.views().values())
    t = time.perf_counter()
    vecs = get_embedder().embed_queries(texts)
    timings["embed"] = timings.get("embed", 0.0) + (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    best: dict[int, float] = {}
    for vec in vecs:
        for cid, score in dense.search(vec, doc_ids, k, view=view):
            best[cid] = max(score, best.get(cid, -1.0))
    timings["dense"] = timings.get("dense", 0.0) + (time.perf_counter() - t) * 1000
    return sorted(best.items(), key=lambda kv: -kv[1])[:k]


ROUTES = ("direct", "translate", "hybrid")


def translate_list(analysis: QueryAnalysis, doc_ids: list[str] | None, k: int,
                   timings: dict | None = None) -> list[tuple[int, float]]:
    """The translate route: the question in English against the English shadow index."""
    timings = {} if timings is None else timings
    query = analysis.english or (analysis.normalized if analysis.lang == "en" else None)
    if not query:
        return []
    t = time.perf_counter()
    vec = get_embedder().embed_queries([query])[0]
    timings["embed"] = timings.get("embed", 0.0) + (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    hits = dense.search(vec, doc_ids, k, view="en")
    timings["dense_en"] = (time.perf_counter() - t) * 1000
    return hits


def first_stage(analysis: QueryAnalysis, doc_ids: list[str] | None, route: str = "hybrid",
                k: int = TOP_K, timings: dict | None = None) -> dict[str, list[tuple[int, float]]]:
    """The candidate lists a route draws from, before fusion."""
    timings = {} if timings is None else timings
    if route == "translate":
        hits = translate_list(analysis, doc_ids, k, timings)
        if hits:
            return {"translate": hits}
        timings["translate_fallback"] = 1.0  # no shadow index or no English view: search as direct
        return {"dense": dense_lists(analysis, doc_ids, k, timings=timings)}
    lists = {"dense": dense_lists(analysis, doc_ids, k, timings=timings)}
    if route == "hybrid":
        t = time.perf_counter()
        lexical = get_lexical_index().search(" ".join(analysis.views().values()), doc_ids, k)
        lists["word"] = [(h.chunk_id, h.score) for h in lexical["word"]]
        lists["char"] = [(h.chunk_id, h.score) for h in lexical["char"]]
        if analysis.numbers:
            lists["exact"] = exact_matches(analysis.numbers, doc_ids, k)
        timings["lexical"] = (time.perf_counter() - t) * 1000
    return lists


def rerank(question_views: list[str], candidates: list[Candidate], chunks: dict[int, ChunkInfo],
           reorder: bool = True) -> None:
    """Score every candidate with the cross-encoder; keep the best score over the views.
    reorder=False keeps the fused order (the no-reranker baselines and ablation)."""
    if not candidates:
        return
    passages = [chunks[c.chunk_id].text for c in candidates]
    best = [0.0] * len(candidates)
    for view in question_views:
        for i, score in enumerate(get_reranker().score(view, passages)):
            best[i] = max(best[i], score)
    for c, score in zip(candidates, best):
        c.rerank = score
    if reorder:
        candidates.sort(key=lambda c: -(c.rerank or 0.0))


ROUTE_LISTS = {"direct": ("dense",), "translate": ("translate",), "hybrid": ("dense", "word", "char", "exact")}
PROBE_TOP = 3


def _ocr_of(ids: list[int]) -> dict[int, float | None]:
    if not ids:
        return {}
    from sqlalchemy import select

    from ..db import Chunk, session_scope

    with session_scope() as s:
        return dict(s.execute(select(Chunk.id, Chunk.ocr_conf).where(Chunk.id.in_(ids))).all())


def _chunk_count(doc_ids: list[str]) -> int:
    from sqlalchemy import func, select

    from ..db import Chunk, session_scope

    with session_scope() as s:
        return s.scalar(select(func.count(Chunk.id)).where(Chunk.doc_id.in_(doc_ids))) or 0


def all_lists(analysis: QueryAnalysis, doc_ids: list[str] | None, k: int, timings: dict) -> dict:
    """Every first-stage list, computed once: dense, word, char, exact, translate."""
    lists = first_stage(analysis, doc_ids, "hybrid", k, timings)
    lists["translate"] = translate_list(analysis, doc_ids, k, timings)
    return lists


def route_lists(lists: dict, route: str, timings: dict | None = None) -> dict:
    chosen = {name: lists[name] for name in ROUTE_LISTS[route] if lists.get(name)}
    if route == "translate" and not chosen:  # no shadow index or no English view
        if timings is not None:
            timings["translate_fallback"] = 1.0
        chosen = {"dense": lists.get("dense", [])}
    return chosen


def retrieve(question: str, doc_ids: list[str] | None, route: str | None = "hybrid", **kwargs) -> Retrieval:
    """See _retrieve. While it runs, background translation (the shadow index) pauses."""
    from .. import busy

    with busy.serving():
        return _retrieve(question, doc_ids, route, **kwargs)


def _retrieve(question: str, doc_ids: list[str] | None, route: str | None = "hybrid", k: int = TOP_K,
              n: int = TOP_N, tau: float = GATE_TAU, weights: dict[str, float] | None = None,
              analysis: QueryAnalysis | None = None, options: AnalysisOptions | None = None,
              probe: bool = True, ocr_aware: bool | None = None, use_reranker: bool = True) -> Retrieval:
    """Retrieve for one question. route=None lets the router choose; probe=False skips the
    per-route probe (cheaper; used by evaluations of a fixed route). use_reranker=False keeps
    the fused order and scores only the top n with the cross-encoder, for the gate (the
    no-reranker baselines and ablation, plan sections 9.2 and 9.3)."""
    if route is not None and route not in ROUTES:
        raise ValueError(f"unknown route {route}")
    weights = weights or WEIGHTS
    ocr_aware = OCR_AWARE if ocr_aware is None else ocr_aware
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if analysis is None:
        analysis = analyse_query(question, options)
    timings.update(analysis.latency_ms)
    rerank_views = [question] + ([analysis.devanagari] if analysis.devanagari else [])

    if route is not None and not probe:
        lists = route_lists(first_stage(analysis, doc_ids, route, k, timings), route, timings)
        decision = None
    else:
        lists_all = all_lists(analysis, doc_ids, k, timings)
        t = time.perf_counter()
        probe_ids = {r: [c.chunk_id for c in fuse(route_lists(lists_all, r), weights)[:PROBE_TOP]] for r in ROUTES}
        probe_chunks = load_chunks(sorted({cid for ids in probe_ids.values() for cid in ids}))
        probe_cands = {cid: Candidate(cid) for cid in probe_chunks}
        rerank(rerank_views, list(probe_cands.values()), probe_chunks)
        probes = {}
        for r, ids in probe_ids.items():
            scored = [(cid, probe_cands[cid].rerank or 0.0) for cid in ids if cid in probe_cands]
            probes[r] = {"best": round(max((sc for _, sc in scored), default=0.0), 4),
                         "top": [{"chunk_id": cid, "page_no": probe_chunks[cid].page_no, "rerank": round(sc, 4)}
                                 for cid, sc in scored]}
        timings["probe"] = (time.perf_counter() - t) * 1000
        top_dense = [cid for cid, _ in lists_all.get("dense", [])[:5]]
        ocr_top = _ocr_of(top_dense)
        feats = features(analysis, doc_ids, [ocr_top.get(cid) for cid in top_dense])
        if route is None:
            decision = choose(feats, probes)
        else:
            decision = Decision(route, "forced", "The route was set by the request.", feats, probes)
        route = decision.route
        lists = route_lists(lists_all, route, timings)

    ids = sorted({cid for hits in lists.values() for cid, _ in hits})
    ocr_map = _ocr_of(ids) if (ocr_aware and route == "hybrid") else None
    candidates = fuse(lists, weights, ocr_map)[:k]
    chunks = load_chunks([c.chunk_id for c in candidates])
    candidates = [c for c in candidates if c.chunk_id in chunks]

    t = time.perf_counter()
    if use_reranker:
        rerank(rerank_views, candidates, chunks)
    else:
        rerank(rerank_views, candidates[:n], chunks, reorder=False)
    timings["rerank"] = (time.perf_counter() - t) * 1000
    top = candidates[:n]
    gate = evidence_gate(top, tau)
    if not gate.passed and top and doc_ids and _chunk_count(doc_ids) <= min(n, SMALL_DOC_CHUNKS):
        gate = GateDecision(True, gate.score, tau, f"the whole document ({len(top)} passages) goes to the answer "
                                                   "model, which says if it does not cover the question")
    timings["retrieval_total"] = (time.perf_counter() - t0) * 1000
    r = Retrieval(question, analysis.normalized, route, lists, candidates, top, gate, chunks,
                  {key: round(v, 1) for key, v in timings.items()}, analysis)
    r.decision = decision
    return r
