"""Metrics of the evaluation protocol (plan section 9.1).

Relevance is at PAGE level: a retrieved passage is relevant when it comes from the gold
document and the gold page (a question annotated twice accepts either annotator's page).
Rankings are first reduced to distinct pages, in order, so two chunks of one page count once.
"""

import math
from collections.abc import Iterable, Sequence

Page = tuple[str, int]  # (document sha256 or id, page number)


def distinct_pages(ranked: Iterable[Page]) -> list[Page]:
    seen, out = set(), []
    for p in ranked:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def recall_at(ranked: Sequence[Page], gold: set[Page], k: int) -> float:
    """1 if any gold page is in the top k distinct pages (one gold page per question, so this
    is the share of gold pages found)."""
    if not gold:
        return 0.0
    top = distinct_pages(ranked)[:k]
    return len(gold & set(top)) / len(gold) if len(gold) > 1 else float(bool(gold & set(top)))


def mrr_at(ranked: Sequence[Page], gold: set[Page], k: int = 10) -> float:
    for i, p in enumerate(distinct_pages(ranked)[:k], start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: Sequence[Page], gold: set[Page], k: int = 10) -> float:
    """Binary relevance; the ideal ranking puts the gold page(s) first."""
    pages = distinct_pages(ranked)[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(pages, start=1) if p in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_scores(ranked: Sequence[Page], gold: set[Page]) -> dict[str, float]:
    return {"recall@5": recall_at(ranked, gold, 5), "recall@20": recall_at(ranked, gold, 20),
            "mrr@10": mrr_at(ranked, gold, 10), "ndcg@10": ndcg_at(ranked, gold, 10)}


def citation_pr(cited: set[Page], gold: set[Page]) -> tuple[float | None, float | None]:
    """Page-level citation precision (share of cited pages that are gold) and recall (share of
    gold pages cited). None when undefined (nothing cited / no gold)."""
    precision = len(cited & gold) / len(cited) if cited else None
    recall = len(cited & gold) / len(gold) if gold else None
    return precision, recall


def abstention_pr(abstained: Sequence[bool], answerable: Sequence[bool]) -> dict:
    """Abstention as the positive class: precision = share of abstentions that were on
    unanswerable questions; recall = share of unanswerable questions abstained on."""
    tp = sum(1 for a, ok in zip(abstained, answerable) if a and not ok)
    n_abs = sum(1 for a in abstained if a)
    n_un = sum(1 for ok in answerable if not ok)
    return {"abstained": n_abs, "unanswerable": n_un,
            "precision": tp / n_abs if n_abs else None, "recall": tp / n_un if n_un else None,
            "wrongly_abstained": sum(1 for a, ok in zip(abstained, answerable) if a and ok)}


def risk_coverage(confidence: Sequence[float], error: Sequence[bool]) -> dict:
    """Selective prediction: answer the most confident questions first. At coverage k/N the
    risk is the share of errors among those k. AURC is the mean risk over all k (lower is
    better); `points` are (coverage, risk, threshold) for the curve."""
    order = sorted(range(len(confidence)), key=lambda i: -confidence[i])
    points, errors = [], 0
    for k, i in enumerate(order, start=1):
        errors += bool(error[i])
        points.append((k / len(order), errors / k, confidence[i]))
    aurc = sum(r for _, r, _ in points) / len(points) if points else None
    return {"aurc": aurc, "points": points}


def operating_point(confidence: Sequence[float], error: Sequence[bool], tau: float) -> dict:
    answered = [e for c, e in zip(confidence, error) if c >= tau]
    return {"tau": tau, "coverage": len(answered) / len(confidence) if confidence else None,
            "risk": sum(answered) / len(answered) if answered else None}


def choose_tau(confidence: Sequence[float], error: Sequence[bool], max_risk: float) -> dict:
    """The lowest threshold (largest coverage) whose risk stays at or below max_risk; for
    re-choosing tau on the dev split (plan section 6.7)."""
    best = None
    for coverage, risk, threshold in risk_coverage(confidence, error)["points"]:
        if risk <= max_risk:
            best = {"tau": threshold, "coverage": coverage, "risk": risk}
    return best or {"tau": None, "coverage": 0.0, "risk": None}


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile, q in [0, 100]."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_summary(rows: Sequence[dict]) -> dict[str, dict]:
    """p50 / p95 of every latency stage present in the rows (milliseconds)."""
    stages = sorted({k for r in rows for k in r})
    return {s: {"p50": percentile([r.get(s) for r in rows], 50), "p95": percentile([r.get(s) for r in rows], 95),
                "n": sum(1 for r in rows if r.get(s) is not None)} for s in stages}


def mean(values: Iterable[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None
