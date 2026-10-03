"""Inter-annotator agreement for Sakshya-Bench (plan section 8.2).

- Cohen's kappa on the gold page: each double-annotated question gives two labels, the
  page each annotator chose ("<document>#<page>", or "none" for unanswerable). Kappa
  corrects the observed agreement for the agreement expected by chance from each
  annotator's own label distribution.
- Token-F1 on the evidence spans: the overlap of the two spans' tokens (as in SQuAD),
  averaged over questions where both annotators marked a span.
"""

from collections import Counter

from ..index.lexical import word_tokens
from ..ingest.normalize import normalize


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two annotators' labels on the same items; None with no items."""
    if len(a) != len(b):
        raise ValueError("both annotators must label the same items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def tokens(text: str) -> list[str]:
    return word_tokens(normalize(text or ""))


def token_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(ta), common / len(tb)
    return 2 * precision * recall / (precision + recall)


def page_label(doc_key: str | None, page: int | None) -> str:
    return f"{doc_key}#{page}" if doc_key and page else "none"


def agreement(pairs: list[tuple[dict, dict]]) -> dict:
    """pairs: (primary, second) annotations as dicts with doc, page and span_text."""
    a = [page_label(p.get("doc"), p.get("page")) for p, _ in pairs]
    b = [page_label(s.get("doc"), s.get("page")) for _, s in pairs]
    spans = [token_f1(p["span_text"], s["span_text"]) for p, s in pairs if p.get("span_text") and s.get("span_text")]
    return {
        "double_annotated": len(pairs),
        "page_agreement": round(sum(x == y for x, y in zip(a, b)) / len(pairs), 3) if pairs else None,
        "page_kappa": None if not pairs else round(cohen_kappa(a, b), 3),
        "span_pairs": len(spans),
        "span_token_f1": round(sum(spans) / len(spans), 3) if spans else None,
    }
