"""Numbers, dates and identifiers in questions, matched exactly (plan section 6.3, N2).

Dense embeddings blur numbers, and a scanned GR prints "सर्व्हे क्र. १४२" where the user types
"Survey No. 142". Digits are mapped to ASCII on both sides (app/ingest/normalize.py), and the
numbers a question contains are matched as whole tokens against chunk text, so "142" never
matches "1420" or "138". The chunks that contain them form an extra candidate list for
fusion (the "exact" list).
"""

import re
from collections import Counter

from sqlalchemy import select

from ..db import Chunk, Document, session_scope
from ..ingest.normalize import to_ascii_digits

# A number, possibly with internal separators: 142, 4(1), 21.04.2015, 2023/17, 60:40, 3,00,000
_NUMBER = re.compile(r"\d+(?:[.,/:\-]\d+)*(?:\(\d+\))*")
YEAR_LIKE = re.compile(r"^(19|20)\d\d$")


def extract_numbers(text: str) -> list[str]:
    """Distinct numbers/IDs in a question, ASCII digits, in order. Single digits are skipped
    (they are usually list markers or counts, and match almost every chunk)."""
    out: list[str] = []
    for m in _NUMBER.finditer(to_ascii_digits(text)):
        token = m.group(0).strip(".,:/-")
        if len(token.replace(",", "")) < 2:
            continue
        if token not in out:
            out.append(token)
    return out


def _variants(number: str) -> list[str]:
    """Forms the same number takes in documents: 300000 ~ 3,00,000 ~ 300,000."""
    plain = number.replace(",", "")
    forms = {number, plain}
    if plain.isdigit() and len(plain) > 3:
        last3, rest = plain[-3:], plain[:-3]
        groups = []
        while len(rest) > 2:  # Indian grouping: 3,00,000
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        forms.add(",".join(groups + [last3]))
        forms.add(f"{int(plain):,}")
    return sorted(forms, key=len, reverse=True)


def _pattern(number: str) -> re.Pattern:
    alternatives = "|".join(re.escape(v) for v in _variants(number))
    return re.compile(rf"(?<![\d.,/])(?:{alternatives})(?![\d])")


def exact_matches(numbers: list[str], doc_ids: list[str] | None, k: int = 40) -> list[tuple[int, float]]:
    """Chunks containing the question's numbers as whole tokens, best first.

    Score = number of distinct numbers matched (a chunk with "142" and "2019" beats one with
    only "2019"); a bare year counts half, since years appear everywhere.
    """
    if not numbers:
        return []
    patterns = {n: _pattern(n) for n in numbers}
    weights = {n: 0.5 if YEAR_LIKE.match(n) else 1.0 for n in numbers}
    with session_scope() as s:
        stmt = (select(Chunk.id, Chunk.text_norm).join(Document, Document.id == Chunk.doc_id)
                .where(Document.status == "indexed"))
        if doc_ids is not None:
            stmt = stmt.where(Chunk.doc_id.in_(list(doc_ids)))
        rows = s.execute(stmt).all()
    scores: Counter = Counter()
    for cid, text in rows:
        for n, pat in patterns.items():
            if pat.search(text):
                scores[cid] += weights[n]
    return [(cid, float(score)) for cid, score in scores.most_common(k)]
