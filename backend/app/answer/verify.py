"""Verification of an answer, sentence by sentence (plan section 6.7 step 3, N4).

1. Citation check: the answer is split into sentences, and a sentence counts as cited when
   any part of it falls in a text block the API marked with a citation. Uncited sentences are
   removed (and listed, so the UI can say what was dropped).
2. Number check: every number of two or more digits in a sentence must occur in a passage the
   sentence cites (Devanagari digits read as ASCII; commas and the separators . / - : ignored,
   and a part of a passage number such as "2026" of "2026-27" counts). A number written
   differently from the passage ("13,27.74" for "१३२७.७४") is rewritten as the passage has it;
   a sentence with a number found in none of its passages is removed. Single digits and numbers
   written in words are not checked. The pretrained NLI model could not tell "तीन लाख" from
   "पाच लाख"; this check catches wrong figures without a model.
3. Optional NLI check (SAKSHYA_NLI=on): each remaining sentence must be entailed by at least
   one of the passages it cites (app/ml/nli.py); sentences that are not are removed too.
4. If no sentence survives, or fewer than a third of them do, the answer becomes an
   abstention: the model's text was not supported well enough by the passages.
"""

import re
from dataclasses import asdict, dataclass, field

from ..ingest.chunking import sentence_spans
from ..ingest.normalize import to_ascii_digits, to_devanagari_digits
from ..llm.base import Citation, Segment
from ..ml import nli as nli_mod

MIN_KEPT_SHARE = 1 / 3
_WORDS = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


@dataclass
class Sentence:
    text: str
    citations: list[Citation] = field(default_factory=list)
    kept: bool = True
    reason: str | None = None  # why it was removed: "no_citation", "number_not_in_passage", "not_entailed"
    numbers_fixed: list[tuple[str, str]] = field(default_factory=list)  # (as written, as the passage has it)
    entailment: float | None = None

    @property
    def chunk_ids(self) -> list[int]:
        return list(dict.fromkeys(c.chunk_id for c in self.citations))


@dataclass
class Verification:
    sentences: list[Sentence]
    abstain: bool
    reason: str | None
    nli: bool

    @property
    def kept(self) -> list[Sentence]:
        return [s for s in self.sentences if s.kept]

    @property
    def removed(self) -> list[Sentence]:
        return [s for s in self.sentences if not s.kept]

    def as_log(self) -> dict:
        return {"sentences": len(self.sentences), "kept": len(self.kept), "abstain": self.abstain,
                "reason": self.reason, "nli": self.nli,
                "removed": [{"text": s.text, "reason": s.reason, "entailment": s.entailment} for s in self.removed],
                "numbers_fixed": [list(f) for s in self.kept for f in s.numbers_fixed],
                "entailment": [s.entailment for s in self.kept]}


def split_sentences(segments: list[Segment]) -> list[Sentence]:
    """Sentences of the whole answer, each with the citations of the blocks it overlaps."""
    text, owners = "", []  # owners[i] = index of the segment that wrote character i
    for i, seg in enumerate(segments):
        text += seg.text
        owners.extend([i] * len(seg.text))
    out: list[Sentence] = []
    offset = 0
    for line in text.split("\n"):
        for s, e in sentence_spans(line, offset):
            body = text[s:e].strip()
            if not _WORDS.search(body):  # punctuation or a bare number: glue, not a claim
                if out and body:
                    out[-1].text += body if body[0] in ".,;:!?।)" else " " + body
                continue
            segs = sorted(set(owners[s:e]))
            cits = [c for i in segs for c in segments[i].citations]
            out.append(Sentence(body, cits))
        offset += len(line) + 1
    return out


_NUMBER = re.compile(r"[0-9०-९]+(?:[.,/:\-][0-9०-९]+)*")


def _key(number: str) -> str:
    return re.sub(r"[./:\-]", ".", to_ascii_digits(number).replace(",", ""))


def _passage_numbers(text: str) -> dict[str, str]:
    """key -> the number as the passage writes it, for whole numbers and their parts."""
    out: dict[str, str] = {}
    for m in _NUMBER.finditer(text):
        surface = m.group(0).strip(".,:/-")
        key = _key(surface)
        out.setdefault(key, surface)
        for part in key.split("."):
            out.setdefault(part, part)
    return out


def check_numbers(sentence: Sentence, passages: dict[int, str]) -> None:
    """Rewrite numbers to the form of the cited passage; remove the sentence if one is missing."""
    known: dict[str, str] = {}
    for cid in sentence.chunk_ids:
        if cid in passages:
            for k, v in _passage_numbers(passages[cid]).items():
                known.setdefault(k, v)
    text = sentence.text
    for m in reversed(list(_NUMBER.finditer(text))):
        surface = m.group(0).strip(".,:/-")
        key = _key(surface)
        if len(key.replace(".", "")) < 2:
            continue
        if key not in known:
            sentence.kept, sentence.reason = False, "number_not_in_passage"
            return
        source = known[key]
        want = to_devanagari_digits(to_ascii_digits(source)) if re.search(r"[०-९]", surface) else to_ascii_digits(source)
        if want != surface:
            start = m.start() + m.group(0).index(surface)
            text = text[:start] + want + text[start + len(surface):]
            sentence.numbers_fixed.insert(0, (surface, want))
    sentence.text = text


def verify(segments: list[Segment], passages: dict[int, str], use_nli: bool | None = None) -> Verification:
    """`passages`: chunk id -> passage text, for the number and NLI checks."""
    use_nli = nli_mod.NLI_ON if use_nli is None else use_nli
    sentences = split_sentences(segments)
    for s in sentences:
        if not s.citations:
            s.kept, s.reason = False, "no_citation"
        else:
            check_numbers(s, passages)
    if use_nli:
        pairs, owners = [], []
        for idx, s in enumerate(sentences):
            if not s.kept:
                continue
            for cid in s.chunk_ids:
                if cid in passages:
                    pairs.append((passages[cid], s.text))
                    owners.append(idx)
        if pairs:
            scores = nli_mod.get_nli().entailment(pairs)
            best: dict[int, float] = {}
            for idx, score in zip(owners, scores):
                best[idx] = max(score, best.get(idx, 0.0))
            for idx, score in best.items():
                sentences[idx].entailment = round(score, 4)
                if score < nli_mod.THRESHOLD:
                    sentences[idx].kept, sentences[idx].reason = False, "not_entailed"
    kept = [s for s in sentences if s.kept]
    if not kept:
        return Verification(sentences, True, "No sentence of the answer was supported by a cited passage.", use_nli)
    if len(kept) / len(sentences) < MIN_KEPT_SHARE:
        return Verification(sentences, True, f"Only {len(kept)} of {len(sentences)} sentences were supported "
                                             "by a cited passage.", use_nli)
    return Verification(sentences, False, None, use_nli)


def sentence_view(s: Sentence) -> dict:
    return {"text": s.text, "chunk_ids": s.chunk_ids, "entailment": s.entailment,
            **({"reason": s.reason} if s.reason else {})}


def as_dict(v: Verification) -> dict:
    return asdict(v)
