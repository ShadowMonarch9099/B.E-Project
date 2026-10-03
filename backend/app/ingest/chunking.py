"""Page-anchored chunking (plan section 6.2 step 6).

A chunk never crosses a page. Each page is split on paragraph and sentence boundaries into
chunks of about 200-350 tokens (the dense model's tokens), with a one-sentence overlap
between neighbouring chunks of the same page. A paragraph that continues on the next page
becomes two chunks, each citing its own page.

Every chunk is a contiguous span of its page's text (`char_start`, `char_end`), and keeps
the boxes of the lines it covers, so the viewer can highlight it on the page.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from . import tokens as tok

MIN_TOKENS = 200
MAX_TOKENS = 350
MIN_TAIL_TOKENS = 40

# Abbreviations whose full stop does not end a sentence. English ones are compared in
# lower case; Devanagari ones as written ("क्र." = number, "दि." = date, "श्री." = Shri).
ABBREVIATIONS = {
    "no", "nos", "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "eg", "e.g", "ie", "i.e",
    "art", "arts", "cl", "cls", "sec", "secs", "govt", "dept", "rs", "fig", "figs", "vol", "pp", "co", "ltd",
    "inc", "hon", "shri", "smt", "ref", "dt", "viz", "approx", "ch", "para", "sl", "dist", "tal", "est",
    "क्र", "क्रं", "सं", "दि", "डॉ", "श्री", "श्रीमती", "प्रा", "मा", "ता", "जि", "नं", "रु", "इ", "वि", "कि",
    "मि", "ग्रा", "कं", "पृ", "शा", "नि", "उ", "सौ", "कु", "ले", "प्र", "मु", "पो",
}
_BOUNDARY = re.compile(r"[.?!।॥]+[\"'”’)\]]*(?=\s)")
_TOKEN_BEFORE = re.compile(r"([^\s(\[\"'“‘]+)$")


def sentence_spans(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """(start, end) of each sentence in `text`, shifted by `offset`, whitespace trimmed."""
    spans = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        mark = m.group(0)
        if mark[0] == ".":
            before = _TOKEN_BEFORE.search(text, 0, m.start())
            word = before.group(1) if before else ""
            line_start = text.rfind("\n", 0, m.start()) + 1
            if (word.lower() in ABBREVIATIONS or word in ABBREVIATIONS
                    or (len(word) == 1 and word.isalpha())  # an initial: "R. K. Rao", "क. ख."
                    or (word.isdigit() and len(word) <= 2 and before.start(1) == line_start)):  # "1. Scope"
                continue
            nxt = text[m.end():m.end() + 2].strip()[:1]
            if nxt and nxt.isascii() and nxt.islower():  # "approx. two", "i.e. the"
                continue
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s + offset, e + offset))
    return out


@dataclass
class Unit:
    start: int
    end: int
    para: int
    tokens: int


@dataclass
class ChunkDraft:
    page_no: int
    para_idx: int
    chunk_idx: int
    char_start: int
    char_end: int
    text_raw: str
    bbox: list[list[float]]
    ocr_conf: float | None
    token_count: int


def _paragraph_spans(layout: list) -> list[tuple[int, int]]:
    spans = []
    for para in layout:
        words = [w for line in para for w in line]
        if words:
            spans.append((words[0]["s"], words[-1]["s"] + len(words[-1]["t"])))
    return spans


def _split_long(text: str, s: int, e: int, para: int, count: Callable[[str], int]) -> list[Unit]:
    """Split an over-long sentence at line breaks, then between words."""
    pieces: list[tuple[int, int]] = []
    pos = s
    for line in text[s:e].split("\n"):
        if line.strip():
            ls = text.index(line, pos)
            pieces.append((ls, ls + len(line)))
            pos = ls + len(line)
    units: list[Unit] = []
    for ps, pe in pieces:
        n = count(text[ps:pe])
        if n <= MAX_TOKENS:
            units.append(Unit(ps, pe, para, n))
            continue
        cur_s = cur_e = None
        for m in re.finditer(r"\S+", text[ps:pe]):
            ws, we = ps + m.start(), ps + m.end()
            if cur_s is None:
                cur_s, cur_e = ws, we
                continue
            if count(text[cur_s:we]) > MAX_TOKENS:
                units.append(Unit(cur_s, cur_e, para, count(text[cur_s:cur_e])))
                cur_s = ws
            cur_e = we
        if cur_s is not None:
            units.append(Unit(cur_s, cur_e, para, count(text[cur_s:cur_e])))
    return units


def _units(text: str, layout: list, count: Callable[[str], int]) -> list[Unit]:
    units: list[Unit] = []
    for para, (ps, pe) in enumerate(_paragraph_spans(layout)):
        spans = sentence_spans(text[ps:pe], ps)
        counts = tok.count_many([text[s:e] for s, e in spans]) if count is tok.count else \
            [count(text[s:e]) for s, e in spans]
        for (s, e), n in zip(spans, counts):
            if n > MAX_TOKENS:
                units.extend(_split_long(text, s, e, para, count))
            else:
                units.append(Unit(s, e, para, n))
    return units


def _boxes(layout: list, start: int, end: int) -> tuple[list[list[float]], float | None]:
    """Line boxes and mean OCR confidence (0-1) of the words in [start, end)."""
    boxes, confs = [], []
    for para in layout:
        for line in para:
            box = None
            for w in line:
                ws, we = w["s"], w["s"] + len(w["t"])
                if we <= start or ws >= end:
                    continue
                if w.get("c") is not None:
                    confs.append(w["c"])
                b = w.get("b")
                if b is None:
                    continue
                box = list(b) if box is None else [min(box[0], b[0]), min(box[1], b[1]),
                                                   max(box[2], b[2]), max(box[3], b[3])]
            if box is not None:
                boxes.append(box)
    conf = round(sum(confs) / len(confs) / 100, 4) if confs else None
    return boxes, conf


def chunk_page(page_no: int, text: str, layout: list,
               count: Callable[[str], int] = tok.count) -> list[ChunkDraft]:
    """Chunks of one page. `layout` is the page layout with word offsets (pages.build_text)."""
    units = _units(text, layout, count)
    groups: list[tuple[list[Unit], int]] = []  # units, index of the first unit that is not overlap
    cur: list[Unit] = []
    fresh = 0  # where the new (non-overlap) units of `cur` start

    def tokens_of(us: list[Unit]) -> int:
        return count(text[us[0].start:us[-1].end]) if us else 0

    def emit() -> None:
        nonlocal cur, fresh
        if len(cur) > fresh:  # never emit a chunk that is only the overlap
            groups.append((cur, fresh))
            last = cur[-1]
            cur, fresh = ([last], 1) if last.tokens <= MAX_TOKENS // 2 else ([], 0)

    prev_para = None
    for u in units:
        if prev_para is not None and u.para != prev_para and len(cur) > fresh and tokens_of(cur) >= MIN_TOKENS:
            emit()  # prefer to cut at a paragraph boundary once the chunk is big enough
        if len(cur) > fresh and tokens_of(cur + [u]) > MAX_TOKENS:
            emit()
        if cur and tokens_of(cur + [u]) > MAX_TOKENS:  # the overlap sentence does not fit with u
            cur, fresh = [], 0
        cur.append(u)
        prev_para = u.para
    emit()

    # A scrap at the end of the page (a signature line, a page number) joins the chunk before it.
    if len(groups) > 1:
        (prev, prev_first), (last, last_first) = groups[-2], groups[-1]
        tail = last[last_first:]
        if tokens_of(tail) < MIN_TAIL_TOKENS and tokens_of(prev + tail) <= MAX_TOKENS:
            groups[-2:] = [(prev + tail, prev_first)]

    chunks = []
    for i, (us, first_new) in enumerate(groups):
        start, end = us[0].start, us[-1].end
        boxes, conf = _boxes(layout, start, end)
        chunks.append(ChunkDraft(page_no, us[first_new].para, i, start, end, text[start:end], boxes, conf,
                                 tokens_of(us)))
    return chunks
