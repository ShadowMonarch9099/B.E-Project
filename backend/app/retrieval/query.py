"""Query understanding (plan section 6.3, N2): the views of a question that retrieval searches.

For each question:
- script and language with IndicLID (romanized Hindi/Marathi told apart from English);
- a code-mix ratio for romanized questions (share of English words, e.g. "ATKT", "allowed");
- view "as_typed";
- view "devanagari": an IndicXlit transliteration, if the question is romanized Hindi or
  Marathi ("gharkul yojana" -> "घरकुल योजना"); acronyms and numbers are kept as written;
- view "english": a translation to English (IndicTrans2, or OPUS-MT until IndicTrans2 is
  available), if the question is Hindi or Marathi;
- numbers, dates and IDs for exact matching, with Devanagari digits mapped to ASCII.

Every view and decision is stored in query_logs. Each component can be switched off for the
ablations in plan section 9.3 (AnalysisOptions).
"""

import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field

from ..ingest.normalize import normalize, script_of
from .numbers import extract_numbers

log = logging.getLogger("sakshya.query")

_TOKEN = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


# The unit tests switch the language models off (SAKSHYA_QUERY_MODELS=off); they are 1 GB+.
MODELS_ON = os.environ.get("SAKSHYA_QUERY_MODELS", "on") != "off"


@dataclass
class AnalysisOptions:
    langid: bool = MODELS_ON
    transliterate: bool = MODELS_ON
    translate: bool = MODELS_ON
    digits: bool = True  # Devanagari digits -> ASCII on the query side
    numbers: bool = True  # exact matching of numbers and IDs


@dataclass
class QueryAnalysis:
    as_typed: str
    normalized: str
    script: str | None  # devanagari / latin / mixed / None
    lang: str  # en / hi / mr / other
    lang_label: str | None = None  # IndicLID's raw label
    lang_confidence: float | None = None
    lang_scores: dict = field(default_factory=dict)
    romanized: bool = False
    code_mix: float = 0.0
    english_words: list[str] = field(default_factory=list)
    devanagari: str | None = None
    english: str | None = None
    english_model: str | None = None
    numbers: list[str] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)  # a model that failed; the question still runs

    def views(self) -> dict[str, str]:
        """The texts retrieval searches with, in a stable order."""
        out = {"as_typed": self.normalized}
        if self.devanagari:
            out["devanagari"] = self.devanagari
        if self.english:
            out["english"] = self.english
        return out

    def as_log(self) -> dict:
        return asdict(self)


def transliterate_query(text: str, lang: str) -> str:
    """Romanized words -> Devanagari with IndicXlit; acronyms, numbers and punctuation kept."""
    from ..ml.xlit import transliterate_word

    parts = []
    for tok in _TOKEN.findall(text):
        if tok.isalpha() and not (len(tok) >= 2 and tok.isupper()):
            parts.append(transliterate_word(tok.lower(), lang))
        else:
            parts.append(tok)
    return "".join(parts)


def analyse_query(question: str, options: AnalysisOptions | None = None) -> QueryAnalysis:
    opts = options or AnalysisOptions()
    timings: dict[str, float] = {}
    norm = normalize(question, digits=opts.digits)
    a = QueryAnalysis(question, norm, script_of(question), "other")

    def step(name: str, fn) -> None:
        t = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # a missing or broken model must not stop the question
            log.warning("Query %s failed: %s", name, exc)
            a.errors[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
        timings[name] = (time.perf_counter() - t) * 1000

    def langid() -> None:
        from ..ml.langid import get_langid

        lid = get_langid()
        r = lid.identify(question)
        a.lang, a.lang_label, a.lang_confidence, a.lang_scores, a.romanized = (
            r.lang, r.label, r.confidence, r.scores, r.romanized)
        if r.romanized:
            words = re.findall(r"[A-Za-z]+", question)
            a.english_words = [w for w in words if lid.is_english_word(w)]
            a.code_mix = round(len(a.english_words) / len(words), 3) if words else 0.0

    def transliterate() -> None:
        a.devanagari = normalize(transliterate_query(question, a.lang), a.lang, digits=opts.digits)

    def translate() -> None:
        from ..ml.translate import get_translator

        source = a.devanagari or (norm if a.script in ("devanagari", "mixed") else None)
        if source:
            tr = get_translator()
            a.english = (tr.to_english([source], a.lang)[0] or "").strip() or None
            a.english_model = tr.label(a.lang)

    if opts.langid:
        step("langid", langid)
    elif a.script == "devanagari":
        a.lang = "hi"  # without IndicLID: Devanagari is Hindi or Marathi; the scores stay empty
    elif a.script == "latin":
        a.lang = "en"
    if opts.transliterate and a.romanized and a.lang in ("hi", "mr"):
        step("transliterate", transliterate)
    if opts.translate and a.lang in ("hi", "mr"):
        step("translate", translate)
    if opts.numbers:
        a.numbers = extract_numbers(question)
    a.latency_ms = {k: round(v, 1) for k, v in timings.items()}
    return a
