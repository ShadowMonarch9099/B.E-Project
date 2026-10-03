"""Script-faithful answering (plan section 6.7 step 4, N5).

The answer's language and script are compared with the question's (IndicLID on both):
- English question -> English answer;
- Devanagari Hindi/Marathi question -> an answer in the same language, in Devanagari;
- romanized Hindi/Marathi question -> an answer in the same language, in Latin letters,
  with a "Show in Devanagari" view (IndicXlit, roman -> Devanagari).

The answer language is also stated with every question (language_instruction), which is what
mostly keeps the model in the right language. Mismatches are repaired sentence by sentence, so
every sentence keeps its citations:
- right language, wrong script: IndicXlit (Devanagari -> roman, or roman -> Devanagari);
- wrong language: the answer model rewrites its own sentences into the right language
  (apply_rewrite). The rewrite is accepted only if every sentence keeps its numbers; otherwise
  the answer is shown as the model wrote it, with a note. Machine translation with NLLB-200 is
  no longer used here: on 2026-10-04 it turned a correct Marathi answer into English with an
  invented sentence ("She is the first woman to be elected to the Senate.").
What was checked and what was repaired is returned and logged (query_logs.fidelity).
"""

import logging
from dataclasses import dataclass, field

from ..ingest.normalize import script_of
from ..retrieval.query import MODELS_ON, QueryAnalysis

log = logging.getLogger("sakshya.fidelity")


@dataclass
class Fidelity:
    expected: dict
    found: dict
    lang_match: bool | None
    script_match: bool | None
    repair: str | None = None  # "transliterated" / "translated" / "translated+transliterated"
    repair_model: str | None = None
    after: dict = field(default_factory=dict)
    texts: list[str] = field(default_factory=list)  # the sentences after any repair
    needs_rewrite: bool = False  # wrong language: the answer model is asked to rewrite it
    note: str | None = None  # e.g. why a wrong-language answer is shown as written
    devanagari: list[str] | None = None  # the "Show in Devanagari" view, for romanized answers
    error: str | None = None

    def as_log(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("texts", "devanagari")} | {
            "has_devanagari_view": self.devanagari is not None}


def _identify(text: str) -> dict:
    script = script_of(text)
    if not MODELS_ON:
        return {"lang": "en" if script == "latin" else None, "script": script, "romanized": False}
    from ..ml.langid import get_langid

    r = get_langid().identify(text)
    return {"lang": r.lang if r.lang != "other" else None, "script": script, "romanized": r.romanized,
            "confidence": r.confidence}


def check_and_repair(sentences: list[str], analysis: QueryAnalysis | None) -> Fidelity:
    if analysis is None or analysis.lang not in ("en", "hi", "mr") or not sentences:
        return Fidelity({}, {}, None, None, texts=sentences)
    want_lang = analysis.lang
    want_script = "latin" if (want_lang == "en" or analysis.romanized) else "devanagari"
    expected = {"lang": want_lang, "script": want_script, "romanized": analysis.romanized}
    found = _identify(" ".join(sentences))
    got_script = "latin" if found["script"] == "latin" else "devanagari" if found["script"] in (
        "devanagari", "mixed") else found["script"]
    lang_match = found["lang"] == want_lang if found["lang"] else None
    script_match = got_script == want_script
    fid = Fidelity(expected, found, lang_match, script_match, texts=list(sentences))
    if not MODELS_ON:
        return fid
    try:
        from ..ml.xlit import to_devanagari, to_roman

        texts = list(sentences)
        if lang_match is False:
            fid.needs_rewrite = True  # the service asks the answer model (apply_rewrite)
            return fid
        if not script_match and want_lang in ("hi", "mr"):
            if want_script == "latin":
                texts = [to_roman(t, want_lang) for t in texts]
            else:
                texts = [to_devanagari(t, want_lang) for t in texts]
            fid.repair, fid.repair_model = "transliterated", "ai4bharat/IndicXlit"
        if fid.repair:
            fid.texts = texts
            fid.after = _identify(" ".join(texts))
        if analysis.romanized and want_lang in ("hi", "mr"):
            fid.devanagari = [to_devanagari(t, want_lang) for t in fid.texts]
    except Exception as exc:  # a repair that fails leaves the answer as the model wrote it
        log.warning("Fidelity repair failed: %s", exc)
        fid.error = f"{type(exc).__name__}: {str(exc)[:200]}"
    return fid


LANG_NAME = {"en": "English", "hi": "Hindi", "mr": "Marathi"}


def language_instruction(analysis: QueryAnalysis | None) -> str | None:
    """The answer language, stated with every question to the answer model."""
    if analysis is None or analysis.lang not in LANG_NAME:
        return None
    if analysis.lang == "en":
        return "Answer in English."
    name = LANG_NAME[analysis.lang]
    if analysis.romanized:
        return (f"Answer in {name} written in Latin letters (romanized, like the question), not in Devanagari "
                "and not in English.")
    return f"Answer in {name}, in Devanagari script."


def _number_keys(text: str) -> list[str]:
    from .verify import _NUMBER, _key

    return sorted(k for k in (_key(m.group(0).strip(".,:/-")) for m in _NUMBER.finditer(text))
                  if len(k.replace(".", "")) >= 2)


def apply_rewrite(fid: Fidelity, rewritten: tuple[list[str], str] | None, analysis: QueryAnalysis) -> Fidelity:
    """Accept the answer model's rewrite of a wrong-language answer if it is one sentence for one
    and every sentence keeps its numbers; otherwise keep the answer as written, with a note."""
    found = LANG_NAME.get(fid.found.get("lang") or "", "another language")
    if not rewritten or len(rewritten[0]) != len(fid.texts):
        fid.note = f"The answer model wrote in {found}; it is shown as written."
        return fid
    texts, model = rewritten
    if any(_number_keys(a) != _number_keys(b) for a, b in zip(fid.texts, texts)):
        fid.note = (f"The answer model wrote in {found}; its rewrite changed a figure, so the answer is "
                    "shown as written.")
        return fid
    try:
        from ..ml.xlit import to_devanagari, to_roman

        want = fid.expected.get("lang")
        if fid.expected.get("script") == "latin" and want in ("hi", "mr"):
            texts = [to_roman(t, want) if script_of(t) in ("devanagari", "mixed") else t for t in texts]
            if analysis.romanized:
                fid.devanagari = [to_devanagari(t, want) for t in texts]
        fid.texts, fid.repair, fid.repair_model = texts, "rewritten", model
        fid.after = _identify(" ".join(texts))
    except Exception as exc:
        log.warning("Rewrite repair failed: %s", exc)
        fid.error = f"{type(exc).__name__}: {str(exc)[:200]}"
    return fid
