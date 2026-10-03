"""Script-faithful answering (plan section 6.7 step 4, N5).

The answer's language and script are compared with the question's (IndicLID on both):
- English question -> English answer;
- Devanagari Hindi/Marathi question -> an answer in the same language, in Devanagari;
- romanized Hindi/Marathi question -> an answer in the same language, in Latin letters,
  with a "Show in Devanagari" view (IndicXlit, roman -> Devanagari).

Mismatches are repaired sentence by sentence, so every sentence keeps its citations:
- right language, wrong script: IndicXlit (Devanagari -> roman, or roman -> Devanagari);
- wrong language: translation with NLLB-200 (IndicTrans2's model here translates only into
  English), then transliteration if the question was romanized.
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
            from ..ml.translate import get_nllb

            nllb = get_nllb()
            source = found["lang"]
            if source in ("hi", "mr") and got_script == "latin":  # romanized in the wrong language
                texts = [to_devanagari(t, source) for t in texts]
            texts = nllb.translate(texts, source, want_lang)
            fid.repair, fid.repair_model = "translated", nllb.label(want_lang)
            if want_script == "latin" and want_lang != "en":
                texts = [to_roman(t, want_lang) for t in texts]
                fid.repair = "translated+transliterated"
        elif not script_match and want_lang in ("hi", "mr"):
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
