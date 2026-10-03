"""Phase 6 (part 1): numbers/IDs for exact matching, and IndicXlit transliteration."""

import json

import pytest

from app.retrieval.numbers import _pattern, extract_numbers


def test_numbers_and_ids_are_extracted_with_ascii_digits():
    assert extract_numbers("सर्व्हे क्र. १४२ मधील जमीन, दि. २१.०४.२०१५") == ["142", "21.04.2015"]
    assert extract_numbers("Survey No. 142 under section 4(1), 1 item") == ["142", "4(1)"]


def test_whole_number_matching_handles_indian_grouping():
    assert _pattern("142").search("सर्व्हे क्र. 142 मधील") and not _pattern("142").search("1420 आणि 138")
    assert _pattern("300000").search("उत्पन्न ₹3,00,000 पेक्षा")


def _xlit_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("ai4bharat/IndicXlit",
                                                 "indicxlit-en-indic-v1.0/transformer/indicxlit.pt"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _xlit_cached(), reason="IndicXlit weights are not in the Hugging Face cache")
def test_indicxlit_runs_the_official_weights_without_fairseq():
    from app.ml.xlit import IndicXlit

    x = IndicXlit()
    assert x.transliterate("gharkul", "mr") == "घरकुल"
    assert x.transliterate("yojana", "mr") == "योजना"
    assert x.transliterate("142", "mr") == "142"  # not a word: left as it is


# ---------- views, exact matching and logging through retrieval ----------

from fastapi.testclient import TestClient  # noqa: E402

from app.db import QueryLog, session_scope  # noqa: E402
from app.main import app  # noqa: E402
from app.retrieval import query as query_mod  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402
from tests.helpers import native_pdf  # noqa: E402

PAGES = [
    "Land in village Shirdi, Survey No. 138, was valued by the Collector for the district road. " * 4,
    "The Collector fixed the compensation for land in Survey No. 142 at rupees eighteen lakh forty "
    "thousand per hectare. " * 3,
]


@pytest.fixture(scope="module")
def survey_doc():
    with TestClient(app) as c:
        r = c.post("/api/documents", files={"file": ("acquisition.pdf", native_pdf(PAGES), "application/pdf")})
        yield r.json()["document"]["id"]


def test_devanagari_digits_in_the_question_match_the_number_exactly(survey_doc):
    r = retrieve("सर्व्हे क्र. १४२ ची किंमत?", [survey_doc])
    assert r.analysis.numbers == ["142"]
    exact = r.lists["exact"]
    assert exact and all(r.chunks.get(cid) is None or "142" in r.chunks[cid].text for cid, _ in exact)
    assert r.chunks[r.top[0].chunk_id].page_no == 2  # the 142 page, not the 138 page


def test_views_and_features_are_logged(survey_doc):
    with TestClient(app) as c:
        conv = c.post("/api/conversations", json={"doc_id": survey_doc}).json()
        text = c.post(f"/api/conversations/{conv['id']}/messages",
                      json={"question": "What compensation was fixed for Survey No. 142?"}).text
    log_id = json.loads([b for b in text.split("\n\n") if b.startswith("event: done")][0].split("data: ", 1)[1])["query_log_id"]
    with session_scope() as s:
        row = s.get(QueryLog, log_id)
        assert row.views["as_typed"].startswith("What compensation") and row.views["numbers"] == ["142"]
        assert row.features["has_number_or_id"] is True and "views" in row.features


def test_a_failing_language_model_does_not_stop_the_question(monkeypatch):
    import app.ml.langid as langid_mod

    def broken():
        raise OSError("model file missing")

    monkeypatch.setattr(langid_mod, "get_langid", broken)
    a = query_mod.analyse_query("gharkul yojana kiti ahe?", query_mod.AnalysisOptions(langid=True))
    assert "langid" in a.errors and a.views() == {"as_typed": "gharkul yojana kiti ahe?"}


def _langid_present() -> bool:
    from app.ml.langid import models_present

    return models_present()


@pytest.mark.skipif(not (_langid_present() and _xlit_cached()), reason="IndicLID / IndicXlit models not present")
def test_real_models_build_the_views():
    opts = query_mod.AnalysisOptions(langid=True, transliterate=True, translate=False)
    rom = query_mod.analyse_query("gharkul yojana sathi utpannachi maryada kiti ahe?", opts)
    assert rom.lang == "mr" and rom.romanized and "घरकुल" in rom.devanagari
    mixed = query_mod.analyse_query("third year me ATKT ke liye kitne backlog allowed hai?", opts)
    assert mixed.lang == "hi" and mixed.romanized and mixed.code_mix > 0.2 and "ATKT" in mixed.devanagari
    assert query_mod.analyse_query("घरकुल योजनेसाठी उत्पन्नाची मर्यादा किती आहे?", opts).lang == "mr"
    assert query_mod.analyse_query("टाइफाइड कंजुगेट वैक्सीन किस उम्र में दी जाती है?", opts).lang == "hi"
    eng = query_mod.analyse_query("What is the income limit?", opts)
    assert eng.lang == "en" and not eng.romanized and list(eng.views()) == ["as_typed"]


def _nllb_present() -> bool:
    from app.ml.translate import NLLB_DIR

    return (NLLB_DIR / "model.bin").exists()


@pytest.mark.skipif(not _nllb_present(), reason="NLLB (CTranslate2) is not converted yet")
def test_real_translation_gives_an_english_view(monkeypatch):
    monkeypatch.setenv("SAKSHYA_TRANSLATOR", "nllb")
    from app.ml.translate import NllbTranslator

    out = NllbTranslator().to_english(["घरकुल योजना साठी उत्पन्नाची मर्यादा किती आहे?"], "mr")[0]
    assert "income" in out.lower()


def _indiclid_or_skip():
    from app.ml import langid

    if not langid.models_present():
        pytest.skip("IndicLID models are not downloaded")
    return langid.get_langid()


def test_english_with_the_is_not_romanized_hindi():
    """"the" is also a romanized Hindi word ("were"); it once turned every English passage Hindi."""
    lid = _indiclid_or_skip()
    passage = ("Emphasise the importance of a balanced and nutritious diet during pregnancy. The diet of the woman "
               "should include more food. You should know how to calculate the EDD and communicate this to the woman.")
    assert lid.identify(passage).lang == "en"
    assert lid.identify("What is the area of the land under the scheme?").lang == "en"
    hinglish = lid.identify("third year me ATKT ke liye kitne backlog allowed hai?")
    assert hinglish.lang == "hi" and hinglish.romanized
    assert lid.identify("gharkul yojana sathi utpannachi maryada kiti ahe?").lang == "mr"


def test_transliteration_keeps_english_words_and_spells_function_words():
    from app.retrieval import query as q

    calls = []

    def fake_xlit(word, lang):
        calls.append(word)
        return {"ghar": "घर", "dino": "दिनो"}.get(word, word.upper())

    import app.ml.xlit as xlit

    original = xlit.transliterate_word
    xlit.transliterate_word = fake_xlit
    try:
        out = q.transliterate_query("ghar pe delivery ke liye ATKT 3 dino?", "hi", keep={"delivery"})
    finally:
        xlit.transliterate_word = original
    assert out == "घर पे delivery के लिए ATKT 3 दिनो?"
    assert calls == ["ghar", "dino"]  # function words, English words and acronyms never reach IndicXlit
