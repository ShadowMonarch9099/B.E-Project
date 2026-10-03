"""Phase 9: every sentence needs a citation; optional NLI; language and script fidelity."""

import json

import pytest
from fastapi.testclient import TestClient

from app.answer import fidelity as fid_mod
from app.answer.verify import split_sentences, verify
from app.db import QueryLog, session_scope
from app.llm import get_provider
from app.llm.base import AnswerResult, Citation, Segment
from app.main import app
from app.ml import nli as nli_mod
from app.retrieval.query import QueryAnalysis
from tests.helpers import native_pdf


def seg(text, *chunk_ids):
    return Segment(text, [Citation(c, text) for c in chunk_ids])


def test_sentences_take_the_citations_of_the_blocks_they_overlap():
    segments = [seg("Here is what the GR says. "), seg("The limit is three lakh rupees", 7), seg(". "),
                seg("Applicants attach a certificate", 8), seg(", and the Tahsildar issues it.")]
    out = split_sentences(segments)
    assert [s.text for s in out] == ["Here is what the GR says.", "The limit is three lakh rupees.",
                                     "Applicants attach a certificate, and the Tahsildar issues it."]
    assert [s.chunk_ids for s in out] == [[], [7], [8]]


def test_uncited_sentences_are_removed():
    v = verify([seg("Here is the answer. "), seg("The limit is three lakh rupees.", 7), seg(" This is generous.")], {})
    assert not v.abstain and [s.text for s in v.kept] == ["The limit is three lakh rupees."]
    assert {s.reason for s in v.removed} == {"no_citation"} and len(v.removed) == 2
    # a sentence that is only partly cited counts as cited ("In short:" is not a sentence of its own)
    assert [s.text for s in verify([seg("In short: "), seg("the limit is three lakh rupees.", 7)], {}).kept] == [
        "In short: the limit is three lakh rupees."]


def test_an_answer_with_no_supported_sentence_becomes_an_abstention():
    v = verify([seg("The documents provided do not mention RUSA 3.0.")], {})
    assert v.abstain and v.kept == []


def test_too_few_supported_sentences_also_abstain():
    v = verify([seg("A is true. B is true. C is true."), seg(" D is in the GR.", 3)], {})
    assert v.abstain and "1 of 4" in v.reason


def test_nli_removes_sentences_their_passage_does_not_entail(monkeypatch):
    class FakeNli:
        def entailment(self, pairs):
            return [0.9 if "three lakh" in h and "three lakh" in p else 0.1 for p, h in pairs]

    monkeypatch.setattr(nli_mod, "get_nli", lambda: FakeNli())
    passages = {7: "The annual income must not exceed three lakh rupees.", 8: "The office opens at ten."}
    v = verify([seg("The limit is three lakh rupees.", 7), seg(" The scheme ends in 2030.", 8)], passages,
               use_nli=True)
    assert [s.text for s in v.kept] == ["The limit is three lakh rupees."]
    assert v.removed[0].reason == "not_entailed" and v.removed[0].entailment == 0.1


def test_fidelity_reports_a_match_without_repairing():
    a = QueryAnalysis("What is the limit?", "What is the limit?", "latin", "en")
    f = fid_mod.check_and_repair(["The limit is three lakh rupees."], a)
    assert f.script_match and f.repair is None and f.texts == ["The limit is three lakh rupees."]


# ---------- through the API ----------

@pytest.fixture(scope="module")
def doc_id():
    pages = ["The annual income of the beneficiary family must not exceed three lakh rupees. " * 4]
    with TestClient(app) as c:
        return c.post("/api/documents", files={"file": ("limit.pdf", native_pdf(pages), "application/pdf")}
                      ).json()["document"]["id"]


def ask(doc_id, question):
    with TestClient(app) as c:
        conv = c.post("/api/conversations", json={"doc_id": doc_id}).json()
        text = c.post(f"/api/conversations/{conv['id']}/messages", json={"question": question}).text
    return dict((b.split("\n")[0][7:], json.loads(b.split("data: ", 1)[1])) for b in text.strip().split("\n\n"))


def test_the_answer_shows_only_cited_sentences_and_lists_the_rest(doc_id):
    fake = get_provider()
    chunk = None

    def script(request):
        nonlocal chunk
        chunk = request.passages[0].chunk_id

    fake.requests.clear()
    events = ask(doc_id, "What is the annual income limit for the beneficiary family?")  # default fake answer
    chunk = fake.requests[-1].passages[0].chunk_id
    fake.script.append(AnswerResult([seg("Good question! "), seg("The limit is three lakh rupees.", chunk),
                                     seg(" Many families qualify.")], "complete", "end_turn", "fake"))
    events = ask(doc_id, "What is the annual income limit for the beneficiary family?")
    p = events["answer"]["payload"]
    assert [s["text"] for s in p["sentences"]] == ["The limit is three lakh rupees."]
    assert p["sentences"][0]["citations"][0]["page_no"] == 1
    assert [r["text"] for r in p["removed"]] == ["Good question!", "Many families qualify."]
    assert events["answer"]["content"] == "The limit is three lakh rupees."
    with session_scope() as s:
        row = s.get(QueryLog, events["done"]["query_log_id"])
        assert row.verification["kept"] == 1 and len(row.verification["removed"]) == 2


def test_an_unsupported_answer_is_withheld(doc_id):
    get_provider().script.append(AnswerResult([seg("The documents do not say.")], "complete", "end_turn", "fake"))
    events = ask(doc_id, "What is the annual income limit for the beneficiary family?")
    a = events["answer"]
    assert a["status"] == "abstained" and a["payload"]["unsupported"] and a["payload"]["nearest"]
    with session_scope() as s:
        row = s.get(QueryLog, events["done"]["query_log_id"])
        assert row.llm_called and row.abstained and row.verification["abstain"]


# ---------- real models (run when present) ----------

def _models() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        from app.ml.langid import models_present
        from app.ml.translate import NLLB_DIR

        xlit = all(isinstance(try_to_load_from_cache("ai4bharat/IndicXlit", f"indicxlit-{d}-v1.0/transformer/indicxlit.pt"),
                              str) for d in ("en-indic", "indic-en"))
        return models_present() and xlit and (NLLB_DIR / "model.bin").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _models(), reason="IndicLID / IndicXlit / NLLB not present")
def test_real_script_and_language_repairs(monkeypatch):
    monkeypatch.setattr(fid_mod, "MODELS_ON", True)
    rom_q = QueryAnalysis("gharkul yojana sathi utpannachi maryada kiti ahe?", "", "latin", "mr", romanized=True)
    f = fid_mod.check_and_repair(["घरकुल योजनेसाठी वार्षिक उत्पन्नाची मर्यादा तीन लाख रुपये आहे."], rom_q)
    assert f.repair == "transliterated" and f.texts[0][:7] == "gharkul"
    assert f.after["script"] == "latin" and f.devanagari and "घरकुल" in f.devanagari[0]
    mr_q = QueryAnalysis("घरकुल योजनेसाठी उत्पन्नाची मर्यादा किती आहे?", "", "devanagari", "mr")
    g = fid_mod.check_and_repair(["The income limit for the Gharkul scheme is three lakh rupees."], mr_q)
    assert g.lang_match is False and g.repair == "translated" and g.after["lang"] == "mr"
