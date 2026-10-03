"""Phase 7: language of each chunk, the English shadow index, and the three routes."""

import json

import pytest
from fastapi.testclient import TestClient

from app.db import Chunk, Embedding, Page, QueryLog, session_scope
from app.ingest import language
from app.main import app
from app.ml import translate
from app.retrieval.hybrid import retrieve
from tests.helpers import native_pdf, scanned_pdf
from app.ocr import engine_status

PAGES = [
    "The district collector reviewed the irrigation canal repairs in Nashik division. " * 5,
    "Every beneficiary family must submit an income certificate issued by the Tahsildar before March. " * 4,
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def english_doc(client):
    r = client.post("/api/documents", files={"file": ("circular_en.pdf", native_pdf(PAGES), "application/pdf")})
    return r.json()["document"]["id"]


def test_chunks_and_pages_get_a_language(client, english_doc):
    d = client.get(f"/api/documents/{english_doc}").json()
    assert d["stats"]["languages"] == {"en": 1.0}
    with session_scope() as s:
        assert {c.lang for c in s.query(Chunk).filter_by(doc_id=english_doc)} == {"en"}
        assert {p.lang for p in s.query(Page).filter_by(doc_id=english_doc)} == {"en"}


def test_english_chunks_are_copied_into_the_shadow_index(client, english_doc):
    d = client.get(f"/api/documents/{english_doc}").json()
    shadow = d["stats"]["shadow"]
    assert shadow["state"] == "ready" and shadow["copied_english"] > 0 and shadow["translated"] == 0
    with session_scope() as s:
        chunks = s.query(Chunk).filter_by(doc_id=english_doc).all()
        assert all(c.text_en == c.text_raw and c.text_en_model == "source" for c in chunks)
        assert s.query(Embedding).filter_by(doc_id=english_doc, view="en").count() == len(chunks)


@pytest.mark.parametrize("route", ["direct", "translate", "hybrid"])
def test_every_route_finds_the_page(english_doc, route):
    r = retrieve("Which certificate must the beneficiary family submit?", [english_doc], route=route)
    assert r.route == route and r.top
    assert r.chunks[r.top[0].chunk_id].page_no == 2
    if route == "translate":
        assert list(r.lists) == ["translate"]  # the English shadow index, not the passages


def test_a_route_can_be_forced_and_is_logged(client, english_doc):
    conv = client.post("/api/conversations", json={"doc_id": english_doc}).json()
    text = client.post(f"/api/conversations/{conv['id']}/messages",
                       json={"question": "Which certificate must the family submit?", "route": "translate"}).text
    done = json.loads([b for b in text.split("\n\n") if b.startswith("event: done")][0].split("data: ", 1)[1])
    with session_scope() as s:
        row = s.get(QueryLog, done["query_log_id"])
        assert row.route == "translate" and row.route_reason.startswith("[forced]")


class FakeTranslator:
    """Stands in for NLLB / IndicTrans2: a fixed dictionary of whole sentences."""

    name = "fake"

    def to_english(self, texts, lang):
        return ["Survey number 142 was acquired for the district road." if "142" in t or "१४२" in t else
                "The scheme's annual income limit is three lakh rupees." for t in texts]

    def label(self, lang):
        return "fake-translator"


@pytest.mark.skipif(not engine_status()["available"], reason="Tesseract with eng/hin/mar is not installed")
def test_marathi_chunks_are_translated_for_the_translate_route(client, monkeypatch):
    monkeypatch.setattr(translate, "get_translator", lambda: FakeTranslator())
    r = client.post("/api/documents", files={"file": ("gr_scan.pdf", scanned_pdf(), "application/pdf")})
    doc_id = r.json()["document"]["id"]
    with session_scope() as s:
        devanagari = [c for c in s.query(Chunk).filter_by(doc_id=doc_id) if c.script in ("devanagari", "mixed")]
        assert devanagari and all(c.text_en and c.text_en_model == "fake-translator" for c in devanagari)
    hits = retrieve("Survey number 142 was acquired for the district road", [doc_id], route="translate")
    assert list(hits.lists) == ["translate"] and hits.top


def test_translate_route_falls_back_to_direct_without_a_shadow_index(client, monkeypatch):
    monkeypatch.setattr(language, "SHADOW_ON", False)
    r = client.post("/api/documents", files={"file": ("no_shadow.pdf", native_pdf(["Unique text about the "
                                             "Panchayat water tanker schedule for summer months. " * 6]),
                                             "application/pdf")})
    doc_id = r.json()["document"]["id"]
    assert client.get(f"/api/documents/{doc_id}").json()["stats"]["shadow"] == {"state": "off"}
    hits = retrieve("water tanker schedule", [doc_id], route="translate")
    assert "dense" in hits.lists and hits.latency_ms.get("translate_fallback") == 1.0
