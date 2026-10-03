"""Phase 8: the rule-based router, OCR-aware fusion, probes and the "why this route" data."""

import json

import pytest
from fastapi.testclient import TestClient

from app.db import QueryLog, session_scope
from app.main import app
from app.retrieval.hybrid import fuse, ocr_weight, retrieve
from app.retrieval.router import choose, question_type
from tests.helpers import native_pdf

BASE = {"lang": "en", "doc_main_lang": "en", "romanized": False, "code_mix": 0.0, "has_number_or_id": False,
        "top_dense_ocr_conf": None, "shadow_ready": True}


def decide(**changes):
    return choose({**BASE, **changes}, probes={})


def test_rules_in_order():
    assert decide(has_number_or_id=True).rule == "numbers"
    assert decide(romanized=True, lang="mr", doc_main_lang="mr").rule == "romanized"
    assert decide(code_mix=0.4, lang="hi").route == "hybrid"
    assert decide(top_dense_ocr_conf=0.55).rule == "low_ocr"
    assert decide(lang="en", doc_main_lang="mr").route == "translate"  # cross-lingual, shadow ready
    assert decide(lang="en", doc_main_lang="mr", shadow_ready=False).route == "hybrid"
    assert decide(lang="mr", doc_main_lang="mr").route == "direct"
    assert decide().rule == "default"


def test_numbers_win_over_romanized():
    d = decide(has_number_or_id=True, romanized=True)
    assert d.rule == "numbers" and "exactly" in d.reason


def test_a_clearly_better_probe_overrides_the_rule():
    d = choose({**BASE, "lang": "mr", "doc_main_lang": "mr"},
               probes={"direct": {"best": 0.30}, "translate": {"best": 0.90}, "hybrid": {"best": 0.40}})
    assert d.route == "translate" and d.override == "direct" and d.rule == "same_language+probe_gap"
    d = choose({**BASE, "lang": "mr", "doc_main_lang": "mr"},
               probes={"direct": {"best": 0.70}, "translate": {"best": 0.80}, "hybrid": {"best": 0.75}})
    assert d.route == "direct" and d.override is None  # a small gap is not enough


def test_question_types_in_three_languages():
    assert question_type("What is the purpose of the scheme?") == "other"
    assert question_type("What is the income limit?") == "numeric"
    assert question_type("How much grant did the college get?") == "numeric"
    assert question_type("घरकुल योजनेसाठी उत्पन्नाची मर्यादा किती आहे?") == "numeric"
    assert question_type("kab tak apply karna hai?") == "date"
    assert question_type("डॉ. वंजारी यांची बदली कुठे झाली?") == "place"
    assert question_type("ATKT ka niyam kya hai") == "other"  # "ka" is not "why" in Hindi


def test_ocr_aware_weights_shift_to_character_ngrams_on_bad_scans():
    assert ocr_weight("char", 0.6, None) == 0.6 and ocr_weight("char", 0.6, 0.95) == 0.6
    assert ocr_weight("char", 0.6, 0.40) > 0.6 > ocr_weight("word", 0.6, 0.40)
    lists = {"word": [(1, 9.0), (2, 8.0)], "char": [(2, 9.0), (1, 8.0)]}
    weights = {"word": 1.0, "char": 0.6}
    assert fuse(lists, weights)[0].chunk_id == 1  # native pages: words weigh more
    assert fuse(lists, weights, ocr={1: 0.95, 2: 0.35})[0].chunk_id == 2  # chunk 2 is a bad scan


@pytest.fixture(scope="module")
def doc_id():
    pages = ["Circular about the hostel admission process and the documents students must submit. " * 5,
             "The hostel fee for the academic year 2024 is rupees twelve thousand per student. " * 4]
    with TestClient(app) as c:
        r = c.post("/api/documents", files={"file": ("hostel.pdf", native_pdf(pages), "application/pdf")})
        return r.json()["document"]["id"]


def test_the_router_chooses_and_records_its_probes(doc_id):
    r = retrieve("What is the hostel fee for 2024?", [doc_id], route=None)
    d = r.decision
    assert d.route == r.route == "hybrid" and d.rule == "numbers"
    assert set(d.probes) == {"direct", "translate", "hybrid"} and all("best" in p for p in d.probes.values())
    assert d.features["has_number_or_id"] and d.features["doc_main_lang"] == "en"
    assert r.chunks[r.top[0].chunk_id].page_no == 2


def test_the_decision_reaches_the_log_and_the_answer_payload(doc_id):
    with TestClient(app) as c:
        conv = c.post("/api/conversations", json={"doc_id": doc_id}).json()
        text = c.post(f"/api/conversations/{conv['id']}/messages",
                      json={"question": "What is the hostel fee for 2024?"}).text
    events = [(b.split("\n")[0][7:], json.loads(b.split("data: ", 1)[1])) for b in text.strip().split("\n\n")]
    answer = dict(events)["answer"]
    routing = answer["payload"]["routing"]
    assert routing["route"] == "hybrid" and routing["rule"] == "numbers" and routing["reason"]
    assert set(routing["probes"]) == {"direct", "translate", "hybrid"} and routing["features"]["has_number_or_id"]
    with session_scope() as s:
        row = s.get(QueryLog, dict(events)["done"]["query_log_id"])
        assert row.route == "hybrid" and row.route_reason.startswith("[numbers]")
        assert set(row.route_scores) == {"direct", "translate", "hybrid"}
        assert row.features["question_type"] and "scanned_share" in row.features
