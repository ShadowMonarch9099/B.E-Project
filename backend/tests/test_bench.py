"""Phase 10: Sakshya-Bench tooling: annotation, double annotation, agreement, export, splits."""

import json

import pytest
from fastapi.testclient import TestClient

from app.bench import manifest as man
from app.bench.metrics import cohen_kappa, token_f1
from app.bench.store import needs_second
from app.main import app
from tests.helpers import native_pdf


def test_cohen_kappa_on_a_worked_example():
    # observed agreement 3/4; chance agreement (2*1 + 2*3) / 16 = 0.5; kappa = (0.75 - 0.5) / 0.5
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.5)
    assert cohen_kappa(["x", "y"], ["x", "y"]) == 1.0
    assert cohen_kappa([], []) is None


def test_token_f1_on_spans():
    assert token_f1("income limit is three lakh", "income limit is three lakh") == 1.0
    assert token_f1("the limit is three lakh rupees", "three lakh rupees") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert token_f1("उत्पन्न मर्यादा", "") == 0.0


def manifest_of(n: int) -> dict:
    docs = [{"id": f"doc-{i:02d}", "languages": ["mr" if i % 2 else "en"], "scanned": i % 3 == 0} for i in range(n)]
    return {"documents": docs, "frozen": None}


def test_splits_are_by_document_and_near_50_15_35():
    m = man.assign_splits(manifest_of(60))
    counts = {s: sum(1 for d in m["documents"] if d["split"] == s) for s in man.SPLITS}
    assert sum(counts.values()) == 60
    assert 25 <= counts["train"] <= 35 and 6 <= counts["dev"] <= 12 and 18 <= counts["test"] <= 24
    again = man.assign_splits(manifest_of(60))
    assert [d["split"] for d in again["documents"]] == [d["split"] for d in m["documents"]]  # deterministic


def test_a_frozen_test_split_never_changes():
    m = man.freeze(man.assign_splits(manifest_of(20)))
    test_before = list(m["frozen"]["test"])
    m["documents"].extend({"id": f"late-{i}", "languages": ["hi"], "scanned": False} for i in range(10))
    man.assign_splits(m)
    assert all(d["split"] in ("train", "dev") for d in m["documents"] if d["id"].startswith("late-"))
    man.check_frozen(m)
    assert sorted(d["id"] for d in m["documents"] if d["split"] == "test") == test_before
    with pytest.raises(ValueError):
        man.freeze(m)
    next(d for d in m["documents"] if d["split"] == "test")["split"] = "train"
    with pytest.raises(ValueError):
        man.check_frozen(m)


def test_about_fifteen_percent_get_a_second_annotation():
    share = sum(needs_second(f"q-{i}") for i in range(2000)) / 2000
    assert 0.12 < share < 0.18


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def doc(client):
    pages = ["Introduction to the circular. " * 6,
             "The hostel fee for the academic year is rupees twelve thousand per student. " * 3]
    r = client.post("/api/documents", files={"file": ("hostel_circular.pdf", native_pdf(pages), "application/pdf")})
    doc_id = r.json()["document"]["id"]
    text = client.get(f"/api/documents/{doc_id}/pages").json()["pages"][1]["text"]
    start = text.index("rupees twelve thousand")
    return doc_id, start, start + len("rupees twelve thousand")


def ask(client, doc, annotator, text, slice_="same_language", answerable=True, status="verified"):
    doc_id, s, e = doc
    gold = {"doc_id": doc_id, "page": 2, "span_start": s, "span_end": e} if answerable else None
    return client.post("/api/bench/questions", json={"text": text, "slice": slice_, "answerable": answerable,
                                                     "doc_id": doc_id, "annotator": annotator, "gold": gold,
                                                     "status": status})


def test_a_question_with_its_gold_span_and_boxes(client, doc):
    r = ask(client, doc, "tanmay", "What is the hostel fee per student?")
    assert r.status_code == 201
    q = r.json()
    (a,) = q["annotations"]
    assert a["role"] == "primary" and a["page"] == 2 and a["span_text"] == "rupees twelve thousand"
    assert a["bbox"] and all(len(b) == 4 for b in a["bbox"])
    assert q["needs_second"] == needs_second(q["id"])


def test_rules_for_slices_and_spans(client, doc):
    assert ask(client, doc, "kush", "Who is the warden?", slice_="unanswerable", answerable=True).status_code == 422
    assert ask(client, doc, "kush", "Who is the warden?", slice_="unanswerable", answerable=False).status_code == 201
    doc_id, _, _ = doc
    bad = client.post("/api/bench/questions", json={"text": "Fee?", "slice": "same_language", "answerable": True,
                                                    "doc_id": doc_id, "annotator": "kush",
                                                    "gold": {"doc_id": doc_id, "page": 2, "span_start": 5, "span_end": 99999}})
    assert bad.status_code == 422


def test_second_annotation_is_blind_and_by_someone_else(client, doc, monkeypatch):
    import app.bench.store as store

    monkeypatch.setattr(store, "needs_second", lambda _id: True)
    q = ask(client, doc, "tanmay", "How much is the yearly hostel fee?").json()
    queue = client.get("/api/bench/second", params={"annotator": "kush"}).json()["questions"]
    item = next(x for x in queue if x["id"] == q["id"])
    assert "annotations" not in item  # the first annotation is hidden
    assert all(x["id"] != q["id"] for x in client.get("/api/bench/second", params={"annotator": "tanmay"}).json()["questions"])
    doc_id, s, e = doc
    same = client.post(f"/api/bench/questions/{q['id']}/second",
                       json={"annotator": "tanmay", "gold": {"doc_id": doc_id, "page": 2, "span_start": s, "span_end": e}})
    assert same.status_code == 422
    ok = client.post(f"/api/bench/questions/{q['id']}/second",
                     json={"annotator": "kush", "gold": {"doc_id": doc_id, "page": 2, "span_start": s - 7, "span_end": e}})
    assert ok.status_code == 200
    agreement = client.get("/api/bench/stats").json()["agreement"]
    assert agreement["double_annotated"] >= 1 and agreement["page_agreement"] == 1.0
    assert 0 < agreement["span_token_f1"] < 1


def test_drafts_are_not_exported_until_verified(client, doc):
    q = ask(client, doc, "llm-draft", "Is there a hostel fee?", status="draft").json()
    lines = [json.loads(x) for x in client.get("/api/bench/export").text.splitlines()]
    assert all(r["id"] != q["id"] for r in lines) and lines
    record = lines[0]
    assert {"question", "slice", "answerable", "gold", "split"} <= set(record)
    client.post(f"/api/bench/questions/{q['id']}/verify")
    assert any(json.loads(x)["id"] == q["id"] for x in client.get("/api/bench/export").text.splitlines())
