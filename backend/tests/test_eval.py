"""Phase 11: the evaluation harness. Metrics on worked examples, the paired bootstrap, synthetic
OCR noise, and an end-to-end run over a small document with hand-written test questions
(test fixtures for the harness, not Sakshya-Bench)."""

import hashlib
import random

import pytest
from fastapi.testclient import TestClient

from app.main import app
from eval import data, metrics, noise, report, runner
from eval.bootstrap import paired_bootstrap
from eval.systems import GROUPS, resolve
from tests.helpers import native_pdf

G = {("d", 3)}


def test_retrieval_metrics_at_page_level():
    ranked = [("d", 1), ("d", 1), ("d", 2), ("d", 3), ("e", 3)]  # two chunks of page 1 count once
    assert metrics.distinct_pages(ranked) == [("d", 1), ("d", 2), ("d", 3), ("e", 3)]
    assert metrics.recall_at(ranked, G, 2) == 0.0 and metrics.recall_at(ranked, G, 5) == 1.0
    assert metrics.mrr_at(ranked, G, 10) == pytest.approx(1 / 3)
    assert metrics.ndcg_at(ranked, G, 10) == pytest.approx(1 / 2)  # 1/log2(4) over ideal 1/log2(2)
    assert metrics.ndcg_at([("d", 3)], G, 10) == 1.0


def test_citations_abstention_and_risk_coverage():
    assert metrics.citation_pr({("d", 3), ("d", 4)}, G) == (0.5, 1.0)
    assert metrics.citation_pr(set(), G) == (None, 0.0)
    ab = metrics.abstention_pr([True, True, False, False], [False, True, False, True])
    assert ab["precision"] == 0.5 and ab["recall"] == 0.5 and ab["wrongly_abstained"] == 1
    # most confident first: right, right, wrong, wrong -> risks 0, 0, 1/3, 1/2
    rc = metrics.risk_coverage([0.9, 0.8, 0.3, 0.1], [False, False, True, True])
    assert rc["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)
    best = metrics.choose_tau([0.9, 0.8, 0.3, 0.1], [False, False, True, True], max_risk=0.0)
    assert best == {"tau": 0.8, "coverage": 0.5, "risk": 0.0}
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3 and metrics.percentile([], 95) is None


def test_paired_bootstrap():
    same = paired_bootstrap([0.5] * 50, [0.5] * 50)
    assert same["diff"] == 0 and not same["real"]
    rng = random.Random(1)
    a = [rng.random() * 0.5 for _ in range(80)]
    better = paired_bootstrap(a, [x + 0.3 for x in a])
    assert better["diff"] == pytest.approx(0.3) and better["real"] and better["low"] > 0


def test_synthetic_ocr_noise_hits_its_level():
    text = ("शासन निर्णय क्रमांक संकीर्ण २०२३ प्र.क्र. १४२ The applicant must attach an income certificate. " * 30)
    assert noise.add_noise(text, 0.0, random.Random(1)) == text
    for level in (0.05, 0.2, 0.3):
        noisy = noise.add_noise(text, level, random.Random(3))
        cer = noise.char_error_rate(text[:600], noisy[:600])
        assert abs(cer - level) < 0.07, (level, cer)
    assert noise.add_noise(text, 0.1, random.Random(5)) == noise.add_noise(text, 0.1, random.Random(5))


def test_system_groups_cover_the_plan():
    names = [s.name for s in resolve("all")]
    assert names[0] == "sakshya"
    assert {"base:dense", "base:translate", "base:hybrid"} <= set(names)  # plan 9.2 (1-3)
    assert {"route:direct", "route:translate", "route:hybrid"} <= set(names)  # plan 9.2 (4)
    assert {"abl:-xlit", "abl:-digits", "abl:-ocr-aware", "abl:-reranker", "abl:-gate"} <= set(names)  # 9.3
    assert set(GROUPS["rq1"]) <= set(names)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


PAGES = ["Hostel rules. The warden closes the gate at ten in the night. " * 4,
         "The hostel fee for the academic year is rupees twelve thousand per student. " * 3,
         "Library hours. The library opens at eight in the morning on working days. " * 4]


def test_end_to_end_retrieval_run(client, monkeypatch, tmp_path):
    pdf = native_pdf(PAGES)
    assert client.post("/api/documents", files={"file": ("eval_circular.pdf", pdf, "application/pdf")}).status_code in (200, 201)
    sha = hashlib.sha256(pdf).hexdigest()  # the app keys documents by the SHA-256 of their bytes
    monkeypatch.setattr(data.man, "split_of_sha", lambda m: {sha: ("eval-doc", "dev")})
    gold = lambda page: {"sha256": sha, "page": page}  # noqa: E731
    questions = [
        {"id": "q1", "question": "What is the hostel fee for the academic year?", "slice": "same_language",
         "lang": "en", "answerable": True, "split": "dev", "sha256": sha, "gold": gold(2), "second": None},
        {"id": "q2", "question": "When does the library open?", "slice": "same_language", "lang": "en",
         "answerable": True, "split": "dev", "sha256": sha, "gold": gold(3), "second": None},
        {"id": "q3", "question": "Who won the cricket world cup in 2011?", "slice": "unanswerable", "lang": "en",
         "answerable": False, "split": "dev", "sha256": sha, "gold": None, "second": None},
    ]
    systems = resolve("rq1,baselines,abl:-reranker,abl:-gate")
    result = runner.run(questions, systems, "document", progress=None)
    rows = {q["id"]: q for q in result["questions"]}
    assert not any(q.get("skipped") for q in rows.values())
    assert rows["q1"]["systems"]["route:hybrid"]["recall@5"] == 1.0
    assert rows["q2"]["systems"]["sakshya"]["mrr@10"] == 1.0
    assert "latency_ms" in rows["q1"]["systems"]["sakshya"]
    assert rows["q3"]["systems"]["abl:-gate"]["answered"] is True  # no gate: always answers
    result["meta"] = {"title": "test", "split": "dev", "command": "python -m eval run --split dev", "commit": "x",
                      "date": "today", "machine": {}, "peak_rss_mb": 1}
    summary = report.write(result, tmp_path / "run.md", resamples=200)
    md = (tmp_path / "run.md").read_text(encoding="utf8")
    assert "_dev split · n = 2 · `python -m eval run --split dev`_" in md
    assert "Oracle" in md and "Baselines" in md and (tmp_path / "run.risk_coverage.svg").exists()
    assert summary["systems"]["sakshya"]["recall@20"]["mean"] == 1.0
    assert summary["best_fixed_route"] in ("route:direct", "route:translate", "route:hybrid")


def test_answer_scoring_from_a_stored_message():
    from eval import answers

    docs = {"abc": {"doc_id": "D", "manifest_id": "m", "doc_type": "native", "split": "dev"}}
    q = {"id": "q", "question": "Fee?", "slice": "same_language", "lang": "en", "answerable": True,
         "gold": {"sha256": "abc", "page": 2}, "second": None}
    msg = {"status": "complete", "content": "The fee is 12,000.",
           "payload": {"kind": "answer", "status": "complete",
                       "sentences": [{"text": "The fee is 12,000.",
                                      "citations": [{"doc_id": "D", "page_no": 2}, {"doc_id": "D", "page_no": 5}]}]}}
    log = {"verification": {"sentences": 2, "kept": 1}, "fidelity": {"lang_match": True, "script_match": True},
           "llm": {"served_by": "openai/gpt-oss-120b via Groq", "usage": {"input_tokens": 3000, "output_tokens": 90}}}
    row = answers.score(q, {"message": msg, "log": log}, docs)
    assert row["answered"] and row["citation_precision"] == 0.5 and row["citation_recall"] == 1.0
    assert row["removed"] == 1
    s = answers.summarize([row])
    assert s["removed_sentence_share"] == 0.5 and s["served_by"] == {"openai/gpt-oss-120b via Groq": 1}
    assert s["tokens"]["per_answer_input"] == 3000
