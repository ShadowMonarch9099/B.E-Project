"""Phase 3: lexical (BM25 words + character n-grams) and dense (E5 in pgvector) indexes."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import Chunk, get_engine, is_postgres, session_scope
from app.index import dense
from app.index.lexical import char_ngrams, get_lexical_index, word_tokens
from app.main import app
from app.ml.embedder import E5Embedder, get_embedder
from tests.helpers import native_pdf

PAGES = [
    "Chapter one describes the irrigation canal maintenance budget for the Nashik division. " * 6,
    "The beneficiary family must have an annual income below three lakh rupees to qualify for the "
    "Gharkul housing scheme. Applicants attach the income certificate issued by the Tahsildar. " * 3,
    "Survey number 142 in village Shirdi was acquired for the district road widening project. " * 6,
]
TARGET = "The beneficiary family must have an annual income below three lakh rupees to qualify for the Gharkul housing scheme."


@pytest.fixture(scope="module")
def doc_ids():
    with TestClient(app) as client:
        a = client.post("/api/documents", files={"file": ("gharkul_gr.pdf", native_pdf(PAGES), "application/pdf")})
        b = client.post("/api/documents", files={"file": ("other.pdf", native_pdf(
            ["Minutes of the school committee meeting about the annual sports day and prizes. " * 8]),
            "application/pdf")})
        yield a.json()["document"]["id"], b.json()["document"]["id"]


def page_of(chunk_id: int) -> int:
    with session_scope() as s:
        return s.get(Chunk, chunk_id).page_no


def test_tokens_and_char_ngrams():
    assert word_tokens("Survey No. 142, घरकुल योजना!") == ["survey", "no", "142", "घरकुल", "योजना"]
    grams = char_ngrams(["घर"])
    assert grams == ["#घर#"] or "#घर" in grams
    assert "#gha" in char_ngrams(["gharkul"]) and "kul#" in char_ngrams(["gharkul"])


def test_own_sentence_is_the_top_hit_in_both_lexical_indexes(doc_ids):
    gharkul, _ = doc_ids
    hits = get_lexical_index().search(TARGET.lower(), [gharkul], k=5)
    for name in ("word", "char"):
        assert hits[name], name
        assert page_of(hits[name][0].chunk_id) == 2, name


def test_own_sentence_is_the_top_hit_in_the_dense_index(doc_ids):
    gharkul, _ = doc_ids
    qvec = get_embedder().embed_queries([TARGET])[0]
    (top, score), *_ = dense.search(qvec, [gharkul], k=5)
    assert page_of(top) == 2 and score > 0.5


def test_search_stays_inside_the_chosen_documents(doc_ids):
    gharkul, other = doc_ids
    hits = get_lexical_index().search(TARGET, [other], k=5)
    with session_scope() as s:
        assert all(s.get(Chunk, h.chunk_id).doc_id == other for h in hits["word"] + hits["char"])
    qvec = get_embedder().embed_queries([TARGET])[0]
    with session_scope() as s:
        assert all(s.get(Chunk, cid).doc_id == other for cid, _ in dense.search(qvec, [other], k=5))


def test_search_endpoint_returns_all_three_indexes(doc_ids):
    gharkul, _ = doc_ids
    with TestClient(app) as client:
        r = client.get("/api/search", params={"q": "income certificate Tahsildar", "doc_id": gharkul, "k": 3})
    body = r.json()["hits"]
    assert set(body) == {"word", "char", "dense"}
    assert body["word"][0]["page_no"] == 2 and body["word"][0]["filename"] == "gharkul_gr.pdf"


@pytest.mark.skipif(not is_postgres(), reason="HNSW indexes exist on PostgreSQL only")
def test_hnsw_index_exists_on_postgres(doc_ids):
    name = dense.index_name(get_embedder().name, "passage")
    with get_engine().connect() as conn:
        defn = conn.execute(text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"), {"n": name}).scalar()
    assert defn and "hnsw" in defn and "vector_cosine_ops" in defn


def _e5_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("intfloat/multilingual-e5-base", "model.safetensors"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _e5_cached(), reason="multilingual-e5-base is not in the Hugging Face cache")
def test_real_e5_ranks_the_own_sentence_and_crosses_languages():
    e5 = E5Embedder("intfloat/multilingual-e5-base")
    passages = [
        "लाभार्थी कुटुंबाचे वार्षिक उत्पन्न ₹3,00,000 पेक्षा जास्त नसावे.",  # income limit (Marathi)
        "ग्रामसभेत घरकुल लाभार्थी यादी मंजूर करण्यात यावी.",  # beneficiary list approval (Marathi)
        "शाळेच्या क्रीडा दिनाचे बक्षीस वितरण सोमवारी होईल.",  # school sports day (Marathi)
        "Survey number 142 was acquired for the district road project.",
    ]
    p = e5.embed_passages(passages)
    own = e5.embed_queries([passages[3]])[0]
    assert int((p @ own).argmax()) == 3
    cross = e5.embed_queries(["What is the annual income limit for beneficiaries?"])[0]
    assert int((p @ cross).argmax()) == 0  # English question, Marathi answer
