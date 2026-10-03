"""Phase 1: stored documents, background ingestion, persistence (plan sections 6.2, 6.8)."""

import time

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.db import Document, Page, init_db, session_scope
from app.ingest import pipeline
from app.main import app
from app.ocr import engine_status
from tests.helpers import docx_bytes, native_pdf, scanned_pdf

ENGLISH = ("Government Resolution on crop insurance. Farmers with land under two hectares are "
           "eligible for the scheme from the next kharif season. ") * 3


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the context manager runs the startup (migrations)
        yield c


def upload(client, name: str, data: bytes, mime: str = "application/pdf"):
    return client.post("/api/documents", files={"file": (name, data, mime)})


def test_health_reports_the_database(client):
    db = client.get("/api/health").json()["database"]
    assert db["available"] is True and db["error"] is None


def test_upload_ingests_every_page_and_keeps_word_boxes(client):
    pdf = native_pdf([ENGLISH + f" Page marker {i}." for i in range(1, 13)])  # more than the OCR API's 10
    r = upload(client, "gr_crop_insurance.pdf", pdf)
    assert r.status_code == 201
    doc_id = r.json()["document"]["id"]

    # TestClient runs background tasks before returning, so ingestion is finished here.
    d = client.get(f"/api/documents/{doc_id}").json()
    assert d["status"] == "indexed" and d["page_count"] == 12
    assert d["progress"] == {"done": 12, "total": 12}
    assert [p["page_no"] for p in d["pages"]] == list(range(1, 13))
    assert {p["source"] for p in d["pages"]} == {"native"}
    assert d["stats"]["pages_by_source"] == {"native": 12}

    with session_scope() as s:
        page = s.query(Page).filter_by(doc_id=doc_id, page_no=12).one()
        assert "Page marker 12." in page.text_raw
        words = [w for para in page.layout for line in para for w in line]
        assert words and all(0 <= w["b"][0] < w["b"][2] <= page.width for w in words)
        assert all(0 <= w["b"][1] < w["b"][3] <= page.height for w in words)
        # each word's offset points at its text in the page text
        assert all(page.text_raw[w["s"]:w["s"] + len(w["t"])] == w["t"] for w in words)


def test_same_bytes_are_recognised_as_a_duplicate(client):
    pdf = native_pdf(["Duplicate check. " * 10])
    first = upload(client, "a.pdf", pdf).json()["document"]["id"]
    again = upload(client, "renamed.pdf", pdf)
    assert again.status_code == 200
    assert again.json()["duplicate"] is True and again.json()["document"]["id"] == first


def test_original_file_is_served_for_the_viewer(client):
    pdf = native_pdf(["Viewer file. " * 10])
    doc_id = upload(client, "viewer.pdf", pdf).json()["document"]["id"]
    r = client.get(f"/api/documents/{doc_id}/file")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content == pdf


def test_out_of_scope_files_are_detected_but_not_stored(client):
    r = upload(client, "notes.txt", b"just some plain text\n" * 5, "text/plain")
    assert r.status_code == 422
    assert r.json()["detection"]["type"]["label"] == "txt"
    names = [d["filename"] for d in client.get("/api/documents").json()["documents"]]
    assert "notes.txt" not in names


def test_docx_without_libreoffice_falls_back_to_paragraph_anchors(client, monkeypatch):
    def no_libreoffice(*_):
        raise RuntimeError("LibreOffice is not installed")

    monkeypatch.setattr(pipeline.docx_convert, "docx_to_pdf", no_libreoffice)
    data = docx_bytes(["First paragraph about the scheme.", "Second paragraph with the income limit."])
    r = upload(client, "circular.docx", data,
               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert r.status_code == 201
    d = client.get(f"/api/documents/{r.json()['document']['id']}").json()
    assert d["status"] == "indexed" and d["anchor"] == "paragraph" and d["has_viewer"] is False
    assert any("paragraphs, not pages" in n for n in d["notes"])
    pages = client.get(f"/api/documents/{d['id']}/pages").json()["pages"]
    assert pages[0]["source"] == "docx" and "income limit" in pages[0]["text"]


@pytest.mark.skipif(not engine_status()["available"], reason="Tesseract with eng/hin/mar is not installed")
def test_scanned_pdf_is_ocrd_with_boxes_in_pdf_points(client):
    r = upload(client, "scan.pdf", scanned_pdf())
    d = client.get(f"/api/documents/{r.json()['document']['id']}").json()
    assert d["status"] == "indexed" and d["kind"] == "scanned"
    page = d["pages"][0]
    assert page["source"] == "ocr" and page["ocr_reason"] == "no_text_layer"
    assert 0.6 < page["ocr_conf"] <= 1
    with session_scope() as s:
        row = s.query(Page).filter_by(doc_id=d["id"]).one()
        assert "महाराष्ट्र" in row.text_raw and "१४२" in row.text_raw
        words = [w for para in row.layout for line in para for w in line]
        assert all(w["c"] is not None for w in words)
        assert all(0 <= w["b"][0] < w["b"][2] <= row.width + 1 for w in words)  # points, not pixels


def test_documents_and_pages_survive_a_restart(client):
    doc_id = upload(client, "persist.pdf", native_pdf(["Persistence check. " * 10])).json()["document"]["id"]
    assert init_db()  # what a fresh start does: connect and migrate (a no-op now)
    with TestClient(app) as fresh:
        d = fresh.get(f"/api/documents/{doc_id}").json()
        assert d["status"] == "indexed" and len(d["pages"]) == 1


def test_unfinished_ingestion_is_resumed_at_startup(client):
    doc_id = upload(client, "resume.pdf", native_pdf(["Resume check. " * 10])).json()["document"]["id"]
    with session_scope() as s:  # pretend the server stopped half-way
        s.get(Document, doc_id).status = "processing"
        s.query(Page).filter_by(doc_id=doc_id).delete()
    pipeline.resume_pending()
    for _ in range(100):
        if client.get(f"/api/documents/{doc_id}").json()["status"] == "indexed":
            break
        time.sleep(0.1)
    d = client.get(f"/api/documents/{doc_id}").json()
    assert d["status"] == "indexed" and len(d["pages"]) == 1


def test_delete_removes_rows_and_files(client):
    doc_id = upload(client, "delete-me.pdf", native_pdf(["Delete check. " * 10])).json()["document"]["id"]
    folder = storage.doc_dir(doc_id)
    assert folder.exists()
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert not folder.exists()
    with session_scope() as s:
        assert s.query(Page).filter_by(doc_id=doc_id).count() == 0


@pytest.mark.skipif(not engine_status()["available"], reason="Tesseract with eng/hin/mar is not installed")
def test_a_scanned_image_gets_a_viewer_pdf_and_boxes_on_its_page(client):
    import pymupdf

    from app.db import Chunk
    from tests.helpers import png_bytes, scan_image

    r = upload(client, "notice.png", png_bytes(scan_image()), "image/png")
    d = client.get(f"/api/documents/{r.json()['document']['id']}").json()
    assert d["status"] == "indexed" and d["kind"] == "image" and d["has_viewer"] is True
    pdf = client.get(f"/api/documents/{d['id']}/file")
    assert pdf.headers["content-type"] == "application/pdf"
    with pymupdf.open(stream=pdf.content, filetype="pdf") as doc:
        rect = doc[0].rect
    with session_scope() as s:
        chunks = s.query(Chunk).filter_by(doc_id=d["id"]).all()
        boxes = [b for c in chunks for b in c.bbox]
    assert boxes and all(0 <= x0 < x1 <= rect.width + 1 and 0 <= y0 < y1 <= rect.height + 1
                         for x0, y0, x1, y1 in boxes)


def test_control_characters_in_pdf_metadata_are_dropped():
    # Found in a district collectorate scan: producer "Adobe PSL 1.3e for Canon\x00", which
    # PostgreSQL refuses to store in text or JSON.
    from app.inspect_file import Facts

    facts = Facts()
    facts.add("Created with", "Adobe PSL 1.3e for Canon" + chr(0))
    facts.add("Title", chr(0))
    assert facts.items == [("Created with", "Adobe PSL 1.3e for Canon")]
