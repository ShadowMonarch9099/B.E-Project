"""Phase 2: page-anchored chunks (plan section 6.2 steps 3, 5, 6, 7)."""

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.db import Chunk, Page, session_scope
from app.ingest.chunking import MAX_TOKENS, chunk_page, sentence_spans
from app.ingest.normalize import normalize, script_of
from app.ingest.pages import build_text
from app.ingest.tokens import estimate
from app.main import app
from app.ocr import engine_status
from tests.helpers import scanned_pdf


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def test_abbreviations_do_not_end_a_sentence():
    assert sentences("The Collector fixed the rate for Survey No. 142 in the order. It is final.") == [
        "The Collector fixed the rate for Survey No. 142 in the order.", "It is final."]
    assert sentences("सर्व्हे क्र. १४२ मधील जमीन संपादित केली. दि. ०४.०४.२०२३ रोजी आदेश निघाला.") == [
        "सर्व्हे क्र. १४२ मधील जमीन संपादित केली.", "दि. ०४.०४.२०२३ रोजी आदेश निघाला."]
    assert sentences("Dr. R. K. Rao signed it. Was it approved? Yes!") == [
        "Dr. R. K. Rao signed it.", "Was it approved?", "Yes!"]
    assert sentences("योजना लागू आहे। अर्ज करा॥ पुढे") == ["योजना लागू आहे।", "अर्ज करा॥", "पुढे"]


def test_normalization_keeps_meaning_and_maps_digits():
    assert normalize("सर्व्हे  क्र. १४२\n\n  मधील") == "सर्व्हे क्र. 142\nमधील"
    assert normalize("Survey No. १४२", digits=False) == "Survey No. १४२"
    assert script_of("महाराष्ट्र शासन") == "devanagari" and script_of("Government") == "latin"
    assert script_of("ATKT ke liye कितने") == "mixed" and script_of("१४२ ...") is None


def layout_from(paragraphs: list[str], box: bool = True) -> tuple[str, list]:
    """A page layout: one line per paragraph, words laid out left to right."""
    paras = []
    for i, para in enumerate(paragraphs):
        y = 50 + i * 14
        line = [{"t": w, "b": [50 + j * 2, y, 51 + j * 2, y + 10] if box else None, "c": 90}
                for j, w in enumerate(para.split())]
        paras.append([line])
    return build_text(paras)


def test_chunks_respect_size_overlap_and_cover_the_page():
    paragraphs = [" ".join(f"Sentence {p}-{i} talks about clause {i} of the scheme rules." for i in range(6))
                  for p in range(40)]
    text, layout = layout_from(paragraphs)
    chunks = chunk_page(1, text, layout, count=estimate)
    assert len(chunks) > 3
    for c in chunks:
        assert c.text_raw == text[c.char_start:c.char_end]
        assert c.token_count <= MAX_TOKENS
        assert c.bbox and c.ocr_conf == 0.9
    # neighbours share exactly one sentence, and together the chunks cover the page
    for a, b in zip(chunks, chunks[1:]):
        last = sentences(a.text_raw)[-1]
        assert b.text_raw.startswith(last) and sentences(b.text_raw)[0] == last
    assert chunks[0].char_start == 0 and chunks[-1].char_end == len(text)
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))


def test_a_very_long_sentence_is_split_between_words():
    text, layout = layout_from([" ".join(["शब्द"] * 900)])
    chunks = chunk_page(1, text, layout, count=estimate)
    assert len(chunks) >= 3 and all(c.token_count <= MAX_TOKENS for c in chunks)


def test_short_page_is_one_chunk():
    text, layout = layout_from(["Only one short paragraph here."])
    (c,) = chunk_page(4, text, layout, count=estimate)
    assert c.page_no == 4 and c.para_idx == 0 and c.text_raw == text


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def two_page_pdf_with_a_paragraph_across_the_break() -> bytes:
    body = " ".join(f"Clause {i}: the beneficiary must submit the income certificate within thirty days." for i in range(70))
    doc = pymupdf.open()
    rest = body
    for _ in range(2):  # whatever does not fit on page 1 flows onto page 2
        page = doc.new_page()
        words = rest.split()
        lo, hi = 0, len(words)
        while lo < hi:  # the longest prefix that fits on the page
            mid = (lo + hi + 1) // 2
            probe = pymupdf.open()
            if probe.new_page().insert_textbox(pymupdf.Rect(50, 50, 545, 800), " ".join(words[:mid]), fontsize=11) >= 0:
                lo = mid
            else:
                hi = mid - 1
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), " ".join(words[:lo]), fontsize=11)
        rest = " ".join(words[lo:])
    return doc.tobytes()


def test_no_chunk_spans_two_pages_and_boxes_lie_on_their_page(client):
    pdf = two_page_pdf_with_a_paragraph_across_the_break()
    r = client.post("/api/documents", files={"file": ("flowing.pdf", pdf, "application/pdf")})
    doc_id = r.json()["document"]["id"]
    with session_scope() as s:
        pages = {p.page_no: p for p in s.query(Page).filter_by(doc_id=doc_id)}
        chunks = s.query(Chunk).filter_by(doc_id=doc_id).order_by(Chunk.seq).all()
        assert set(pages) == {1, 2} and {c.page_no for c in chunks} == {1, 2}
        for c in chunks:
            page = pages[c.page_no]
            assert c.page_id == page.id
            assert page.text_raw[c.char_start:c.char_end] == c.text_raw  # wholly inside its page
            assert c.bbox
            for x0, y0, x1, y1 in c.bbox:
                assert 0 <= x0 < x1 <= page.width and 0 <= y0 < y1 <= page.height
            assert c.source == "native" and c.ocr_conf is None and c.script == "latin"
            assert c.text_norm == normalize(c.text_raw)
        # the sentence cut by the page break is two chunks, one on each page
        last_p1 = [c for c in chunks if c.page_no == 1][-1].text_raw
        first_p2 = [c for c in chunks if c.page_no == 2][0].text_raw
        assert not last_p1.rstrip().endswith(".")  # page 1 ends mid-sentence ...
        assert first_p2 == pages[2].text_raw[:len(first_p2)]  # ... and page 2 starts with the rest
        assert [c.seq for c in chunks] == list(range(len(chunks)))
    assert client.get(f"/api/documents/{doc_id}").json()["chunks"] == len(chunks)


@pytest.mark.skipif(not engine_status()["available"], reason="Tesseract with eng/hin/mar is not installed")
def test_ocr_chunks_carry_confidence_and_boxes_in_points(client):
    r = client.post("/api/documents", files={"file": ("scan-chunks.pdf", scanned_pdf(), "application/pdf")})
    doc_id = r.json()["document"]["id"]
    with session_scope() as s:
        page = s.query(Page).filter_by(doc_id=doc_id).one()
        chunks = s.query(Chunk).filter_by(doc_id=doc_id).all()
        assert chunks and all(c.source == "ocr" and 0.5 < c.ocr_conf <= 1 for c in chunks)
        for c in chunks:
            for x0, y0, x1, y1 in c.bbox:
                assert 0 <= x0 < x1 <= page.width + 1 and 0 <= y0 < y1 <= page.height + 1
        assert any("142" in c.text_norm for c in chunks)  # १४२ in the scan, ASCII in the index


def test_a_scrap_at_the_end_of_a_page_joins_the_chunk_before_it():
    body = [" ".join(f"Rule {p}-{i} says the applicant must attach proof of income." for i in range(8))
            for p in range(5)]
    text, layout = layout_from(body + ["Page 3"])
    chunks = chunk_page(3, text, layout, count=estimate)
    assert chunks[-1].text_raw.endswith("Page 3") and chunks[-1].token_count > 40


def test_spaces_drawn_in_another_font_still_separate_words():
    """A DOCX converted by LibreOffice (Docker image): Devanagari runs in Lohit, the spaces between
    them in DejaVu, which PyMuPDF puts on lines of their own (the text came out as one word)."""
    from pathlib import Path

    import pymupdf

    from app.ingest.pages import native_page

    path = Path(__file__).parent / "fixtures" / "libreoffice_marathi.pdf"
    with pymupdf.open(path) as doc:
        assert "वसतिगृहपरिपत्रक" in doc[0].get_text()  # the defect
        page = native_page(doc[0], 1)
    lines = [" ".join(w["t"] for w in line) for para in page.paragraphs for line in para]
    assert lines[0] == "वसतिगृह परिपत्रक"
    assert "विद्यार्थी बारा हजार रुपये आहे." in lines


def test_dotted_abbreviations_do_not_end_a_sentence():
    # An answer was cut at "C.T.S." and its first half dropped as uncited (2026-10-04).
    from app.ingest.chunking import sentence_spans

    text = "The acquisition is for 1327.74 sq m of C.T.S. No. 20806 at Shahapur. Notices go to the owner."
    assert [text[s:e] for s, e in sentence_spans(text)] == [
        "The acquisition is for 1327.74 sq m of C.T.S. No. 20806 at Shahapur.", "Notices go to the owner."]
    mr = "मौजे शहापूर सि.स.नं. २०८०६ पै. ही जमीन आरक्षित आहे. पुढील सूचना."
    assert len(sentence_spans(mr)) == 2
    assert len(sentence_spans("The rate is 1.5. The next rule applies.")) == 2  # numbers still end sentences
