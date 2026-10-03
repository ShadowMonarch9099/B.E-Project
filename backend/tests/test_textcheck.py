"""The text-layer check: a Devanagari text layer that disagrees with OCR of its own page is
not trusted, and its Devanagari pages are read with OCR. OCR is replaced by a stub here."""

from app.ingest import pages as pages_mod
from app.ingest import textcheck
from tests.helpers import html_pdf

CLEAN = ("शासन निर्णय : राज्यात सुरू असलेल्या शिवभोजन योजनेच्या अंमलबजावणीसाठी लागू केलेले सर्व शासन निर्णय "
         "व परिपत्रके अधिक्रमित करून खालीलप्रमाणे सर्वसमावेशक नियमावली विहित करण्यात येत आहे. ") * 6
# The same sentence as one real GR's text layer stores it (sample from mh-gr-04).
BROKEN = ("िासन वनणवय : राज्यात सुरु असलेल्या विवभोजन योजनेच्या अांमलबजावणीसाठी लागू केलेले सवव िासन वनणवय "
          "व पवरपत्रके अवधक्रवमत करुन खालीलप्रमाणे सववसमावेिक वनयमावली वववहत करण्यात येत आहे. ") * 6


def test_agreement_separates_clean_and_broken_text():
    assert textcheck.agreement(CLEAN, CLEAN)[0] == 1.0
    share, n = textcheck.agreement(BROKEN, CLEAN)
    assert n > 40 and share < textcheck.AGREEMENT


def test_check_pdf_with_a_stubbed_ocr(tmp_path):
    path = tmp_path / "gr.pdf"
    path.write_bytes(html_pdf([f"<p>{CLEAN}</p>"]))
    layer = textcheck.words(__import__("pymupdf").open(path)[0].get_text())
    assert layer, "the bundled Lohit font should give a Devanagari text layer"
    same = textcheck.check_pdf(path, ocr=lambda image, page: " ".join(layer))
    assert same.trusted and same.page == 1 and same.agreement == 1.0
    other = textcheck.check_pdf(path, ocr=lambda image, page: "पूर्णपणे वेगळा मजकूर " * 40)
    assert not other.trusted and "OCR" in other.note()
    english = tmp_path / "en.pdf"
    english.write_bytes(html_pdf(["<p>" + "An English circular about hostel fees. " * 20 + "</p>"]))
    assert textcheck.check_pdf(english, ocr=lambda image, page: "") is None  # nothing to judge


def test_distrusted_devanagari_pages_go_to_ocr(tmp_path, monkeypatch):
    path = tmp_path / "gr.pdf"
    path.write_bytes(html_pdf([f"<p>{CLEAN}</p>", "<p>" + "Annexure in English only. " * 20 + "</p>"]))
    seen = []

    def fake_ocr(image, page_no, width, height, reason):
        seen.append((page_no, reason))
        return pages_mod.ExtractedPage(page_no, "ocr", width, height, [], ocr_reason=reason)

    monkeypatch.setattr(pages_mod, "_ocr_page", fake_ocr)
    # PyMuPDF's HTML output of Lohit conjuncts already trips the stray-character check; take the
    # classifier out of the way to see the text check alone.
    monkeypatch.setattr(pages_mod, "classify_page", lambda page: ("text", None, None))
    trusted = {p.page_no: p.source for p in pages_mod.extract_pdf(path, True)}
    assert trusted == {1: "native", 2: "native"} and seen == []
    distrusted = {p.page_no: p.source for p in pages_mod.extract_pdf(path, True, distrust_devanagari=True)}
    assert distrusted == {1: "ocr", 2: "native"} and seen == [(1, "text_mismatch")]
