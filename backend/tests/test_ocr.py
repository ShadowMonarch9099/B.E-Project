import io
import os

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ocr import engine_status, run_ocr

FONT_DIR = "/usr/share/fonts/truetype/lohit-devanagari"
FONT_FILE = "Lohit-Devanagari.ttf"

pytestmark = [
    pytest.mark.skipif(not engine_status()["available"], reason="Tesseract with eng/hin/mar is not installed"),
    pytest.mark.skipif(not os.path.exists(os.path.join(FONT_DIR, FONT_FILE)),
                       reason="Lohit Devanagari font needed to render the test page"),
]

HTML = """<div style='font-family: Lohit; font-size: 13pt; line-height: 1.5'>
<p><b>Government of Maharashtra</b><br>Revenue and Forest Department</p>
<p>Survey number 142 is acquired for the district road project.</p>
<p>महाराष्ट्र शासन महसूल व वन विभाग. सर्वे क्रमांक १४२ जिल्हा रस्ता प्रकल्पासाठी संपादित करण्यात येत आहे.</p>
</div>"""


def scan_image(dpi: int = 200) -> Image.Image:
    """Render a trilingual page and rasterise it, like a scanner would."""
    doc = pymupdf.open()
    page = doc.new_page()
    css = f"@font-face {{font-family: Lohit; src: url({FONT_FILE});}}"
    page.insert_htmlbox(pymupdf.Rect(60, 60, 540, 780), HTML, css=css, archive=pymupdf.Archive(FONT_DIR))
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def scanned_pdf(with_text_page: bool = False) -> bytes:
    doc = pymupdf.open()
    if with_text_page:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), "Native text page. " * 20)
    page = doc.new_page()
    page.insert_image(page.rect, stream=png_bytes(scan_image()))
    return doc.tobytes()


def test_scanned_pdf_is_read_in_all_three_languages():
    r = run_ocr(scanned_pdf(), "pdf")
    page = r["pages"][0]
    assert r["summary"]["pages_ocr"] == 1
    assert page["source"] == "ocr"
    assert "Government of Maharashtra" in page["text"]
    assert "142" in page["text"]
    assert "महाराष्ट्र" in page["text"] and "१४२" in page["text"]
    assert page["confidence"] > 60 and page["check_page"] is False
    word = page["words"][0]
    assert set(word) == {"text", "conf", "bbox"} and len(word["bbox"]) == 4


def test_mixed_pdf_ocrs_only_pages_without_text():
    r = run_ocr(scanned_pdf(with_text_page=True), "pdf")
    assert [p["source"] for p in r["pages"]] == ["text_layer", "ocr"]
    assert r["summary"]["pages_ocr"] == 1


def test_image_ocr():
    r = run_ocr(png_bytes(scan_image()), "png")
    assert "Revenue and Forest Department" in r["pages"][0]["text"]


def test_blank_page_is_not_flagged():
    r = run_ocr(png_bytes(Image.new("L", (1200, 1600), 255)), "png")
    page = r["pages"][0]
    assert page["text"] == "" and page["confidence"] is None and page["check_page"] is False


def test_unreadable_page_with_ink_is_flagged():
    noise = Image.effect_noise((900, 1200), 120).point(lambda v: 0 if v < 110 else 255)
    r = run_ocr(png_bytes(noise), "png")
    page = r["pages"][0]
    assert page["check_page"] is True
    assert page["check_reason"] in {"no_text", "low_confidence"}
    assert r["summary"]["check_pages"] == [1]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_ocr_endpoint(client):
    res = client.post("/api/ocr", files={"file": ("scan.pdf", scanned_pdf(), "application/pdf")})
    assert res.status_code == 200
    body = res.json()
    assert body["engine"]["name"] == "Tesseract"
    assert "Government" in body["pages"][0]["text"]


def test_ocr_endpoint_rejects_non_scannable_files(client):
    res = client.post("/api/ocr", files={"file": ("notes.md", b"# Title\n\nSome text.\n", "text/markdown")})
    assert res.status_code == 422


def test_health_reports_ocr(client):
    assert client.get("/api/health").json()["ocr"]["available"] is True
