"""Small documents built in code for the tests."""

import io
from pathlib import Path

import pymupdf
from PIL import Image

FONT_DIR = str(Path(__file__).parent / "fixtures" / "fonts")
FONT_FILE = "Lohit-Devanagari.ttf"
CSS = f"@font-face {{font-family: Lohit; src: url({FONT_FILE});}} * {{font-family: Lohit;}}"

SCAN_HTML = """<div style='font-size: 13pt; line-height: 1.5'>
<p><b>Government of Maharashtra</b><br>Revenue and Forest Department</p>
<p>Survey number 142 is acquired for the district road project.</p>
<p>महाराष्ट्र शासन महसूल व वन विभाग. सर्वे क्रमांक १४२ जिल्हा रस्ता प्रकल्पासाठी संपादित करण्यात येत आहे.</p>
</div>"""


def native_pdf(pages: list[str], fontsize: float = 11) -> bytes:
    """A PDF with one text page per string (Latin text, built-in font)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), text, fontsize=fontsize)
    return doc.tobytes()


def html_pdf(pages: list[str]) -> bytes:
    """A PDF with one page per HTML string, in the bundled Lohit Devanagari font (Unicode text)."""
    doc = pymupdf.open()
    for html in pages:
        page = doc.new_page()
        page.insert_htmlbox(pymupdf.Rect(50, 50, 545, 800), html, css=CSS, archive=pymupdf.Archive(FONT_DIR))
    return doc.tobytes()


def scan_image(html: str = SCAN_HTML, dpi: int = 200) -> Image.Image:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_htmlbox(pymupdf.Rect(60, 60, 540, 780), html, css=CSS, archive=pymupdf.Archive(FONT_DIR))
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def scanned_pdf(html: str = SCAN_HTML) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=png_bytes(scan_image(html)))
    return doc.tobytes()


def docx_bytes(paragraphs: list[str]) -> bytes:
    import docx

    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()
