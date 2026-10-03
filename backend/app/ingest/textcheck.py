"""Does a PDF's Devanagari text layer say what the page shows? (plan section 6.2 step 2)

Many government PDFs carry a Unicode text layer whose Devanagari is wrong although every
character is valid: "शासन निर्णय" stored as "िासन वनणवय", "मुंबई" as "मुांबई". The legacy-font
and stray-character checks (app/legacy_fonts.py) do not see this. So, for a PDF with Devanagari
text, the page with the most Devanagari among the first few is rendered and read with OCR, and
the share of its text-layer words that OCR also reads is measured.

Calibration (2026-10-03, one page per document, words of 3+ Devanagari letters): Wikipedia's
PDF export of a Marathi and a Hindi article (clean Unicode) agreed on 0.92-0.97; every
"native" Maharashtra GR and collectorate notice in Sakshya-Bench agreed on 0.31-0.73, and the
team's two sample GRs on 0.37 and 0.65-0.68. Below AGREEMENT (0.85) the document's Devanagari
pages are read with OCR instead of the text layer.
"""

import os
import re
from dataclasses import asdict, dataclass

from .normalize import normalize

AGREEMENT = float(os.environ.get("SAKSHYA_TEXT_CHECK_AGREEMENT", "0.85"))
SAMPLE_PAGES = 6  # look for the sample page among the first pages
MIN_DEVANAGARI = 200  # characters on the sample page, else there is too little to judge
MIN_WORDS = 40
PAGE_DEVANAGARI = 20  # a page with at least this much Devanagari is OCR'd when the check fails
DPI = 200

_DEVA = re.compile(r"[ऀ-ॿ]")
_STRIP = ".,;:()[]\"'-–।|/"


def enabled() -> bool:
    return os.environ.get("SAKSHYA_TEXT_CHECK", "on") != "off"


def devanagari_count(text: str) -> int:
    return len(_DEVA.findall(text))


def words(text: str) -> list[str]:
    out = []
    for w in normalize(text).split():
        w = w.strip(_STRIP)
        if len(w) >= 3 and _DEVA.search(w) and not re.search(r"[A-Za-z0-9]", w):
            out.append(w)
    return out


def agreement(text_layer: str, ocr_text: str) -> tuple[float, int]:
    """Share of the text layer's Devanagari words that OCR also read, and how many there were."""
    tl, seen = words(text_layer), set(words(ocr_text))
    return (sum(1 for w in tl if w in seen) / len(tl) if tl else 1.0), len(tl)


@dataclass
class TextCheck:
    page: int
    agreement: float
    words: int
    threshold: float
    trusted: bool

    def as_dict(self) -> dict:
        return asdict(self)

    def note(self) -> str:
        return (f"The Devanagari text layer did not match the page: on page {self.page} only "
                f"{self.agreement:.0%} of its words were also read by OCR (clean text layers reach about 92-97%), "
                "so pages with Devanagari text were read with OCR.")


def check_pdf(path, ocr=None) -> TextCheck | None:
    """None when there is too little Devanagari text to judge (or the sample page fails to OCR).
    `ocr(image, page_no) -> text` defaults to Tesseract (app/ocr.py)."""
    import pymupdf
    from PIL import Image

    if ocr is None:
        from ..ocr import ocr_image

        def ocr(image, page_no):
            return ocr_image(image, page_no).text

    with pymupdf.open(path) as doc:
        best, index = 0, None
        for i in range(min(doc.page_count, SAMPLE_PAGES)):
            n = devanagari_count(doc[i].get_text())
            if n > best:
                best, index = n, i
        if index is None or best < MIN_DEVANAGARI:
            return None
        page = doc[index]
        layer = page.get_text()
        pix = page.get_pixmap(dpi=DPI)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    try:
        text = ocr(image, index + 1)
    except Exception:
        return None
    share, n = agreement(layer, text)
    if n < MIN_WORDS:
        return None
    return TextCheck(index + 1, round(share, 3), n, AGREEMENT, share >= AGREEMENT)
