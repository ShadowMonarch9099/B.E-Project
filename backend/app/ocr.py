"""OCR for scanned PDFs and images (plan section 6.1 step 2).

Tesseract 5 with the hin, mar and eng language packs, used as-is. PDF pages
are rendered at 300 DPI. For every word we keep Tesseract's confidence and
bounding box, and each page gets an OCR confidence equal to its mean word
confidence. Low-confidence pages get a "check the page" flag (plan N3).

Only PDF pages without a usable text layer are OCR'd; pages that already have
one are read from the text layer. A text layer in a legacy (non-Unicode)
Devanagari font is not usable, so those pages are OCR'd too.
"""

import io
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

from PIL import Image, ImageOps

from .inspect_file import classify_page

# Several pages run in parallel, so each Tesseract process uses one thread.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# Language order matters. "hin" first read the digit 1 as a danda on our test page
# (142 -> 42). On 7 scanned pages of real Maharashtra GRs, "mar+hin+eng" had the lowest
# character error rate (19.3%) and the most amounts read exactly (60%); "eng+hin+mar"
# was 19.6% and 56%. Small samples: re-check on Sakshya-Bench.
OCR_LANGS = os.environ.get("SAKSHYA_OCR_LANGS", "mar+hin+eng")
OCR_MAX_PAGES = int(os.environ.get("SAKSHYA_OCR_MAX_PAGES", "10"))
PDF_DPI = 300
LOW_CONFIDENCE = 60.0  # mean word confidence (0-100) below which a page is flagged
INK_SHARE = 0.01  # a page with more dark pixels than this is not blank
PAGE_TIMEOUT_S = 90
MIN_LONG_SIDE_PX = 1500  # smaller images are upscaled; Tesseract reads small text poorly
WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Why a PDF page is OCR'd, by its classify_page() kind.
OCR_REASON = {"none": "no_text_layer", "legacy": "legacy_font", "broken": "broken_text"}


class OcrUnavailable(RuntimeError):
    pass


def _configure() -> None:
    import pytesseract

    cmd = os.environ.get("SAKSHYA_TESSERACT_CMD")
    if not cmd and not shutil.which("tesseract") and os.path.exists(WINDOWS_DEFAULT):
        cmd = WINDOWS_DEFAULT
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


@lru_cache(maxsize=1)
def engine_status() -> dict:
    """Whether Tesseract and the needed language packs are installed."""
    wanted = OCR_LANGS.split("+")
    try:
        import pytesseract

        _configure()
        version = str(pytesseract.get_tesseract_version())
        installed = set(pytesseract.get_languages(config=""))
    except Exception as exc:  # not installed, or not on PATH
        return {"available": False, "engine": "Tesseract", "version": None, "languages": wanted,
                "missing": wanted, "error": f"Tesseract not found ({type(exc).__name__})"}
    missing = [lang for lang in wanted if lang not in installed]
    return {"available": not missing, "engine": "Tesseract", "version": version,
            "languages": wanted, "missing": missing,
            "error": f"Missing language data: {', '.join(missing)}" if missing else None}


@dataclass
class PageResult:
    page: int
    source: str  # "ocr" or "text_layer"
    text: str
    confidence: float | None = None
    words: list[dict] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    has_ink: bool = False
    ocr_reason: str | None = None  # "no_text_layer", "legacy_font" or "broken_text" for OCR'd PDF pages
    # Words with Tesseract's block / paragraph / line numbers, for page-anchored ingestion.
    # Not part of the /api/ocr response.
    layout_words: list[dict] = field(default_factory=list)

    def check_reason(self) -> str | None:
        """Why a reader should check this page against the original, if at all."""
        if self.source != "ocr":
            return None
        if self.confidence is not None and self.confidence < LOW_CONFIDENCE:
            return "low_confidence"
        if not self.words and self.has_ink:
            return "no_text"  # marks on the page, but nothing readable
        return None

    def as_dict(self) -> dict:
        reason = self.check_reason()
        return {
            "page": self.page,
            "source": self.source,
            "text": self.text,
            "confidence": None if self.confidence is None else round(self.confidence, 1),
            "check_page": reason is not None,
            "check_reason": reason,
            "ocr_reason": self.ocr_reason,
            "word_count": len(self.words) if self.source == "ocr" else len(self.text.split()),
            "size": {"width": self.width, "height": self.height} if self.width else None,
            "words": self.words,
        }


def _prepare(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    long_side = max(image.size)
    if long_side < MIN_LONG_SIDE_PX:
        factor = min(3.0, MIN_LONG_SIDE_PX / long_side)
        image = image.resize((round(image.width * factor), round(image.height * factor)), Image.LANCZOS)
    return image


def ocr_image(image: Image.Image, page: int) -> PageResult:
    """OCR one page image; rebuild its lines and paragraphs from Tesseract's word table."""
    import pytesseract

    image = _prepare(image)
    gray = image.convert("L")
    dark = gray.histogram()[:128]
    has_ink = sum(dark) / (gray.width * gray.height) > INK_SHARE
    data = pytesseract.image_to_data(image, lang=OCR_LANGS, output_type=pytesseract.Output.DICT,
                                     timeout=PAGE_TIMEOUT_S)
    words, confs, layout_words = [], [], []
    lines: dict[tuple, list[str]] = {}
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({"text": text, "conf": round(conf, 1), "bbox": [x, y, x + w, y + h]})
        confs.append(conf)
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        layout_words.append({"text": text, "conf": round(conf, 1), "bbox": [x, y, x + w, y + h], "key": key})
        lines.setdefault(key, []).append(text)

    paragraphs: list[str] = []
    last_par = None
    for (block, par, _), line_words in lines.items():
        line = " ".join(line_words)
        if (block, par) != last_par:
            paragraphs.append(line)
            last_par = (block, par)
        else:
            paragraphs[-1] += "\n" + line
    mean = sum(confs) / len(confs) if confs else None  # a blank page has no words to score
    return PageResult(page, "ocr", "\n\n".join(paragraphs), mean, words, image.width, image.height, has_ink,
                      layout_words=layout_words)


def _pdf_pages(data: bytes) -> tuple[list[tuple[int, object, str | None]], int]:
    """(page number, rendered image or text-layer text, OCR reason) for the first OCR_MAX_PAGES pages."""
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    items = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("The PDF is password-protected.")
        total = doc.page_count
        for i in range(min(total, OCR_MAX_PAGES)):
            page = doc[i]
            kind, text, _ = classify_page(page)
            if kind == "text":
                items.append((i + 1, text.strip(), None))
            else:
                pix = page.get_pixmap(dpi=PDF_DPI, colorspace=pymupdf.csGRAY)
                reason = OCR_REASON[kind]
                items.append((i + 1, Image.open(io.BytesIO(pix.tobytes("png"))), reason))
    return items, total


def _image_pages(data: bytes) -> tuple[list[tuple[int, object, str | None]], int]:
    items = []
    with Image.open(io.BytesIO(data)) as im:
        total = getattr(im, "n_frames", 1)
        for i in range(min(total, OCR_MAX_PAGES)):
            im.seek(i)
            items.append((i + 1, im.copy(), None))
    return items, total


def run_ocr(data: bytes, label: str) -> dict:
    """OCR a scanned PDF or an image. `label` is the detected type (e.g. "pdf", "png")."""
    status = engine_status()
    if not status["available"]:
        raise OcrUnavailable(status["error"])

    items, total = _pdf_pages(data) if label == "pdf" else _image_pages(data)
    to_ocr = [(n, img) for n, img, _ in items if isinstance(img, Image.Image)]
    workers = max(1, min(4, os.cpu_count() or 1, len(to_ocr)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ocr_results = {r.page: r for r in pool.map(lambda item: ocr_image(item[1], item[0]), to_ocr)}
    for n, _, reason in items:
        if n in ocr_results:
            ocr_results[n].ocr_reason = reason

    pages = [ocr_results[n] if n in ocr_results else PageResult(n, "text_layer", content)
             for n, content, _ in items]
    ocr_pages = [p for p in pages if p.source == "ocr"]
    confs = [p.confidence for p in ocr_pages if p.words]
    return {
        "engine": {"name": status["engine"], "version": status["version"], "languages": status["languages"]},
        "summary": {
            "pages_total": total,
            "pages_read": len(pages),
            "pages_ocr": len(ocr_pages),
            "truncated": total > len(pages),
            "max_pages": OCR_MAX_PAGES,
            "mean_confidence": round(sum(confs) / len(confs), 1) if confs else None,
            "check_pages": [p.page for p in ocr_pages if p.check_reason()],
            "legacy_font_pages": [p.page for p in ocr_pages if p.ocr_reason == "legacy_font"],
            "broken_text_pages": [p.page for p in ocr_pages if p.ocr_reason == "broken_text"],
            "low_confidence_threshold": LOW_CONFIDENCE,
        },
        "pages": [p.as_dict() for p in pages],
    }
