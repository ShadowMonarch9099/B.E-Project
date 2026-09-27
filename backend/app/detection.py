"""Detect what type of document a file is.

Three layers, strongest first:
1. Content: Google's Magika, a prebuilt deep-learning content-type model
   (about 1 MB, runs on CPU in milliseconds). Used as-is; we train nothing.
2. Signature: magic bytes and ZIP/OLE structure, for files where Magika only
   answers "generic text" or "unknown" (very short or truncated files).
3. Text check: valid UTF-8/UTF-16 with almost no control characters. The
   exact text format (Markdown, CSV, ...) then comes from the extension, e.g.
   a short Markdown file, or a Hindi text file Magika calls "unknown".

The file's name never overrides its content; a disagreement is reported as a
mismatch instead.
"""

import io
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePath

from magika import Magika

from . import catalog
from .inspect_file import Facts, inspect

GENERIC_LABELS = {"txt", "unknown"}
TEXT_LABELS = {label for label, info in catalog.TYPES.items() if info.group == "text"}

# Groups of types an extension check treats as interchangeable.
COMPATIBLE = [
    {"txt", "markdown", "rst", "csv", "tsv", "latex"},
    {"jpeg", "png", "gif", "tiff", "bmp", "webp"},
    {"doc", "xls", "ppt"},  # OLE containers look alike
]

ODF_MIMES = {
    b"application/vnd.oasis.opendocument.text": "odt",
    b"application/vnd.oasis.opendocument.spreadsheet": "ods",
    b"application/vnd.oasis.opendocument.presentation": "odp",
    b"application/epub+zip": "epub",
}


@lru_cache(maxsize=1)
def get_magika() -> Magika:
    return Magika()


@dataclass
class Detection:
    label: str
    method: str  # "content", "signature", "extension", "text" or "size"
    confidence: float | None
    model_label: str
    model_description: str
    model_score: float
    info: catalog.TypeInfo
    is_text: bool
    extensions: list[str] | None = None


def sniff_signature(data: bytes, ext: str) -> str | None:
    """Identify a file from its leading bytes and container structure."""
    head = data[:16]
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head.startswith(b"BM"):
        return "bmp"
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        # Legacy Office file; the container does not say which app, so use the extension.
        return ext if ext in {"doc", "xls", "ppt"} else "doc"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = set(z.namelist())
                if "mimetype" in names:
                    mime = z.read("mimetype").strip()
                    if mime in ODF_MIMES:
                        return ODF_MIMES[mime]
        except zipfile.BadZipFile:
            return None
        if any(n.startswith("word/") for n in names):
            return "docx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
        if any(n.startswith("ppt/") for n in names):
            return "pptx"
        return "zip"
    return None


def looks_like_text(data: bytes, sample: int = 65536) -> bool:
    """True when the bytes are UTF-8 (or UTF-16 with a BOM) with almost no control characters."""
    chunk = data[:sample]
    if chunk[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = chunk.decode("utf-16", errors="replace")
    else:
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.start < len(chunk) - 3:  # not just a character cut off at the sample's end
                return False
            text = chunk[: exc.start].decode("utf-8")
    if not text:
        return False
    controls = sum(1 for c in text if ord(c) < 32 and c not in "\t\n\r\f")
    return controls / len(text) < 0.01


def extension_of(filename: str) -> str:
    return PurePath(filename).suffix.lower().lstrip(".")


def detect(data: bytes, filename: str) -> Detection:
    ext = extension_of(filename)
    if not data:
        info = catalog.type_info("empty")
        return Detection("empty", "size", 1.0, "empty", "Empty file", 1.0, info, False)

    result = get_magika().identify_bytes(data)
    out = result.output
    model_label = str(out.label)

    if model_label not in GENERIC_LABELS:
        info = catalog.type_info(model_label, out)
        return Detection(model_label, "content", float(result.score), model_label,
                         out.description, float(result.score), info, bool(out.is_text),
                         list(out.extensions))

    sig = sniff_signature(data, ext)
    if sig:
        info = catalog.type_info(sig)
        return Detection(sig, "signature", None, model_label, out.description,
                         float(result.score), info, sig in TEXT_LABELS)

    # Magika is unsure on short text and on Devanagari text (it can call a Hindi file
    # "unknown"), so check the bytes directly before giving up.
    if out.is_text or looks_like_text(data):
        ext_label = catalog.EXT_TO_LABEL.get(ext)
        label, method = (ext_label, "extension") if ext_label in TEXT_LABELS else ("txt", "text")
        return Detection(label, method, None, model_label, out.description,
                         float(result.score), catalog.type_info(label), True)

    info = catalog.type_info(model_label, out)
    return Detection(model_label, "content", float(result.score), model_label,
                     out.description, float(result.score), info, bool(out.is_text))


def extension_check(ext: str, label: str) -> tuple[str, str | None]:
    """Compare the name's extension with the detected type: match, mismatch, unknown or none."""
    if not ext:
        return "none", None
    expected = catalog.EXT_TO_LABEL.get(ext)
    if expected is None:
        return "unknown", None
    if expected == label or any(expected in s and label in s for s in COMPATIBLE):
        return "match", expected
    return "mismatch", expected


def analyse(data: bytes, filename: str) -> dict:
    """Full result for one file, shaped for the API."""
    det = detect(data, filename)
    ext = extension_of(filename)
    ext_status, ext_expected = extension_check(ext, det.label)
    facts: Facts = inspect(det.label, det.is_text, det.info.group, data)
    status, route = catalog.ingestion_route(det.label, facts.pdf_text_layer, facts.encrypted)
    return {
        "filename": filename,
        "size_bytes": len(data),
        "extension": ext or None,
        "type": {
            "label": det.label,
            "stamp": catalog.stamp_for(det.label, det.extensions),
            "kind": det.info.kind,
            "description": det.info.description,
            "mime_type": det.info.mime,
            "group": det.info.group,
            "is_text": det.is_text,
        },
        "detection": {
            "method": det.method,
            "confidence": det.confidence,
            "model": {"name": "Magika", "label": det.model_label,
                      "description": det.model_description, "score": det.model_score},
        },
        "extension_check": {
            "status": ext_status,
            "expected_label": ext_expected,
            "expected_kind": catalog.type_info(ext_expected).kind if ext_expected else None,
        },
        "facts": [{"label": k, "value": v} for k, v in facts.items],
        "scripts": facts.scripts,
        "pdf_text_layer": facts.pdf_text_layer,
        "ingestion": {"status": status, "route": route},
        "note": facts.note,
    }
