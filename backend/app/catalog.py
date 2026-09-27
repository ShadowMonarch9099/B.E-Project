"""What each detected content type means for Sakshya.

Maps a content-type label (Magika's vocabulary, e.g. "pdf", "docx", "markdown")
to a user-facing kind, a short stamp label, and the ingestion route from the
Review 2 plan (section 6.1). Version 1 ingests PDF, DOCX and scans only
(plan section 3); every other type is detected and reported, not ingested.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TypeInfo:
    kind: str
    description: str
    mime: str
    group: str


# Types we may assign from a file signature or extension, when the content model
# (Magika) is unsure. Labels follow Magika's naming so both paths agree.
TYPES: dict[str, TypeInfo] = {
    "pdf": TypeInfo("PDF", "PDF document", "application/pdf", "document"),
    "docx": TypeInfo("Word document", "Microsoft Word document (.docx)",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "document"),
    "doc": TypeInfo("Word document (legacy)", "Microsoft Word 97-2003 document", "application/msword", "document"),
    "odt": TypeInfo("OpenDocument text", "OpenDocument text document", "application/vnd.oasis.opendocument.text", "document"),
    "rtf": TypeInfo("Rich Text", "Rich Text Format document", "text/rtf", "text"),
    "xlsx": TypeInfo("Spreadsheet", "Microsoft Excel workbook (.xlsx)",
                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "document"),
    "xls": TypeInfo("Spreadsheet", "Microsoft Excel 97-2003 workbook", "application/vnd.ms-excel", "document"),
    "ods": TypeInfo("Spreadsheet", "OpenDocument spreadsheet", "application/vnd.oasis.opendocument.spreadsheet", "document"),
    "pptx": TypeInfo("Presentation", "Microsoft PowerPoint presentation (.pptx)",
                     "application/vnd.openxmlformats-officedocument.presentationml.presentation", "document"),
    "ppt": TypeInfo("Presentation", "Microsoft PowerPoint 97-2003 presentation", "application/vnd.ms-powerpoint", "document"),
    "odp": TypeInfo("Presentation", "OpenDocument presentation", "application/vnd.oasis.opendocument.presentation", "document"),
    "epub": TypeInfo("E-book", "EPUB e-book", "application/epub+zip", "document"),
    "zip": TypeInfo("Archive", "ZIP archive", "application/zip", "archive"),
    "png": TypeInfo("Image", "PNG image", "image/png", "image"),
    "jpeg": TypeInfo("Image", "JPEG image", "image/jpeg", "image"),
    "gif": TypeInfo("Image", "GIF image", "image/gif", "image"),
    "tiff": TypeInfo("Image", "TIFF image", "image/tiff", "image"),
    "bmp": TypeInfo("Image", "BMP image", "image/bmp", "image"),
    "webp": TypeInfo("Image", "WebP image", "image/webp", "image"),
    "svg": TypeInfo("Image", "SVG vector image", "image/svg+xml", "image"),
    "markdown": TypeInfo("Markdown", "Markdown document", "text/markdown", "text"),
    "rst": TypeInfo("reStructuredText", "reStructuredText document", "text/x-rst", "text"),
    "latex": TypeInfo("LaTeX", "LaTeX document", "text/x-tex", "text"),
    "txt": TypeInfo("Plain text", "Plain text document", "text/plain", "text"),
    "csv": TypeInfo("Spreadsheet", "CSV (comma-separated values)", "text/csv", "text"),
    "tsv": TypeInfo("Spreadsheet", "TSV (tab-separated values)", "text/tab-separated-values", "text"),
    "html": TypeInfo("Web page", "HTML document", "text/html", "text"),
    "xml": TypeInfo("Structured data", "XML document", "text/xml", "text"),
    "json": TypeInfo("Structured data", "JSON document", "application/json", "text"),
    "yaml": TypeInfo("Structured data", "YAML document", "application/x-yaml", "text"),
    "empty": TypeInfo("Empty file", "Empty file", "inode/x-empty", "unknown"),
    "unknown": TypeInfo("Unknown", "Unknown binary data", "application/octet-stream", "unknown"),
}

# Kind for labels Magika may return that are not in TYPES.
KIND_BY_LABEL: dict[str, str] = {
    "xlsb": "Spreadsheet", "mht": "Web page", "eml": "Email", "outlook": "Email",
    "jsonl": "Structured data", "toml": "Structured data", "ini": "Structured data",
    "ico": "Image", "psd": "Image", "jp2": "Image", "tga": "Image", "emf": "Image", "wmf": "Image",
    "dicom": "Medical image", "postscript": "PostScript", "one": "OneNote notebook",
    "srt": "Subtitles", "vtt": "Subtitles", "ipynb": "Notebook",
}

KIND_BY_GROUP: dict[str, str] = {
    "image": "Image", "audio": "Audio", "video": "Video", "archive": "Archive",
    "code": "Source code", "executable": "Executable", "font": "Font",
    "text": "Text", "document": "Document", "application": "Application data",
}

# File extension -> label, for the extension fallback and the mismatch check.
EXT_TO_LABEL: dict[str, str] = {
    "pdf": "pdf", "docx": "docx", "doc": "doc", "odt": "odt", "rtf": "rtf",
    "xlsx": "xlsx", "xls": "xls", "ods": "ods", "pptx": "pptx", "ppt": "ppt", "odp": "odp",
    "epub": "epub", "zip": "zip",
    "png": "png", "jpg": "jpeg", "jpeg": "jpeg", "gif": "gif", "tif": "tiff", "tiff": "tiff",
    "bmp": "bmp", "webp": "webp", "svg": "svg",
    "md": "markdown", "markdown": "markdown", "rst": "rst", "tex": "latex", "txt": "txt",
    "csv": "csv", "tsv": "tsv", "html": "html", "htm": "html", "xml": "xml", "json": "json",
    "yaml": "yaml", "yml": "yaml",
}

STAMP_BY_LABEL: dict[str, str] = {
    "markdown": "MD", "jpeg": "JPG", "latex": "TEX", "txt": "TXT", "empty": "EMPTY",
    "unknown": "?", "outlook": "MSG", "javascript": "JS", "typescript": "TS", "python": "PY",
}

# Labels that plan section 3 lists as v1 ingestion formats.
OCR_IMAGE_LABELS = {"png", "jpeg", "tiff", "bmp", "webp", "gif"}


def type_info(label: str, magika_info=None) -> TypeInfo:
    """Kind, description, MIME type and group for a label."""
    if label in TYPES:
        base = TYPES[label]
        if magika_info is None:
            return base
        return TypeInfo(base.kind, magika_info.description, magika_info.mime_type, magika_info.group)
    if magika_info is not None:
        kind = KIND_BY_LABEL.get(label) or KIND_BY_GROUP.get(magika_info.group, "Other")
        return TypeInfo(kind, magika_info.description, magika_info.mime_type, magika_info.group)
    return TypeInfo(KIND_BY_LABEL.get(label, "Other"), label, "application/octet-stream", "unknown")


def stamp_for(label: str, extensions: list[str] | None = None) -> str:
    """Short text for the type stamp: a known abbreviation, else the usual extension."""
    if label in STAMP_BY_LABEL:
        return STAMP_BY_LABEL[label]
    if extensions:
        return extensions[0].upper()[:5]
    return label.upper()[:5]


def ingestion_route(label: str, pdf_text_layer: str | None, encrypted: bool) -> tuple[str, str]:
    """Return (status, route) for Sakshya v1: status is in_scope, blocked or out_of_scope."""
    if label == "pdf":
        if encrypted:
            return "blocked", "Password-protected PDF: it must be unlocked before it can be read."
        if pdf_text_layer == "native":
            return "in_scope", "Parse the text layer (PyMuPDF), then split into page-anchored chunks."
        if pdf_text_layer == "scanned":
            return "in_scope", "Scanned PDF: OCR every page with Tesseract (hin + mar + eng)."
        if pdf_text_layer == "mixed":
            return "in_scope", "Parse pages that have a text layer; OCR the pages that do not."
        return "in_scope", "Parse the PDF; pages without a text layer go to OCR."
    if label == "docx":
        return "in_scope", "Convert to PDF (LibreOffice) so citations open a real page, then parse."
    if label in OCR_IMAGE_LABELS:
        return "in_scope", "Treat as a scan: OCR with Tesseract (hin + mar + eng)."
    if label in {"doc", "odt", "rtf"}:
        return "out_of_scope", "Not ingested in v1. Save it as DOCX or PDF to include it."
    return "out_of_scope", "Detected only. Version 1 ingests PDF, DOCX and scanned images (plan section 3)."
