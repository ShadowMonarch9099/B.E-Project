// Mirrors the JSON returned by the FastAPI backend (backend/app/detection.py).

export type IngestionStatus = "in_scope" | "out_of_scope" | "blocked";

export interface DetectionResult {
  filename: string;
  size_bytes: number;
  extension: string | null;
  type: {
    label: string;
    stamp: string;
    kind: string;
    description: string;
    mime_type: string;
    group: string;
    is_text: boolean;
  };
  detection: {
    method: "content" | "signature" | "extension" | "text" | "size";
    confidence: number | null;
    model: { name: string; label: string; description: string; score: number };
  };
  extension_check: {
    status: "match" | "mismatch" | "unknown" | "none";
    expected_label: string | null;
    expected_kind: string | null;
  };
  facts: { label: string; value: string }[];
  scripts: { devanagari: number; latin: number; other: number; devanagari_digits: boolean } | null;
  pdf_text_layer: "native" | "scanned" | "mixed" | "legacy" | null;
  ingestion: { status: IngestionStatus; route: string; needs_ocr: boolean };
  note: string | null;
}

export interface Health {
  status: string;
  detector: { name: string; version: string; model: string };
  ocr: {
    available: boolean;
    engine: string;
    version: string | null;
    languages: string[];
    missing: string[];
    error: string | null;
  };
  limits: { max_file_mb: number; max_files: number };
}

export interface OcrPage {
  page: number;
  source: "ocr" | "text_layer";
  text: string;
  confidence: number | null;
  check_page: boolean;
  check_reason: "low_confidence" | "no_text" | null;
  ocr_reason: "no_text_layer" | "legacy_font" | null;
  word_count: number;
}

export interface OcrResult {
  filename: string;
  engine: { name: string; version: string; languages: string[] };
  summary: {
    pages_total: number;
    pages_read: number;
    pages_ocr: number;
    truncated: boolean;
    max_pages: number;
    mean_confidence: number | null;
    check_pages: number[];
    legacy_font_pages: number[];
    low_confidence_threshold: number;
  };
  pages: OcrPage[];
}

export type OcrState =
  | { state: "running" }
  | { state: "done"; data: OcrResult }
  | { state: "error"; message: string };

export type Item =
  | { id: string; name: string; size: number; state: "queued" }
  | { id: string; name: string; size: number; state: "uploading"; progress: number }
  | { id: string; name: string; size: number; state: "analysing" }
  | { id: string; name: string; size: number; state: "done"; result: DetectionResult; ocr?: OcrState }
  | { id: string; name: string; size: number; state: "error"; message: string };
