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
  pdf_text_layer: "native" | "scanned" | "mixed" | null;
  ingestion: { status: IngestionStatus; route: string };
  note: string | null;
}

export interface Health {
  status: string;
  detector: { name: string; version: string; model: string };
  limits: { max_file_mb: number; max_files: number };
}

export type Item =
  | { id: string; name: string; size: number; state: "uploading"; progress: number }
  | { id: string; name: string; size: number; state: "analysing" }
  | { id: string; name: string; size: number; state: "done"; result: DetectionResult }
  | { id: string; name: string; size: number; state: "error"; message: string };
