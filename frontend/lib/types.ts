// Mirrors the JSON returned by the FastAPI backend (backend/app).

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
  scripts: ScriptMix | null;
  pdf_text_layer: "native" | "scanned" | "mixed" | "legacy" | "broken" | null;
  ingestion: { status: IngestionStatus; route: string; needs_ocr: boolean };
  note: string | null;
}

export interface ScriptMix {
  devanagari: number;
  latin: number;
  other: number;
  devanagari_digits: boolean;
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
  database: { available: boolean; dialect: string; error: string | null };
  docx_converter: { name: string; available: boolean };
  llm?: LlmStatus;
  limits: { max_file_mb: number; max_files: number };
}

export type DocStatus = "queued" | "processing" | "indexed" | "failed";

export interface DocStats {
  pages?: number;
  pages_by_source?: Record<string, number>;
  ocr_share?: number;
  mean_ocr_conf?: number | null;
  check_pages?: number;
  words?: number;
  scripts?: ScriptMix | null;
  languages?: Record<string, number>;
  shadow?: {
    state: "pending" | "building" | "ready" | "failed" | "off";
    done?: number;
    total?: number | null;
    translated?: number;
    copied_english?: number;
    model?: string | string[] | null;
    error?: string;
  };
}

export interface DocSummary {
  id: string;
  filename: string;
  size_bytes: number;
  label: string;
  stamp: string;
  kind: string | null;
  status: DocStatus;
  stage: string | null;
  progress: { done: number; total: number };
  page_count: number | null;
  anchor: "page" | "paragraph";
  has_viewer: boolean;
  error: string | null;
  notes: string[];
  stats: DocStats;
  chunks: number | null;
  created_at: string | null;
  indexed_at: string | null;
}

export interface PageInfo {
  page_no: number;
  source: "native" | "ocr" | "docx";
  ocr_conf: number | null;
  ocr_reason: "no_text_layer" | "legacy_font" | "broken_text" | null;
  check_reason: "low_confidence" | "no_text" | "ocr_unavailable" | null;
  word_count: number;
  width?: number | null;
  height?: number | null;
  lang: string | null;
  script: string | null;
}

export interface PageText extends PageInfo {
  text: string;
}

export interface DocDetail extends DocSummary {
  detection: DetectionResult | null;
  pages: PageInfo[];
}

export type UploadResult =
  | { kind: "stored"; duplicate: boolean; document: DocSummary; detection: DetectionResult | null }
  | { kind: "rejected"; detail: string; detection: DetectionResult };

// A file on its way to the server, or one the server looked at but did not store.
export type UploadItem =
  | { id: string; name: string; size: number; state: "queued" }
  | { id: string; name: string; size: number; state: "uploading"; progress: number }
  | { id: string; name: string; size: number; state: "analysing" }
  | { id: string; name: string; size: number; state: "rejected"; detail: string; detection: DetectionResult }
  | { id: string; name: string; size: number; state: "error"; message: string };

// ---------- conversations (backend/app/api/conversations.py, app/answer/service.py) ----------

export interface Citation {
  id: number; // chunk id
  doc_id: string;
  filename: string;
  page_no: number;
  para_idx: number;
  chunk_idx: number;
  anchor: "page" | "paragraph";
  lang: string | null;
  script: string | null;
  source: "native" | "ocr" | "docx";
  ocr_conf: number | null;
  check_page: boolean;
  bbox: number[][];
  page_width: number | null;
  page_height: number | null;
  cited_text?: string;
  label?: string;
}

export interface NearestPage extends Citation {
  score: number;
  snippet: string;
}

export interface Gate {
  passed: boolean;
  score: number;
  tau: number;
  reason: string;
}

export interface Routing {
  route: string;
  rule: string;
  reason: string;
  override: string | null;
  features: Record<string, unknown> & {
    doc_lang_mix?: Record<string, number>;
    scanned_share?: number;
    top_dense_ocr_conf?: number | null;
    code_mix?: number;
  };
  probes: Record<string, { best: number | null; page: number | null }>;
  views: Record<string, string>;
  numbers: string[];
}

export interface AnswerPayload {
  routing?: Routing | null;
  kind: "answer";
  segments: { text: string; citations: Citation[] }[];
  status: "complete" | "cut_short" | "refused";
  model: string;
  fallback_used: boolean;
  refusal_category: string | null;
  route: string;
  gate: Gate;
  query_log_id?: number;
}

export interface AbstainPayload {
  routing?: Routing | null;
  kind: "abstain";
  reason: string;
  gate: Gate;
  nearest: NearestPage[];
  route: string;
  query_log_id?: number;
}

export interface ErrorPayload {
  routing?: Routing | null;
  kind: "error";
  code: string;
  message: string;
  nearest: NearestPage[];
  query_log_id?: number;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  status: "complete" | "abstained" | "cut_short" | "refused" | "error";
  payload: AnswerPayload | AbstainPayload | ErrorPayload | Record<string, unknown>;
  created_at: string | null;
}

export interface Conversation {
  id: string;
  doc_id: string | null;
  title: string | null;
  created_at: string | null;
  messages?: ChatMessage[];
}

export interface LlmStatus {
  provider: string;
  configured: boolean;
  model: string;
  effort: string | null;
}
