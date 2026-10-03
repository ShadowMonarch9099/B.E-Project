import type { DocSummary, PageInfo } from "./types";

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
}

export function percent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

export const METHOD_LABEL: Record<string, string> = {
  content: "File content (Magika model)",
  signature: "File signature (magic bytes)",
  extension: "Text check + file extension",
  text: "Text check (valid UTF-8)",
  size: "File size (empty)",
};

export const INGESTION_LABEL: Record<string, string> = {
  in_scope: "Ingested in v1",
  out_of_scope: "Detected only",
  blocked: "Blocked",
};

export const TEXT_LAYER_LABEL: Record<string, string> = {
  native: "Native: text layer on every page",
  scanned: "Scanned: no text layer",
  mixed: "Mixed: some pages need OCR",
  legacy: "Legacy font: the text is not Unicode, so pages need OCR",
  broken: "Broken text: conjuncts are stored as stray characters, so pages need OCR",
};

export const KIND_LABEL: Record<string, string> = {
  native: "Native PDF",
  scanned: "Scanned PDF",
  mixed: "PDF, partly scanned",
  legacy: "PDF in a legacy Devanagari font",
  broken: "PDF with a broken Devanagari text layer",
  docx: "Word document",
  image: "Scanned image",
};

export const CHECK_REASON_LABEL: Record<string, string> = {
  low_confidence: "OCR confidence is low: check the original page",
  no_text: "The page has marks but no readable text; the scan may be too faint",
  ocr_unavailable: "OCR was not available when this page was read",
};

const STAGE_LABEL: Record<string, string> = {
  preparing: "Preparing the file",
  reading: "Reading the text layer",
  ocr: "Reading scanned pages with OCR (Tesseract mar+hin+eng)",
  chunking: "Splitting pages into passages",
  langid: "Identifying the language of each passage",
  translating: "Translating passages to English",
  embedding: "Embedding passages (multilingual-E5)",
  indexing: "Updating the search index",
};

export function isPending(doc: DocSummary): boolean {
  return doc.status === "queued" || doc.status === "processing";
}

export function statusText(doc: DocSummary): string {
  if (doc.status === "queued") return "Waiting to be indexed…";
  if (doc.status === "failed") return "Indexing failed";
  if (doc.status === "indexed") return "Indexed";
  const { done, total } = doc.progress;
  if (total > 0 && (doc.stage === "reading" || doc.stage === "ocr")) return `Indexing… page ${done} of ${total}`;
  if (total > 0 && doc.stage && doc.stage !== "preparing") return `Indexing… ${done} of ${total}`;
  return "Indexing…";
}

export function stageText(doc: DocSummary): string | null {
  return doc.stage ? (STAGE_LABEL[doc.stage] ?? doc.stage) : null;
}

export function pageSourceLabel(p: Pick<PageInfo, "source" | "ocr_reason">): string {
  if (p.source === "native") return "Text layer";
  if (p.source === "docx") return "Word document";
  if (p.ocr_reason === "legacy_font") return "OCR · legacy font";
  if (p.ocr_reason === "broken_text") return "OCR · broken text layer";
  return "OCR";
}
