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
};
