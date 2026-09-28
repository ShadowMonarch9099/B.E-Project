import type { DetectionResult, Health, OcrResult } from "./types";

export async function getHealth(): Promise<Health> {
  const res = await fetch("/api/health", { cache: "no-store" });
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

// XMLHttpRequest rather than fetch, because fetch cannot report upload progress.
export function detectFile(
  file: File,
  onProgress: (fraction: number) => void,
): Promise<DetectionResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("files", file, file.name);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new Error("Could not reach the detector. Is the backend running?"));
    xhr.onload = () => {
      let body: { results?: (DetectionResult | { error: string })[]; detail?: string } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        return reject(new Error(`The detector returned an unreadable response (${xhr.status}).`));
      }
      if (xhr.status >= 400) return reject(new Error(body.detail ?? `Upload failed (${xhr.status}).`));
      const first = body.results?.[0];
      if (!first) return reject(new Error("The detector returned no result."));
      if ("error" in first) return reject(new Error(first.error));
      resolve(first);
    };
    xhr.open("POST", "/api/detect");
    xhr.send(form);
  });
}

export async function runOcr(file: File): Promise<OcrResult> {
  const form = new FormData();
  form.append("file", file, file.name);
  let res: Response;
  try {
    res = await fetch("/api/ocr", { method: "POST", body: form });
  } catch {
    throw new Error("Could not reach the OCR service. Is the backend running?");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ?? `OCR failed (${res.status}).`);
  return body as OcrResult;
}
