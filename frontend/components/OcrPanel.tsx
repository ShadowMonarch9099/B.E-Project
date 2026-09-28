"use client";

import { useState } from "react";
import type { OcrResult, OcrState } from "@/lib/types";

function allText(data: OcrResult): string {
  return data.pages.map((p) => `--- Page ${p.page} ---\n${p.text}`).join("\n\n");
}

export default function OcrPanel({
  ocr,
  filename,
  onRetry,
}: {
  ocr: OcrState;
  filename: string;
  onRetry: () => void;
}) {
  const [copied, setCopied] = useState(false);

  if (ocr.state === "running") {
    return (
      <section className="ocr" aria-busy="true">
        <p className="ocr__label">Extracted text</p>
        <div className="bar bar--indeterminate" aria-label="Running OCR">
          <span />
        </div>
        <p className="meta">Running OCR with Tesseract (English, Hindi, Marathi). A page takes a few seconds.</p>
      </section>
    );
  }

  if (ocr.state === "error") {
    return (
      <section className="ocr">
        <p className="ocr__label">Extracted text</p>
        <p className="error">{ocr.message}</p>
        <button type="button" className="btn-text" onClick={onRetry}>
          Try OCR again
        </button>
      </section>
    );
  }

  const { data } = ocr;
  const s = data.summary;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(allText(data));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const download = () => {
    const url = URL.createObjectURL(new Blob([allText(data)], { type: "text/plain;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filename.replace(/\.[^.]+$/, "")}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="ocr">
      <div className="ocr__head">
        <p className="ocr__label">
          Extracted text
          <span className="ocr__engine">
            {data.engine.name} {data.engine.version} · {data.engine.languages.join("+")}
          </span>
        </p>
        <div className="ocr__actions">
          <button type="button" className="btn-text" onClick={copy}>
            {copied ? "Copied" : "Copy text"}
          </button>
          <button type="button" className="btn-text" onClick={download}>
            Download .txt
          </button>
        </div>
      </div>

      <p className="ocr__summary">
        {s.pages_ocr} of {s.pages_read} {s.pages_read === 1 ? "page" : "pages"} read by OCR
        {s.mean_confidence !== null && <> · mean word confidence {Math.round(s.mean_confidence)}%</>}
        {s.check_pages.length > 0 && (
          <>
            {" "}
            · <strong>check {s.check_pages.length === 1 ? "page" : "pages"} {s.check_pages.join(", ")}</strong>
          </>
        )}
      </p>
      {s.legacy_font_pages.length > 0 && (
        <p className="note">
          {s.legacy_font_pages.length === 1 ? "Page" : "Pages"} {s.legacy_font_pages.join(", ")} had a text
          layer in a legacy (non-Unicode) Devanagari font, so {s.legacy_font_pages.length === 1 ? "it was" : "they were"}{" "}
          read with OCR instead.
        </p>
      )}
      {s.truncated && (
        <p className="note">
          Only the first {s.max_pages} of {s.pages_total} pages were read.
        </p>
      )}

      <div className="ocr__pages">
        {data.pages.map((p, i) => (
          <details key={p.page} className="ocr-page" open={i === 0}>
            <summary>
              <span className="ocr-page__num">Page {p.page}</span>
              <span className="ocr-page__meta">
                {p.source === "text_layer" ? "Text layer" : p.ocr_reason === "legacy_font" ? "OCR · legacy font" : "OCR"}
                {p.confidence !== null && ` · ${Math.round(p.confidence)}%`}
                {` · ${p.word_count} words`}
              </span>
              {p.check_page && (
                <span
                  className="check-badge"
                  title={
                    p.check_reason === "no_text"
                      ? "The page has marks but no readable text"
                      : `Mean word confidence below ${s.low_confidence_threshold}%`
                  }
                >
                  Check the page
                </span>
              )}
            </summary>
            {p.text ? (
              <p className="ocr-text">{p.text}</p>
            ) : (
              <p className="note">
                {p.check_reason === "no_text"
                  ? "No readable text. The scan may be too blurry or low-resolution; check the original page."
                  : "No text found on this page."}
              </p>
            )}
          </details>
        ))}
      </div>
    </section>
  );
}
