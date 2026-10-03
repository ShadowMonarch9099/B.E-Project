"use client";

import { useEffect, useState } from "react";
import { getPages } from "@/lib/api";
import { CHECK_REASON_LABEL, pageSourceLabel } from "@/lib/format";
import type { PageText } from "@/lib/types";

function allText(pages: PageText[]): string {
  return pages.map((p) => `--- Page ${p.page_no} ---\n${p.text}`).join("\n\n");
}

// Every stored page's text, with its source (text layer or OCR), OCR confidence and the
// "check the page" badge for low-confidence scans (plan N3). Loaded when first opened.
export default function PagesPanel({
  docId,
  filename,
  paragraphAnchors,
}: {
  docId: string;
  filename: string;
  paragraphAnchors: boolean;
}) {
  const [pages, setPages] = useState<PageText[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let live = true;
    getPages(docId)
      .then((p) => live && setPages(p))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [docId]);

  if (error) return <p className="error">{error}</p>;
  if (!pages) {
    return (
      <div className="bar bar--indeterminate" aria-label="Loading pages">
        <span />
      </div>
    );
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(allText(pages));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const download = () => {
    const url = URL.createObjectURL(new Blob([allText(pages)], { type: "text/plain;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filename.replace(/\.[^.]+$/, "")}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="ocr">
      <div className="ocr__head">
        <p className="ocr__label">Extracted text</p>
        <div className="ocr__actions">
          <button type="button" className="btn-text" onClick={copy}>
            {copied ? "Copied" : "Copy text"}
          </button>
          <button type="button" className="btn-text" onClick={download}>
            Download .txt
          </button>
        </div>
      </div>
      <div className="ocr__pages">
        {pages.map((p, i) => (
          <details key={p.page_no} className="ocr-page" open={i === 0}>
            <summary>
              <span className="ocr-page__num">{paragraphAnchors ? "Paragraphs" : `Page ${p.page_no}`}</span>
              <span className="ocr-page__meta">
                {pageSourceLabel(p)}
                {p.ocr_conf !== null && ` · ${Math.round(p.ocr_conf * 100)}%`}
                {` · ${p.word_count} words`}
              </span>
              {p.check_reason && (
                <span className="check-badge" title={CHECK_REASON_LABEL[p.check_reason]}>
                  Check the page
                </span>
              )}
            </summary>
            {p.text ? (
              <p className="ocr-text">{p.text}</p>
            ) : (
              <p className="note">
                {p.check_reason ? CHECK_REASON_LABEL[p.check_reason] : "No text found on this page."}
              </p>
            )}
          </details>
        ))}
      </div>
    </section>
  );
}
