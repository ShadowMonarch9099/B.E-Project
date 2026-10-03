"use client";

import { useEffect, useRef, useState } from "react";
import { getDocument } from "@/lib/api";
import { formatBytes, KIND_LABEL, languageMix, percent } from "@/lib/format";
import type { DetectionResult, DocSummary } from "@/lib/types";
import DetectionFacts, { ScriptMixBar } from "./DetectionFacts";
import PagesPanel from "./PagesPanel";

// Technical details of one document (shown only with "Technical details" on): how it was
// read, its language mix, the English index, the file analysis and every page's text.
export default function DocDetails({ doc, onClose, onReindex }: { doc: DocSummary; onClose: () => void; onReindex: () => void }) {
  const [detection, setDetection] = useState<DetectionResult | null | undefined>(undefined);
  const panelRef = useRef<HTMLDivElement>(null);
  const stats = doc.stats;

  useEffect(() => {
    let live = true;
    getDocument(doc.id)
      .then((d) => live && setDetection(d.detection))
      .catch(() => live && setDetection(null));
    panelRef.current?.focus();
    return () => {
      live = false;
    };
  }, [doc.id]);

  const shadow = stats.shadow;
  return (
    <div className="viewer" onKeyDown={(e) => e.key === "Escape" && onClose()}>
      <div className="viewer__backdrop" onClick={onClose} aria-hidden />
      <div ref={panelRef} className="viewer__panel details" role="dialog" aria-modal="true"
           aria-label={`Details of ${doc.filename}`} tabIndex={-1}>
        <header className="viewer__head">
          <div className="viewer__title">
            <span className="stamp viewer__stamp" aria-hidden>
              {doc.stamp}
            </span>
            <span className="viewer__file" title={doc.filename}>
              {doc.filename}
            </span>
          </div>
          <div className="viewer__actions">
            <button type="button" className="btn-text" onClick={onReindex} title="Read the file again from scratch">
              Re-index
            </button>
            <button type="button" className="viewer__close" onClick={onClose} aria-label="Close">
              ×
            </button>
          </div>
        </header>
        <div className="details__body">
          <dl className="details__facts">
            <div>
              <dt>Type</dt>
              <dd>{doc.kind ? (KIND_LABEL[doc.kind] ?? doc.kind) : doc.label.toUpperCase()}</dd>
            </div>
            <div>
              <dt>Size</dt>
              <dd>{formatBytes(doc.size_bytes)}</dd>
            </div>
            {doc.page_count !== null && (
              <div>
                <dt>Pages</dt>
                <dd>{doc.anchor === "paragraph" ? "paragraph anchors" : doc.page_count}</dd>
              </div>
            )}
            {doc.chunks !== null && (
              <div>
                <dt>Passages</dt>
                <dd>{doc.chunks}</dd>
              </div>
            )}
            {stats.ocr_share !== undefined && (
              <div>
                <dt>Read with OCR</dt>
                <dd>
                  {percent(stats.ocr_share)}
                  {stats.mean_ocr_conf != null && ` · mean confidence ${percent(stats.mean_ocr_conf)}`}
                </dd>
              </div>
            )}
            {languageMix(stats.languages) && (
              <div>
                <dt>Languages (IndicLID)</dt>
                <dd>{languageMix(stats.languages)}</dd>
              </div>
            )}
            {shadow && (
              <div>
                <dt>English index</dt>
                <dd>
                  {shadow.state === "building" && shadow.total
                    ? `building: ${shadow.done ?? 0} of ${shadow.total} passages`
                    : shadow.state}
                </dd>
              </div>
            )}
          </dl>
          {stats.scripts && <ScriptMixBar scripts={stats.scripts} />}
          {doc.notes.map((n) => (
            <p key={n} className="note">
              {n}
            </p>
          ))}
          <h3 className="details__h">File analysis</h3>
          {detection === undefined ? (
            <div className="bar bar--indeterminate" aria-label="Loading">
              <span />
            </div>
          ) : detection ? (
            <DetectionFacts result={detection} />
          ) : (
            <p className="note">No analysis stored.</p>
          )}
          {doc.status === "indexed" && (
            <>
              <h3 className="details__h">Pages and text</h3>
              <PagesPanel docId={doc.id} filename={doc.filename} paragraphAnchors={doc.anchor === "paragraph"} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
