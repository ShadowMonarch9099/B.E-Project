"use client";

import { useEffect, useState } from "react";
import { getDocument } from "@/lib/api";
import { formatBytes, isPending, KIND_LABEL, languageMix, percent, stageText, statusText } from "@/lib/format";
import type { DetectionResult, DocSummary } from "@/lib/types";
import DetectionFacts, { ScriptMixBar } from "./DetectionFacts";
import PagesPanel from "./PagesPanel";

function FileAnalysis({ docId }: { docId: string }) {
  const [detection, setDetection] = useState<DetectionResult | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    getDocument(docId)
      .then((d) => live && setDetection(d.detection))
      .catch(() => live && setDetection(null));
    return () => {
      live = false;
    };
  }, [docId]);
  if (detection === undefined) {
    return (
      <div className="bar bar--indeterminate" aria-label="Loading">
        <span />
      </div>
    );
  }
  return detection ? <DetectionFacts result={detection} /> : <p className="note">No analysis stored.</p>;
}

// A stored document: its ingestion status, what was read from it, and its pages.
export default function DocumentCard({
  doc,
  busy,
  onDelete,
  onReindex,
  onOpen,
}: {
  doc: DocSummary;
  busy: boolean;
  onDelete: () => void;
  onReindex: () => void;
  onOpen?: () => void;
}) {
  const [panel, setPanel] = useState<"none" | "analysis" | "pages">("none");
  const pending = isPending(doc);
  const { done, total } = doc.progress;
  const stats = doc.stats;
  const toggle = (next: "analysis" | "pages") => setPanel((p) => (p === next ? "none" : next));

  return (
    <article className="result" aria-busy={pending}>
      <div className="result__top">
        <div className={`stamp ${pending ? "stamp--ghost" : "stamp--type"}`} aria-hidden>
          {pending ? "···" : doc.stamp}
        </div>
        <div className="result__title">
          <p className="filename" title={doc.filename}>
            {doc.filename}
          </p>
          <p className="kind">
            {doc.kind ? (KIND_LABEL[doc.kind] ?? doc.kind) : doc.label.toUpperCase()}
            {doc.page_count !== null && (
              <span className="kind__desc">
                {" "}
                · {doc.anchor === "paragraph" ? "paragraph anchors" : `${doc.page_count} ${doc.page_count === 1 ? "page" : "pages"}`}
              </span>
            )}
          </p>
          <p className="meta">
            {formatBytes(doc.size_bytes)}
            {doc.status === "indexed" && doc.chunks !== null && doc.chunks > 0 && ` · ${doc.chunks} passages`}
            {stats.ocr_share !== undefined && stats.ocr_share > 0 && ` · ${percent(stats.ocr_share)} OCR`}
            {stats.mean_ocr_conf != null && ` · mean OCR confidence ${percent(stats.mean_ocr_conf)}`}
          </p>
        </div>
        <span className={`pill pill--${doc.status}`}>{statusText(doc)}</span>
      </div>

      {pending && (
        <div className="doc-progress">
          {total > 0 ? (
            <div className="bar bar--progress" aria-label={statusText(doc)}>
              <span style={{ width: percent(done / total) }} />
            </div>
          ) : (
            <div className="bar bar--indeterminate" aria-label={statusText(doc)}>
              <span />
            </div>
          )}
          <p className="meta">{stageText(doc) ?? "Waiting for the previous document to finish"}</p>
        </div>
      )}

      {doc.status === "failed" && doc.error && <p className="error">{doc.error}</p>}
      {doc.notes.map((n) => (
        <p key={n} className="note">
          {n}
        </p>
      ))}

      {doc.status === "indexed" && stats.scripts && (
        <div className="doc-scripts">
          <ScriptMixBar scripts={stats.scripts} />
        </div>
      )}
      {doc.status === "indexed" && languageMix(stats.languages) && (
        <p className="meta doc-langs">Language of the passages (IndicLID): {languageMix(stats.languages)}</p>
      )}
      {doc.status === "indexed" && stats.shadow?.state === "building" && (
        <div className="doc-progress">
          <div className="bar bar--progress" aria-label="Building the English index">
            <span style={{ width: stats.shadow.total ? percent((stats.shadow.done ?? 0) / stats.shadow.total) : "4%" }} />
          </div>
          <p className="meta">
            Building the English index for the translate route
            {stats.shadow.total ? `: ${stats.shadow.done ?? 0} of ${stats.shadow.total} passages translated` : "…"}
          </p>
        </div>
      )}
      {doc.status === "indexed" && stats.shadow?.state === "queued" && (
        <p className="meta">Searchable now. The English index for the translate route is built when no other document is being indexed.</p>
      )}
      {doc.status === "indexed" && stats.shadow?.state === "failed" && (
        <p className="note">The English index could not be built; the translate route falls back to direct search.</p>
      )}

      <div className="card-actions">
        {onOpen && doc.status === "indexed" && (doc.has_viewer || doc.anchor === "paragraph") && (
          <button type="button" className="btn-text" onClick={onOpen}>
            Open
          </button>
        )}
        {doc.status === "indexed" && (
          <button type="button" className="btn-text" aria-expanded={panel === "pages"} onClick={() => toggle("pages")}>
            {panel === "pages" ? "Hide pages" : "Pages & text"}
          </button>
        )}
        <button type="button" className="btn-text" aria-expanded={panel === "analysis"} onClick={() => toggle("analysis")}>
          {panel === "analysis" ? "Hide analysis" : "File analysis"}
        </button>
        {!pending && (
          <button type="button" className="btn-text" onClick={onReindex} disabled={busy}>
            Re-index
          </button>
        )}
        <button type="button" className="btn-text btn-text--danger" onClick={onDelete} disabled={busy}>
          Delete
        </button>
      </div>

      {panel === "analysis" && <FileAnalysis docId={doc.id} />}
      {panel === "pages" && (
        <PagesPanel docId={doc.id} filename={doc.filename} paragraphAnchors={doc.anchor === "paragraph"} />
      )}
    </article>
  );
}
