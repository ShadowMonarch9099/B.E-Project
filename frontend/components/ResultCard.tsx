import { formatBytes, INGESTION_LABEL } from "@/lib/format";
import type { DetectionResult } from "@/lib/types";
import DetectionFacts from "./DetectionFacts";

// A file the server analysed but did not store (not a PDF, DOCX or scan).
export default function ResultCard({ result: r, onDismiss }: { result: DetectionResult; onDismiss: () => void }) {
  return (
    <article className="result">
      <div className="result__top">
        <div className="stamp stamp--type" aria-hidden>
          {r.type.stamp}
        </div>
        <div className="result__title">
          <p className="filename" title={r.filename}>
            {r.filename}
          </p>
          <p className="kind">
            {r.type.kind}
            <span className="kind__desc"> · {r.type.description}</span>
          </p>
          <p className="meta">
            {formatBytes(r.size_bytes)} · {r.type.mime_type} · not stored
          </p>
        </div>
        <span className={`pill pill--${r.ingestion.status}`}>{INGESTION_LABEL[r.ingestion.status]}</span>
      </div>
      <DetectionFacts result={r} />
      <div className="card-actions">
        <button type="button" className="btn-text" onClick={onDismiss}>
          Dismiss
        </button>
      </div>
    </article>
  );
}
