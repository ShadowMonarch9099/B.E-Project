import { formatBytes, INGESTION_LABEL, METHOD_LABEL, percent, TEXT_LAYER_LABEL } from "@/lib/format";
import type { DetectionResult } from "@/lib/types";

export default function ResultCard({ result: r }: { result: DetectionResult }) {
  const ext = r.extension_check;
  const confidence = r.detection.confidence;

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
            {formatBytes(r.size_bytes)} · {r.type.mime_type}
          </p>
        </div>
        <span className={`pill pill--${r.ingestion.status}`}>{INGESTION_LABEL[r.ingestion.status]}</span>
      </div>

      <dl className="facts">
        <div>
          <dt>Detected by</dt>
          <dd>{METHOD_LABEL[r.detection.method]}</dd>
        </div>
        {confidence !== null ? (
          <div>
            <dt>Confidence</dt>
            <dd className="confidence">
              <span className="bar bar--value" aria-hidden>
                <span style={{ "--w": percent(confidence) } as React.CSSProperties} />
              </span>
              {percent(confidence)}
            </dd>
          </div>
        ) : (
          <div>
            <dt>Model&apos;s guess</dt>
            <dd>
              {r.detection.model.description} ({percent(r.detection.model.score)}), too unsure to use
            </dd>
          </div>
        )}
        <div>
          <dt>Extension</dt>
          <dd>
            {ext.status === "match" && <>.{r.extension} matches the content</>}
            {ext.status === "mismatch" && (
              <strong className="warn">
                ⚠ Named .{r.extension} ({ext.expected_kind}) but the content is {r.type.kind}
              </strong>
            )}
            {ext.status === "unknown" && <>.{r.extension}</>}
            {ext.status === "none" && <>No extension; identified from content alone</>}
          </dd>
        </div>
        {r.pdf_text_layer && (
          <div>
            <dt>Text layer</dt>
            <dd>{TEXT_LAYER_LABEL[r.pdf_text_layer]}</dd>
          </div>
        )}
        {r.facts.map((f) => (
          <div key={f.label}>
            <dt>{f.label}</dt>
            <dd>{f.value}</dd>
          </div>
        ))}
        {r.scripts && (
          <div className="facts__wide">
            <dt>Script mix</dt>
            <dd>
              <span className="scripts" aria-hidden>
                <span className="scripts__deva" style={{ flexGrow: r.scripts.devanagari }} />
                <span className="scripts__latin" style={{ flexGrow: r.scripts.latin }} />
                <span className="scripts__other" style={{ flexGrow: r.scripts.other }} />
              </span>
              <span className="scripts__legend">
                <span><i className="key key--deva" />Devanagari {percent(r.scripts.devanagari)}</span>
                <span><i className="key key--latin" />Latin {percent(r.scripts.latin)}</span>
                {r.scripts.other > 0 && (
                  <span><i className="key key--other" />Other {percent(r.scripts.other)}</span>
                )}
                {r.scripts.devanagari_digits && <span lang="hi">Devanagari digits (१२३) present</span>}
              </span>
            </dd>
          </div>
        )}
      </dl>

      <p className="route">
        <span className="route__label">Sakshya pipeline</span>
        {r.ingestion.route}
      </p>
      {r.note && <p className="note">{r.note}</p>}
    </article>
  );
}
