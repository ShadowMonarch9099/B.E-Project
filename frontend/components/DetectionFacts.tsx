import { METHOD_LABEL, percent, TEXT_LAYER_LABEL } from "@/lib/format";
import type { DetectionResult, ScriptMix } from "@/lib/types";

export function ScriptMixBar({ scripts }: { scripts: ScriptMix }) {
  return (
    <>
      <span className="scripts" aria-hidden>
        <span className="scripts__deva" style={{ flexGrow: scripts.devanagari }} />
        <span className="scripts__latin" style={{ flexGrow: scripts.latin }} />
        <span className="scripts__other" style={{ flexGrow: scripts.other }} />
      </span>
      <span className="scripts__legend">
        <span><i className="key key--deva" />Devanagari {percent(scripts.devanagari)}</span>
        <span><i className="key key--latin" />Latin {percent(scripts.latin)}</span>
        {scripts.other > 0 && <span><i className="key key--other" />Other {percent(scripts.other)}</span>}
        {scripts.devanagari_digits && (
          <span>
            Devanagari digits (<span className="deva-digits" lang="hi">१२३</span>) present
          </span>
        )}
      </span>
    </>
  );
}

// What the detector found out about a file: how it was identified, its facts, and the
// ingestion route (plan section 6.1 steps 1-2).
export default function DetectionFacts({ result: r }: { result: DetectionResult }) {
  const ext = r.extension_check;
  const confidence = r.detection.confidence;
  return (
    <>
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
              <ScriptMixBar scripts={r.scripts} />
            </dd>
          </div>
        )}
      </dl>
      <p className="route">
        <span className="route__label">Sakshya pipeline</span>
        {r.ingestion.route}
      </p>
      {r.note && <p className="note">{r.note}</p>}
    </>
  );
}
