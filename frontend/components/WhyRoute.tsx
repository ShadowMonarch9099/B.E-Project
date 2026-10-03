"use client";

import { LANG_NAME, percent } from "@/lib/format";
import type { Routing } from "@/lib/types";

const ROUTE_NAME: Record<string, string> = { direct: "Direct", translate: "Translate", hybrid: "Hybrid" };
const VIEW_NAME: Record<string, string> = {
  as_typed: "As typed",
  devanagari: "Devanagari (IndicXlit)",
  english: "English (translation)",
};
const QTYPE: Record<string, string> = {
  numeric: "a number or amount",
  date: "a date",
  person: "a person",
  place: "a place",
  procedure: "a procedure",
  list: "a list",
  reason: "a reason",
  other: "other",
};

function lang(code: unknown): string {
  return typeof code === "string" ? (LANG_NAME[code] ?? code) : "—";
}

// "Why this route" (plan section 6.5): the router's rule in words, the question's views,
// the features it looked at, and how well each route's top passages scored.
export default function WhyRoute({ routing }: { routing: Routing }) {
  const f = routing.features;
  const best = Math.max(0, ...Object.values(routing.probes).map((p) => p.best ?? 0));
  const mix = f.doc_lang_mix
    ? Object.entries(f.doc_lang_mix)
        .filter(([k]) => k !== "unknown")
        .map(([k, v]) => `${lang(k)} ${percent(v)}`)
        .join(" · ")
    : "—";
  return (
    <details className="why">
      <summary>
        Why this route: <strong>{ROUTE_NAME[routing.route] ?? routing.route}</strong>
      </summary>
      <p className="why__reason">{routing.reason}</p>
      <dl className="why__views">
        {Object.entries(routing.views).map(([k, v]) => (
          <div key={k}>
            <dt>{VIEW_NAME[k] ?? k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <div className="why__grid">
        <dl className="why__features">
          <div>
            <dt>Question</dt>
            <dd>
              {lang(f.lang)}
              {f.romanized ? ", romanized" : ""}
              {f.code_mix ? `, ${percent(f.code_mix)} English words` : ""}
            </dd>
          </div>
          <div>
            <dt>Asks for</dt>
            <dd>{QTYPE[String(f.question_type)] ?? String(f.question_type ?? "—")}</dd>
          </div>
          <div>
            <dt>Numbers / IDs</dt>
            <dd>{routing.numbers.length ? routing.numbers.join(", ") : "none"}</dd>
          </div>
          <div>
            <dt>Documents</dt>
            <dd>{mix}</dd>
          </div>
          <div>
            <dt>Scanned pages</dt>
            <dd>
              {typeof f.scanned_share === "number" ? percent(f.scanned_share) : "—"}
              {typeof f.top_dense_ocr_conf === "number" && ` · top hits OCR ${percent(f.top_dense_ocr_conf)}`}
            </dd>
          </div>
          <div>
            <dt>English index</dt>
            <dd>{f.shadow_ready ? "ready" : "not built"}</dd>
          </div>
        </dl>
        <ol className="why__probes" aria-label="Best reranker score of each route's top passages">
          {Object.entries(routing.probes).map(([name, p]) => (
            <li key={name} className={name === routing.route ? "is-chosen" : undefined}>
              <span className="why__route">{ROUTE_NAME[name] ?? name}</span>
              <span className="bar bar--value" aria-hidden>
                <span style={{ "--w": percent(best ? (p.best ?? 0) / Math.max(best, 0.01) : 0) } as React.CSSProperties} />
              </span>
              <span className="why__score">
                {(p.best ?? 0).toFixed(2)}
                {p.page ? ` · p. ${p.page}` : ""}
              </span>
            </li>
          ))}
        </ol>
      </div>
      <p className="why__note">
        Scores: the reranker&apos;s best score among each route&apos;s top three passages. The router is a set of
        rules over cheap features; its decisions are logged to train a learned router later.
      </p>
    </details>
  );
}
