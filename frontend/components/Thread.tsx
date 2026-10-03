"use client";

import { percent } from "@/lib/format";
import type { AbstainPayload, AnswerPayload, ChatMessage, Citation, ErrorPayload, NearestPage } from "@/lib/types";
import WhyRoute from "./WhyRoute";

export type OpenCitation = (c: Citation) => void;

export interface PendingAnswer {
  question: string;
  stage: "retrieving" | "answering";
  text: string;
  showQuestion: boolean; // false once the server has stored (and returned) the question
}

function chipKey(c: Citation): string {
  return c.anchor === "paragraph" ? `${c.doc_id}:p${c.para_idx}` : `${c.doc_id}:${c.page_no}`;
}

function chipLabel(c: Citation): string {
  return c.label ?? (c.anchor === "paragraph" ? `¶ ${c.para_idx + 1}` : `p. ${c.page_no}`);
}

// Stamp-style chips after a cited sentence; a dashed "check" mark when the page's OCR
// confidence is low (plan N3).
export function CiteChips({ citations, onOpen }: { citations: Citation[]; onOpen: OpenCitation }) {
  const seen = new Set<string>();
  const unique = citations.filter((c) => {
    const k = chipKey(c);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return (
    <>
      {unique.map((c) => (
        <button
          key={chipKey(c)}
          type="button"
          className={`cite-chip${c.check_page ? " cite-chip--check" : ""}`}
          title={`${c.filename} · ${chipLabel(c)}${
            c.check_page ? ` · scanned page, OCR confidence ${c.ocr_conf !== null ? percent(c.ocr_conf) : "low"}: check the page` : ""
          }${c.cited_text ? `\n“${c.cited_text.slice(0, 160)}”` : ""}`}
          onClick={() => onOpen(c)}
        >
          {chipLabel(c)}
          {c.check_page && <span className="cite-chip__check">check</span>}
        </button>
      ))}
    </>
  );
}

function NearestList({ pages, onOpen }: { pages: NearestPage[]; onOpen: OpenCitation }) {
  if (pages.length === 0) return null;
  return (
    <ol className="nearest">
      {pages.map((p) => (
        <li key={`${p.doc_id}:${p.page_no}:${p.id}`}>
          <button type="button" className="nearest__item" onClick={() => onOpen(p)}>
            <span className="nearest__where">
              {p.filename} · {p.anchor === "paragraph" ? `¶ ${p.para_idx + 1}` : `p. ${p.page_no}`}
              {p.check_page && <span className="cite-chip__check">check</span>}
            </span>
            <span className="nearest__snippet">{p.snippet}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}

function Answer({ payload, onOpen }: { payload: AnswerPayload; onOpen: OpenCitation }) {
  return (
    <>
      <p className="answer-text">
        {payload.segments.map((s, i) => (
          <span key={i} className="answer-seg">
            {s.text}
            {s.citations.length > 0 && <CiteChips citations={s.citations} onOpen={onOpen} />}
          </span>
        ))}
      </p>
      {payload.status === "cut_short" && <p className="msg-note">The answer was cut short before it finished.</p>}
      {payload.fallback_used && <p className="msg-note">Answered by a fallback model after the first model declined.</p>}
    </>
  );
}

function AssistantMessage({ m, onOpen, onRetry }: { m: ChatMessage; onOpen: OpenCitation; onRetry?: () => void }) {
  const p = m.payload as AnswerPayload | AbstainPayload | ErrorPayload;
  if (m.status === "abstained" && p.kind === "abstain") {
    return (
      <div className="msg msg--assistant msg--abstain">
        <span className="stamp stamp--abstain" aria-hidden>
          Not found
        </span>
        <p className="answer-text">{m.content}</p>
        <p className="msg-note">
          Sakshya did not call the answer model: {p.reason}.
        </p>
        <NearestList pages={p.nearest} onOpen={onOpen} />
        {p.routing && <WhyRoute routing={p.routing} />}
      </div>
    );
  }
  if (m.status === "error" && p.kind === "error") {
    return (
      <div className="msg msg--assistant msg--error" role="alert">
        <p className="answer-text">{p.message}</p>
        {onRetry && p.code !== "not_configured" && (
          <button type="button" className="btn-text" onClick={onRetry}>
            Ask again
          </button>
        )}
        {p.nearest.length > 0 && (
          <>
            <p className="msg-note">Meanwhile, these pages matched best:</p>
            <NearestList pages={p.nearest} onOpen={onOpen} />
          </>
        )}
        {p.routing && <WhyRoute routing={p.routing} />}
      </div>
    );
  }
  if (m.status === "refused") {
    return (
      <div className="msg msg--assistant msg--error">
        <p className="answer-text">{m.content}</p>
      </div>
    );
  }
  return (
    <div className="msg msg--assistant">
      {p.kind === "answer" ? <Answer payload={p} onOpen={onOpen} /> : <p className="answer-text">{m.content}</p>}
      {p.kind === "answer" && p.routing && <WhyRoute routing={p.routing} />}
    </div>
  );
}

export default function Thread({
  messages,
  pending,
  error,
  onOpen,
  onRetry,
}: {
  messages: ChatMessage[];
  pending: PendingAnswer | null;
  error: string | null;
  onOpen: OpenCitation;
  onRetry: (question: string) => void;
}) {
  return (
    <ol className="thread" aria-live="polite">
      {messages.map((m, i) => {
        const prevUser = messages
          .slice(0, i)
          .reverse()
          .find((x) => x.role === "user");
        return (
          <li key={m.id} className={`thread__item thread__item--${m.role}`}>
            {m.role === "user" ? (
              <div className="msg msg--user">{m.content}</div>
            ) : (
              <AssistantMessage m={m} onOpen={onOpen} onRetry={prevUser ? () => onRetry(prevUser.content) : undefined} />
            )}
          </li>
        );
      })}
      {pending && (
        <>
          {pending.showQuestion && (
            <li className="thread__item thread__item--user">
              <div className="msg msg--user">{pending.question}</div>
            </li>
          )}
          <li className="thread__item thread__item--assistant">
            <div className="msg msg--assistant msg--streaming" aria-busy="true">
              {pending.text ? (
                <p className="answer-text">
                  {pending.text}
                  <span className="caret" aria-hidden />
                </p>
              ) : (
                <p className="typing">
                  <span className="typing__dots" aria-hidden>
                    <i />
                    <i />
                    <i />
                  </span>
                  {pending.stage === "retrieving" ? "Searching the document…" : "Writing a cited answer…"}
                </p>
              )}
            </div>
          </li>
        </>
      )}
      {error && (
        <li className="thread__item thread__item--assistant">
          <div className="msg msg--assistant msg--error" role="alert">
            <p className="answer-text">{error}</p>
          </div>
        </li>
      )}
    </ol>
  );
}
