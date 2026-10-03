"use client";

import { useEffect, useId, useRef, useState } from "react";
import type { LlmStatus } from "@/lib/types";

export interface AskableDoc {
  id: string;
  name: string;
}

const MAX_CHARS = 1000;

// The question box, fixed at the bottom of the window, with a picker when several
// documents are indexed. The thread above it scrolls; the bar stays put.
export default function AskBar({
  docs,
  selectedId,
  onSelect,
  onAsk,
  busy,
  llm,
}: {
  docs: AskableDoc[];
  selectedId: string;
  onSelect: (id: string) => void;
  onAsk: (question: string) => Promise<boolean>;
  busy: boolean;
  llm?: LlmStatus;
}) {
  const [question, setQuestion] = useState("");
  const boxRef = useRef<HTMLTextAreaElement>(null);
  const inputId = useId();
  const selectId = useId();

  // Grow the box with its text, up to about four lines.
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = `${Math.min(box.scrollHeight, 132)}px`;
    box.style.overflowY = box.scrollHeight > 132 ? "auto" : "hidden";
  }, [question]);

  const doc = docs.find((d) => d.id === selectedId) ?? docs[0];
  const trimmed = question.trim();

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!trimmed || !doc || busy) return;
    setQuestion("");
    const accepted = await onAsk(trimmed);
    if (!accepted) setQuestion(trimmed);
  };

  return (
    <div className="askbar" role="region" aria-label="Ask about a document">
      <form className="askbar__inner" onSubmit={submit}>
        <div className="askbar__scope">
          {docs.length > 1 ? (
            <>
              <label htmlFor={selectId}>Asking about</label>
              <select id={selectId} value={doc?.id} onChange={(e) => onSelect(e.target.value)} disabled={busy}>
                {docs.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </>
          ) : (
            <>
              <span>Asking about</span>
              <strong className="askbar__doc" title={doc?.name}>
                {doc?.name}
              </strong>
            </>
          )}
        </div>
        <div className="askbar__row">
          <label htmlFor={inputId} className="visually-hidden">
            Your question
          </label>
          <textarea
            ref={boxRef}
            id={inputId}
            className="askbar__input"
            rows={1}
            maxLength={MAX_CHARS}
            value={question}
            placeholder="Ask a question about this document, in English, हिंदी, मराठी or romanized…"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e);
            }}
          />
          <button type="submit" className="askbar__send" disabled={!trimmed || busy} aria-label="Ask">
            {busy ? (
              <span className="askbar__spinner" aria-hidden />
            ) : (
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h13M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </button>
        </div>
        <p className="askbar__hint">
          {answerNotice(llm)}
          {llm && !llm.configured && <strong> No API key is set yet: questions are searched, and answers explain how to add one.</strong>}
        </p>
      </form>
    </div>
  );
}

/** Who writes the answers, as the backend reports it (never a key). */
function answerNotice(llm?: LlmStatus): string {
  if (!llm?.provider) return "Answers are written by the answer model; the passages used for each answer are sent to it.";
  const ready = (llm.chain ?? []).filter((p) => p.configured && p.provider !== llm.provider).map((p) => p.provider);
  const backup = ready.length ? `, with ${ready.join(" and ")} as fallback` : "";
  return `Answers are written by ${llm.model ?? "the answer model"} on ${llm.provider}${backup}; the passages used for each answer are sent to it.`;
}
