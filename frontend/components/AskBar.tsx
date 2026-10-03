"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useDetails } from "@/lib/details";
import type { LlmStatus } from "@/lib/types";

const MAX_CHARS = 1000;

// The question box at the bottom of the conversation. The document it asks about is the one
// selected in the sidebar.
export default function AskBar({
  onAsk,
  busy,
  disabled,
  llm,
}: {
  onAsk: (question: string) => Promise<boolean>;
  busy: boolean;
  disabled?: boolean;
  llm?: LlmStatus;
}) {
  const { on: details } = useDetails();
  const [question, setQuestion] = useState("");
  const boxRef = useRef<HTMLTextAreaElement>(null);
  const inputId = useId();

  // Grow the box with its text, up to about four lines.
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = `${Math.min(box.scrollHeight, 132)}px`;
    box.style.overflowY = box.scrollHeight > 132 ? "auto" : "hidden";
  }, [question]);

  const trimmed = question.trim();

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!trimmed || busy || disabled) return;
    setQuestion("");
    const accepted = await onAsk(trimmed);
    if (!accepted) setQuestion(trimmed);
  };

  return (
    <div className="askbar" role="region" aria-label="Ask a question">
      <form className="askbar__inner" onSubmit={submit}>
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
            disabled={disabled}
            placeholder="Ask in English, हिंदी, मराठी or romanized…"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e);
            }}
          />
          <button type="submit" className="askbar__send" disabled={!trimmed || busy || disabled} aria-label="Ask">
            {busy ? (
              <span className="askbar__spinner" aria-hidden />
            ) : (
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h13M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </button>
        </div>
      </form>
      <p className="askbar__hint">
        {llm && !llm.configured ? (
          <strong>No answer-model key is set: questions are searched, but answers need a key (see the README).</strong>
        ) : details ? (
          answerNotice(llm)
        ) : (
          `Every answer cites its page. The passages used are sent to ${llm?.provider ?? "the answer model"} to write it.`
        )}
      </p>
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
