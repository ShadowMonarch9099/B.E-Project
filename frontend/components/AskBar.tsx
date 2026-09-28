"use client";

import { useEffect, useId, useRef, useState } from "react";

export interface AskableDoc {
  id: string;
  name: string;
}

const MAX_CHARS = 1000;

// The question box for asking about an analysed document. It stays fixed at the bottom
// of the window. Answering (retrieval, evidence gate, cited answer: plan sections 6.3-6.7)
// is not built yet, so a question is kept in the box and the bar says so.
export default function AskBar({ docs }: { docs: AskableDoc[] }) {
  // null = follow the top document on the page; set once the user picks one.
  const [picked, setPicked] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const boxRef = useRef<HTMLTextAreaElement>(null);
  const inputId = useId();
  const selectId = useId();

  // Grow the box with its text, up to about four lines.
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = `${Math.min(box.scrollHeight, 132)}px`;
  }, [question]);

  // The picked document, or the top one if nothing was picked or the pick was cleared away.
  const doc = docs.find((d) => d.id === picked) ?? docs[0];
  const trimmed = question.trim();

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!trimmed || !doc) return;
    setStatus(
      "Answering isn't connected yet. Retrieval and cited answers are the next step; your question is kept here.",
    );
  };

  return (
    <div className="askbar" role="region" aria-label="Ask about a document">
      <form className="askbar__inner" onSubmit={submit}>
        <div className="askbar__scope">
          {docs.length > 1 ? (
            <>
              <label htmlFor={selectId}>Asking about</label>
              <select id={selectId} value={doc?.id} onChange={(e) => setPicked(e.target.value)}>
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
            onChange={(e) => {
              setQuestion(e.target.value);
              setStatus(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e);
            }}
          />
          <button type="submit" className="askbar__send" disabled={!trimmed} aria-label="Ask">
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12h13M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
        <p className="askbar__hint" aria-live="polite">
          {status ?? (
            <>
              <kbd>Enter</kbd> to ask · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line
              {question.length > MAX_CHARS * 0.8 && ` · ${MAX_CHARS - question.length} characters left`}
            </>
          )}
        </p>
      </form>
    </div>
  );
}
