"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestion, createConversation, latestConversation } from "@/lib/api";
import type { ChatMessage, Conversation, DocSummary, LlmStatus } from "@/lib/types";
import AskBar from "./AskBar";
import Thread, { type OpenCitation, type PendingAnswer } from "./Thread";

// The conversation about the selected document: its latest thread, then the ask bar.
export default function Chat({
  doc,
  llm,
  onOpen,
  onOpenDocument,
}: {
  doc: DocSummary;
  llm?: LlmStatus;
  onOpen: OpenCitation;
  onOpenDocument: () => void;
}) {
  const [conv, setConv] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState<PendingAnswer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const docId = doc.id;

  useEffect(() => {
    let live = true;
    setConv(null);
    setMessages([]);
    setError(null);
    latestConversation(docId)
      .then((c) => {
        if (!live) return;
        setConv(c);
        setMessages(c?.messages ?? []);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [docId]);

  // Keep the newest message in view.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages.length, pending?.text.length, pending]);

  const ask = useCallback(
    async (question: string) => {
      if (pending) return false;
      setError(null);
      setPending({ question, stage: "retrieving", text: "", showQuestion: true });
      try {
        const c = conv ?? (await createConversation(docId));
        if (!conv) setConv(c);
        abortRef.current = new AbortController();
        await askQuestion(c.id, question, {
          signal: abortRef.current.signal,
          onEvent: (event, data) => {
            if (event === "status") {
              if (data.user_message) {
                setMessages((prev) => [...prev, data.user_message as ChatMessage]);
                setPending((p) => (p ? { ...p, showQuestion: false } : p));
              }
              if (data.stage === "answering") setPending((p) => (p ? { ...p, stage: "answering" } : p));
            } else if (event === "delta") {
              setPending((p) => (p ? { ...p, stage: "answering", text: p.text + String(data.text ?? "") } : p));
            } else if (event === "answer") {
              setMessages((prev) => [...prev, data as unknown as ChatMessage]);
              setPending(null);
            } else if (event === "error") {
              setError(String(data.message ?? "Something went wrong."));
              setPending(null);
            }
          },
        });
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setPending(null);
        abortRef.current = null;
      }
      return true;
    },
    [conv, docId, pending],
  );

  useEffect(() => () => abortRef.current?.abort(), []);

  const newConversation = () => {
    setConv(null);
    setMessages([]);
    setError(null);
  };

  const pages = doc.anchor === "paragraph" ? "Word document" : doc.page_count ? `${doc.page_count} pages` : "";
  return (
    <section className="conversation" aria-label="Conversation">
      <header className="conversation__head">
        <div className="conversation__title">
          <span className="stamp docrow__stamp" aria-hidden>
            {doc.stamp}
          </span>
          <div>
            <h2 className="conversation__name" title={doc.filename}>
              {doc.filename}
            </h2>
            {pages && <p className="conversation__meta">{pages}</p>}
          </div>
        </div>
        <div className="conversation__actions">
          {(doc.has_viewer || doc.anchor === "paragraph") && (
            <button type="button" className="btn-text" onClick={onOpenDocument}>
              View document
            </button>
          )}
          {messages.length > 0 && !pending && (
            <button type="button" className="btn-text" onClick={newConversation}>
              New conversation
            </button>
          )}
        </div>
      </header>

      <div className="conversation__scroll" ref={scrollRef}>
        {messages.length === 0 && !pending && !error ? (
          <div className="conversation__empty">
            <p className="conversation__lead">Ask anything about this document.</p>
            <p>
              Every sentence of the answer links to the page it comes from. If the document does not say,
              Sakshya tells you so and shows the closest pages.
            </p>
            <p className="conversation__try">
              Specific questions work best: amounts, dates, names, survey or GR numbers, eligibility, who must do what.
            </p>
          </div>
        ) : (
          <Thread messages={messages} pending={pending} error={error} onOpen={onOpen} onRetry={ask} />
        )}
      </div>

      <AskBar onAsk={ask} busy={pending !== null} llm={llm} />
    </section>
  );
}
