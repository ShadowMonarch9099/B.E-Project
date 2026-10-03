"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestion, createConversation, latestConversation } from "@/lib/api";
import type { ChatMessage, Conversation, LlmStatus } from "@/lib/types";
import AskBar, { type AskableDoc } from "./AskBar";
import Thread, { type OpenCitation, type PendingAnswer } from "./Thread";

// The conversation about one document: the thread between the analysis and the fixed ask
// bar. Picking another document in the ask bar opens that document's latest conversation.
export default function Chat({ docs, llm, onOpen }: { docs: AskableDoc[]; llm?: LlmStatus; onOpen: OpenCitation }) {
  const [picked, setPicked] = useState<string | null>(null);
  const [conv, setConv] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState<PendingAnswer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const doc = docs.find((d) => d.id === picked) ?? docs[0];
  const docId = doc?.id;

  useEffect(() => {
    if (!docId) return;
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

  useEffect(() => {
    if (pending || messages.length) endRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [messages.length, pending?.text.length, pending]);

  const ask = useCallback(
    async (question: string) => {
      if (!docId || pending) return false;
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

  if (!doc) return null;
  return (
    <>
      <section className="chat" aria-label="Conversation">
        <div className="results__head">
          <h2>
            Questions <span className="count">{doc.name}</span>
          </h2>
          {messages.length > 0 && !pending && (
            <button type="button" className="btn-text" onClick={newConversation}>
              New conversation
            </button>
          )}
        </div>
        {messages.length === 0 && !pending && !error ? (
          <p className="chat__empty">
            Ask about this document in English, हिंदी, मराठी or romanized. Every sentence of an answer cites
            the page it comes from; when the document does not say, Sakshya abstains and shows the nearest
            pages.
          </p>
        ) : (
          <Thread messages={messages} pending={pending} error={error} onOpen={onOpen} onRetry={ask} />
        )}
        <div ref={endRef} className="chat__end" />
      </section>
      <AskBar docs={docs} selectedId={doc.id} onSelect={setPicked} onAsk={ask} busy={pending !== null} llm={llm} />
    </>
  );
}
