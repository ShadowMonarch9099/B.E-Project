"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getPages, listDocuments } from "@/lib/api";
import type { DocSummary, PageText } from "@/lib/types";
import PdfViewer, { type ViewerTarget } from "./PdfViewer";

// Sakshya-Bench annotation (plan section 8): write a question, choose its slice, mark the
// gold document, page and evidence span; a second person annotates about 15% of questions
// without seeing the first annotation. Questions are exported as JSONL.

type Slice = "same_language" | "cross_language" | "romanized" | "unanswerable";
const SLICES: { value: Slice; label: string; hint: string }[] = [
  { value: "same_language", label: "Same language", hint: "Marathi question over a Marathi document" },
  { value: "cross_language", label: "Cross-language", hint: "English question over a Marathi order" },
  { value: "romanized", label: "Romanized / code-mixed", hint: "“ATKT ke liye kitne subject allowed hai?”" },
  { value: "unanswerable", label: "Unanswerable", hint: "The documents do not cover it" },
];

interface Annotation {
  annotator: string;
  role: "primary" | "second";
  doc_id: string | null;
  filename: string | null;
  page: number | null;
  span_text: string | null;
}

interface Question {
  id: string;
  text: string;
  slice: Slice;
  lang: string | null;
  answerable: boolean;
  doc_id: string | null;
  status: "draft" | "verified";
  needs_second: boolean;
  created_by: string;
  annotated_by: string[];
  annotations?: Annotation[];
}

interface Span {
  start: number;
  end: number;
  text: string;
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, { headers: { "Content-Type": "application/json" }, ...init });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status}).`);
  return body as T;
}

// A page's text; selecting part of it marks the evidence span (offsets into the page text).
function PageSpanPicker({
  docId,
  page,
  onPage,
  span,
  onSpan,
  onView,
}: {
  docId: string;
  page: number;
  onPage: (p: number) => void;
  span: Span | null;
  onSpan: (s: Span | null) => void;
  onView: () => void;
}) {
  const [pages, setPages] = useState<PageText[] | null>(null);
  const textRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    setPages(null);
    getPages(docId).then(setPages).catch(() => setPages([]));
  }, [docId]);
  const current = pages?.find((p) => p.page_no === page);

  const capture = () => {
    const sel = window.getSelection();
    const box = textRef.current;
    if (!sel || sel.rangeCount === 0 || !box || sel.isCollapsed) return;
    const range = sel.getRangeAt(0);
    if (!box.contains(range.startContainer) || !box.contains(range.endContainer)) return;
    const before = document.createRange();
    before.selectNodeContents(box);
    before.setEnd(range.startContainer, range.startOffset);
    const start = before.toString().length;
    const text = range.toString();
    if (text.trim()) onSpan({ start, end: start + text.length, text });
  };

  if (!pages) return <p className="meta">Loading pages…</p>;
  return (
    <div className="ann-pages">
      <div className="ann-pages__bar">
        <label>
          Page{" "}
          <select value={page} onChange={(e) => onPage(Number(e.target.value))}>
            {pages.map((p) => (
              <option key={p.page_no} value={p.page_no}>
                {p.page_no}
                {p.source === "ocr" ? " (OCR)" : ""}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn-text" onClick={onView}>
          View the page
        </button>
        {span && (
          <button type="button" className="btn-text" onClick={() => onSpan(null)}>
            Clear span
          </button>
        )}
      </div>
      <div ref={textRef} className="ann-text" onMouseUp={capture} onKeyUp={capture} tabIndex={0}>
        {current?.text || "No text on this page."}
      </div>
      <p className="meta">
        {span ? (
          <>
            Evidence span ({span.end - span.start} characters): <q className="ann-span">{span.text}</q>
          </>
        ) : (
          "Select the evidence in the page text above."
        )}
      </p>
    </div>
  );
}

export default function Annotator() {
  const [name, setName] = useState("");
  const [tab, setTab] = useState<"write" | "second" | "stats">("write");
  const [docs, setDocs] = useState<DocSummary[]>([]);
  const [docId, setDocId] = useState<string>("");
  const [page, setPage] = useState(1);
  const [span, setSpan] = useState<Span | null>(null);
  const [text, setText] = useState("");
  const [slice, setSlice] = useState<Slice>("same_language");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [queue, setQueue] = useState<Question[]>([]);
  const [current, setCurrent] = useState<Question | null>(null);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [viewer, setViewer] = useState<ViewerTarget | null>(null);

  useEffect(() => {
    try {
      setName(localStorage.getItem("sakshya.annotator") ?? "");
    } catch {}
    listDocuments()
      .then((d) => {
        const indexed = d.filter((x) => x.status === "indexed");
        setDocs(indexed);
        if (indexed[0]) setDocId(indexed[0].id);
      })
      .catch((e: Error) => setMessage(e.message));
  }, []);

  const saveName = (v: string) => {
    setName(v);
    try {
      localStorage.setItem("sakshya.annotator", v);
    } catch {}
  };

  const refresh = useCallback(async () => {
    if (docId) setQuestions((await call<{ questions: Question[] }>(`/api/bench/questions?doc_id=${docId}`)).questions);
    if (name.trim()) {
      setQueue((await call<{ questions: Question[] }>(`/api/bench/second?annotator=${encodeURIComponent(name.trim())}`)).questions);
    }
    setStats(await call<Record<string, unknown>>("/api/bench/stats"));
  }, [docId, name]);

  useEffect(() => {
    refresh().catch((e: Error) => setMessage(e.message));
  }, [refresh]);

  const doc = useMemo(() => docs.find((d) => d.id === docId), [docs, docId]);
  const answerable = slice !== "unanswerable";

  const submit = async () => {
    setMessage(null);
    if (!name.trim()) return setMessage("Enter your name first: annotations record who made them.");
    if (!text.trim()) return setMessage("Write the question.");
    if (answerable && !span) return setMessage("Select the evidence span on the gold page.");
    try {
      await call("/api/bench/questions", {
        method: "POST",
        body: JSON.stringify({
          text,
          slice,
          answerable,
          doc_id: docId,
          annotator: name.trim(),
          gold: answerable && span ? { doc_id: docId, page, span_start: span.start, span_end: span.end } : null,
        }),
      });
      setText("");
      setSpan(null);
      setMessage("Saved.");
      await refresh();
    } catch (e) {
      setMessage((e as Error).message);
    }
  };

  const submitSecond = async () => {
    if (!current) return;
    setMessage(null);
    try {
      await call(`/api/bench/questions/${current.id}/second`, {
        method: "POST",
        body: JSON.stringify({
          annotator: name.trim(),
          gold: span ? { doc_id: docId, page, span_start: span.start, span_end: span.end } : null,
        }),
      });
      setCurrent(null);
      setSpan(null);
      setMessage("Second annotation saved.");
      await refresh();
    } catch (e) {
      setMessage((e as Error).message);
    }
  };

  const remove = async (id: string) => {
    if (!window.confirm("Delete this question and its annotations?")) return;
    await fetch(`/api/bench/questions/${id}`, { method: "DELETE" });
    await refresh();
  };

  const openPage = () =>
    doc && setViewer({ docId: doc.id, filename: doc.filename, page, anchor: doc.anchor, boxes: [], label: `p. ${page}` });

  const picker = docId ? (
    <>
      <label className="ann-field">
        <span>Gold document</span>
        <select
          value={docId}
          onChange={(e) => {
            setDocId(e.target.value);
            setPage(1);
            setSpan(null);
          }}
        >
          {docs.map((d) => (
            <option key={d.id} value={d.id}>
              {d.filename}
            </option>
          ))}
        </select>
      </label>
      <PageSpanPicker docId={docId} page={page} onPage={(p) => { setPage(p); setSpan(null); }} span={span} onSpan={setSpan} onView={openPage} />
    </>
  ) : (
    <p className="note">Index some documents first (on the main page).</p>
  );

  const agreement = (stats?.agreement ?? {}) as Record<string, number | null>;
  return (
    <div className="ann">
      <div className="ann-head">
        <label className="ann-field ann-name">
          <span>Your name</span>
          <input value={name} onChange={(e) => saveName(e.target.value)} placeholder="e.g. Tanmay" maxLength={64} />
        </label>
        <div className="ann-tabs" role="tablist">
          {(["write", "second", "stats"] as const).map((t) => (
            <button key={t} type="button" role="tab" aria-selected={tab === t} className={tab === t ? "is-on" : ""} onClick={() => { setTab(t); setSpan(null); }}>
              {t === "write" ? "Write questions" : t === "second" ? `Second annotation (${queue.length})` : "Agreement & export"}
            </button>
          ))}
        </div>
      </div>
      {message && <p className="notice">{message}</p>}

      {tab === "write" && (
        <section className="ann-grid">
          <div className="ann-form">
            <label className="ann-field">
              <span>Question</span>
              <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} placeholder="Write the question in the language and script a user would type." />
            </label>
            <fieldset className="ann-slices">
              <legend>Slice</legend>
              {SLICES.map((s) => (
                <label key={s.value} title={s.hint}>
                  <input type="radio" name="slice" value={s.value} checked={slice === s.value} onChange={() => setSlice(s.value)} />
                  {s.label}
                  <span className="meta"> · {s.hint}</span>
                </label>
              ))}
            </fieldset>
            {answerable ? picker : <p className="note">An unanswerable question has no gold page; it is about the selected document&apos;s topic but not answered in it.</p>}
            <button type="button" className="btn-primary" onClick={submit}>
              Save question
            </button>
          </div>
          <div className="ann-list">
            <h3>Questions on this document ({questions.length})</h3>
            <ol>
              {questions.map((q) => (
                <li key={q.id}>
                  <p className="ann-q">{q.text}</p>
                  <p className="meta">
                    {q.slice.replace("_", " ")} · {q.lang ?? "?"} · by {q.created_by}
                    {q.annotations?.[0]?.page ? ` · p. ${q.annotations[0].page}` : ""}
                    {q.needs_second ? ` · second annotation: ${q.annotated_by.includes("second") ? "done" : "waiting"}` : ""}
                    {q.status === "draft" ? " · DRAFT (not exported)" : ""}
                  </p>
                  <button type="button" className="btn-text btn-text--danger" onClick={() => remove(q.id)}>
                    Delete
                  </button>
                </li>
              ))}
            </ol>
          </div>
        </section>
      )}

      {tab === "second" && (
        <section className="ann-grid">
          <div className="ann-list">
            <h3>Waiting for you ({queue.length})</h3>
            {!name.trim() && <p className="note">Enter your name to see the questions others wrote.</p>}
            <ol>
              {queue.map((q) => (
                <li key={q.id}>
                  <button type="button" className={`ann-pick${current?.id === q.id ? " is-on" : ""}`} onClick={() => { setCurrent(q); setSpan(null); if (q.doc_id) setDocId(q.doc_id); }}>
                    {q.text}
                  </button>
                </li>
              ))}
            </ol>
          </div>
          <div className="ann-form">
            {current ? (
              <>
                <p className="ann-q">{current.text}</p>
                <p className="meta">Mark the gold page and span yourself; the first annotation is hidden. {current.answerable ? "" : "If you think it is unanswerable, save without a span."}</p>
                {picker}
                <button type="button" className="btn-primary" onClick={submitSecond}>
                  Save second annotation
                </button>
              </>
            ) : (
              <p className="note">Pick a question on the left.</p>
            )}
          </div>
        </section>
      )}

      {tab === "stats" && stats && (
        <section className="ann-stats">
          <dl className="facts">
            <div><dt>Verified questions</dt><dd>{String(stats.questions)}</dd></div>
            <div><dt>Drafts</dt><dd>{String(stats.drafts)}</dd></div>
            <div><dt>By slice</dt><dd>{JSON.stringify(stats.by_slice)}</dd></div>
            <div><dt>By language</dt><dd>{JSON.stringify(stats.by_lang)}</dd></div>
            <div><dt>By split</dt><dd>{JSON.stringify(stats.by_split)}</dd></div>
            <div><dt>Double-annotated</dt><dd>{String(agreement.double_annotated ?? 0)}</dd></div>
            <div><dt>Page agreement · Cohen&apos;s κ</dt><dd>{agreement.page_agreement ?? "—"} · {agreement.page_kappa ?? "—"}</dd></div>
            <div><dt>Span token-F1</dt><dd>{agreement.span_token_f1 ?? "—"}</dd></div>
            <div className="facts__wide"><dt>Manifest</dt><dd>{JSON.stringify(stats.manifest)}</dd></div>
          </dl>
          <p>
            <a className="btn-text" href="/api/bench/export" download>
              Download the benchmark (JSONL)
            </a>
          </p>
        </section>
      )}
      {viewer && <PdfViewer target={viewer} onClose={() => setViewer(null)} />}
    </div>
  );
}
