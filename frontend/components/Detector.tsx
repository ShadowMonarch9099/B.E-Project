"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { detectFile, getHealth, runOcr } from "@/lib/api";
import { formatBytes, percent } from "@/lib/format";
import { createLimiter } from "@/lib/limit";
import type { Health, Item, OcrState } from "@/lib/types";
import AskBar from "./AskBar";
import ResultCard from "./ResultCard";

const DEFAULT_LIMITS = { max_file_mb: 200, max_files: 20 };

// Large files are held in memory while the backend reads them, so upload two at a time
// and OCR one file at a time (each OCR job already uses several CPU cores).
const uploadSlot = createLimiter(2);
const ocrSlot = createLimiter(1);

export default function Detector() {
  const [items, setItems] = useState<Item[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const inputId = useId();
  const dragDepth = useRef(0);
  const files = useRef(new Map<string, File>()); // kept so OCR can be run (or retried) after detection

  useEffect(() => {
    getHealth()
      .then((h) => setHealth(h))
      .catch(() => setOffline(true));
  }, []);

  const limits = health?.limits ?? DEFAULT_LIMITS;

  const update = useCallback((id: string, next: Item) => {
    setItems((prev) => prev.map((it) => (it.id === id ? next : it)));
  }, []);

  const setOcr = useCallback((id: string, ocr: OcrState) => {
    setItems((prev) => prev.map((it) => (it.id === id && it.state === "done" ? { ...it, ocr } : it)));
  }, []);

  // Scanned PDFs and images go to OCR (plan section 6.1 step 2) as soon as their type is known.
  const startOcr = useCallback(
    (id: string) => {
      const file = files.current.get(id);
      if (!file) return;
      setOcr(id, { state: "running" });
      ocrSlot(() => runOcr(file))
        .then((data) => setOcr(id, { state: "done", data }))
        .catch((err: Error) => setOcr(id, { state: "error", message: err.message }));
    },
    [setOcr],
  );

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list || list.length === 0) return;
      let picked = Array.from(list);
      setNotice(null);
      if (picked.length > limits.max_files) {
        setNotice(`Only the first ${limits.max_files} files were taken. Upload the rest in another batch.`);
        picked = picked.slice(0, limits.max_files);
      }
      const batch: Item[] = picked.map((f) => ({
        id: crypto.randomUUID(),
        name: f.name,
        size: f.size,
        state: "queued",
      }));
      setItems((prev) => [...batch, ...prev]);

      picked.forEach((file, i) => {
        const base = { id: batch[i].id, name: file.name, size: file.size };
        files.current.set(base.id, file);
        if (file.size > limits.max_file_mb * 1024 * 1024) {
          update(base.id, { ...base, state: "error", message: `Larger than ${limits.max_file_mb} MB.` });
          return;
        }
        uploadSlot(() => {
          update(base.id, { ...base, state: "uploading", progress: 0 });
          return detectFile(file, (fraction) => {
            update(base.id, fraction < 1
              ? { ...base, state: "uploading", progress: fraction }
              : { ...base, state: "analysing" });
          });
        })
          .then((result) => {
            update(base.id, { ...base, state: "done", result });
            if (result.ingestion.needs_ocr) startOcr(base.id);
          })
          .catch((err: Error) => update(base.id, { ...base, state: "error", message: err.message }));
      });
    },
    [limits, update, startOcr],
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const done = items.filter((it) => it.state === "done");
  const askable = done.map((it) => ({ id: it.id, name: it.name }));
  const kinds = done.reduce<Record<string, number>>((acc, it) => {
    if (it.state === "done") acc[it.result.type.kind] = (acc[it.result.type.kind] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <>
      <div className="status-line" role="status">
        <span className={`dot ${offline ? "dot--off" : health ? "dot--on" : ""}`} aria-hidden />
        {offline
          ? "Detector offline: start the backend on port 8000"
          : health
            ? `Detector ready · ${health.detector.name} ${health.detector.version} (${health.detector.model}) · ${
                health.ocr.available
                  ? `OCR: ${health.ocr.engine} ${health.ocr.version} (${health.ocr.languages.join("+")})`
                  : "OCR unavailable"
              }`
            : "Connecting to the detector…"}
      </div>

      <label
        htmlFor={inputId}
        className={`dropzone${dragging ? " dropzone--active" : ""}`}
        onDragEnter={(e) => {
          e.preventDefault();
          dragDepth.current += 1;
          setDragging(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => {
          dragDepth.current -= 1;
          if (dragDepth.current <= 0) setDragging(false);
        }}
        onDrop={onDrop}
      >
        <input
          id={inputId}
          type="file"
          multiple
          className="visually-hidden"
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <span className="dropzone__icon" aria-hidden>
          <svg viewBox="0 0 64 64" width="56" height="56" fill="none" stroke="currentColor" strokeWidth="2">
            <path className="sheet sheet--back" d="M20 10h20l10 10v30H20z" />
            <path className="sheet sheet--front" d="M14 16h20l10 10v30H14z" fill="var(--surface)" />
            <path className="sheet sheet--front" d="M34 16v10h10" />
            <path className="sheet sheet--front" d="M21 36h16M21 42h16M21 48h10" strokeLinecap="round" />
          </svg>
        </span>
        <span className="dropzone__title">
          {dragging ? "Release to identify" : "Drop files here"}
        </span>
        <span className="dropzone__sub">
          or <span className="link-like">browse your computer</span>
        </span>
        <span className="dropzone__hint">
          Any file type · up to {limits.max_file_mb} MB each · {limits.max_files} at a time · read in
          memory, never stored
        </span>
      </label>

      {notice && <p className="notice">{notice}</p>}

      {items.length > 0 && (
        <section className="results" aria-live="polite">
          <div className="results__head">
            <h2>
              Results <span className="count">{items.length}</span>
            </h2>
            <button type="button" className="btn-text" onClick={() => {
                setItems([]);
                files.current.clear();
              }}>
              Clear all
            </button>
          </div>
          {Object.keys(kinds).length > 0 && (
            <ul className="summary">
              {Object.entries(kinds).map(([kind, n]) => (
                <li key={kind}>
                  {kind} <span>{n}</span>
                </li>
              ))}
            </ul>
          )}
          <ol className="cards">
            {items.map((it, i) => (
              <li key={it.id} className="card" style={{ "--i": Math.min(i, 8) } as React.CSSProperties}>
                {it.state === "done" ? (
                  <ResultCard result={it.result} ocr={it.ocr} onRetryOcr={() => startOcr(it.id)} />
                ) : (
                  <Pending item={it} />
                )}
              </li>
            ))}
          </ol>
        </section>
      )}

      {askable.length > 0 && <AskBar docs={askable} />}
    </>
  );
}

function Pending({ item }: { item: Item }) {
  return (
    <div className="pending">
      <div className="stamp stamp--ghost" aria-hidden>
        {item.state === "error" ? "!" : "···"}
      </div>
      <div className="pending__body">
        <p className="filename" title={item.name}>
          {item.name}
        </p>
        <p className="meta">{formatBytes(item.size)}</p>
        {item.state === "queued" && <p className="meta">Waiting to upload…</p>}
        {item.state === "uploading" && (
          <>
            <div className="bar bar--progress" aria-label={`Uploading ${percent(item.progress)}`}>
              <span style={{ width: percent(item.progress) }} />
            </div>
            <p className="meta">Uploading {percent(item.progress)}</p>
          </>
        )}
        {item.state === "analysing" && (
          <>
            <div className="bar bar--indeterminate" aria-label="Analysing">
              <span />
            </div>
            <p className="meta">Reading the content…</p>
          </>
        )}
        {item.state === "error" && <p className="error">{item.message}</p>}
      </div>
    </div>
  );
}
