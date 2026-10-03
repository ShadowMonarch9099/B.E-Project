"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { deleteDocument, getHealth, listDocuments, reindexDocument, uploadDocument } from "@/lib/api";
import { formatBytes, isPending, percent } from "@/lib/format";
import { createLimiter } from "@/lib/limit";
import type { Citation, DocSummary, Health, UploadItem } from "@/lib/types";
import Chat from "./Chat";
import DocumentCard from "./DocumentCard";
import ResultCard from "./ResultCard";

const DEFAULT_LIMITS = { max_file_mb: 200, max_files: 20 };
const POLL_MS = 1500;

// Large files are held in memory while the backend reads them, so upload two at a time.
const uploadSlot = createLimiter(2);

export default function Workspace() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [docs, setDocs] = useState<DocSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const inputId = useId();
  const dragDepth = useRef(0);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments());
      setListError(null);
    } catch (e) {
      setListError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    getHealth()
      .then((h) => setHealth(h))
      .catch(() => setOffline(true));
    refresh();
  }, [refresh]);

  // Poll while any document is still being ingested.
  const anyPending = docs?.some(isPending) ?? false;
  useEffect(() => {
    if (!anyPending) return;
    const t = setTimeout(refresh, POLL_MS);
    return () => clearTimeout(t);
  }, [anyPending, docs, refresh]);

  const limits = health?.limits ?? DEFAULT_LIMITS;

  const setUpload = useCallback((id: string, next: UploadItem | null) => {
    setUploads((prev) => (next ? prev.map((u) => (u.id === id ? next : u)) : prev.filter((u) => u.id !== id)));
  }, []);

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list || list.length === 0) return;
      let picked = Array.from(list);
      setNotice(null);
      if (picked.length > limits.max_files) {
        setNotice(`Only the first ${limits.max_files} files were taken. Upload the rest in another batch.`);
        picked = picked.slice(0, limits.max_files);
      }
      const batch: UploadItem[] = picked.map((f) => ({ id: crypto.randomUUID(), name: f.name, size: f.size, state: "queued" }));
      setUploads((prev) => [...batch, ...prev]);

      picked.forEach((file, i) => {
        const base = { id: batch[i].id, name: file.name, size: file.size };
        if (file.size > limits.max_file_mb * 1024 * 1024) {
          setUpload(base.id, { ...base, state: "error", message: `Larger than ${limits.max_file_mb} MB.` });
          return;
        }
        uploadSlot(() => {
          setUpload(base.id, { ...base, state: "uploading", progress: 0 });
          return uploadDocument(file, (fraction) =>
            setUpload(base.id, fraction < 1 ? { ...base, state: "uploading", progress: fraction } : { ...base, state: "analysing" }),
          );
        })
          .then((res) => {
            if (res.kind === "rejected") {
              setUpload(base.id, { ...base, state: "rejected", detail: res.detail, detection: res.detection });
              return;
            }
            setUpload(base.id, null);
            if (res.duplicate) setNotice(`${file.name} is already here as “${res.document.filename}”.`);
            setDocs((prev) => [res.document, ...(prev ?? []).filter((d) => d.id !== res.document.id)]);
          })
          .catch((err: Error) => setUpload(base.id, { ...base, state: "error", message: err.message }));
      });
    },
    [limits, setUpload],
  );

  const withBusy = async (id: string, task: () => Promise<void>) => {
    setBusy((b) => new Set(b).add(id));
    try {
      await task();
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setBusy((b) => {
        const next = new Set(b);
        next.delete(id);
        return next;
      });
    }
  };

  const remove = (doc: DocSummary) =>
    withBusy(doc.id, async () => {
      if (!window.confirm(`Delete “${doc.filename}” and everything indexed from it?`)) return;
      await deleteDocument(doc.id);
      setDocs((prev) => (prev ?? []).filter((d) => d.id !== doc.id));
    });

  const reindex = (doc: DocSummary) =>
    withBusy(doc.id, async () => {
      const next = await reindexDocument(doc.id);
      setDocs((prev) => (prev ?? []).map((d) => (d.id === doc.id ? next : d)));
    });

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  // Opens the cited page of the original document. (The in-app viewer with the passage
  // highlighted replaces this in phase 5.)
  const openCitation = useCallback((c: Citation) => {
    window.open(`/api/documents/${c.doc_id}/file#page=${c.page_no}`, "_blank", "noopener");
  }, []);

  const indexed = (docs ?? []).filter((d) => d.status === "indexed");
  const askable = indexed.map((d) => ({ id: d.id, name: d.filename }));
  const count = (docs?.length ?? 0) + uploads.length;
  const dbDown = health ? !health.database.available : false;

  return (
    <>
      <div className="status-line" role="status">
        <span className={`dot ${offline ? "dot--off" : health ? "dot--on" : ""}`} aria-hidden />
        {offline
          ? "Backend offline: start it on port 8000"
          : health
            ? [
                `Detector ready · ${health.detector.name} ${health.detector.version}`,
                health.ocr.available
                  ? `OCR: ${health.ocr.engine} ${health.ocr.version} (${health.ocr.languages.join("+")})`
                  : "OCR unavailable",
                health.database.available ? "Database connected" : "Database offline",
              ].join(" · ")
            : "Connecting…"}
      </div>

      {dbDown && (
        <p className="notice notice--warn">
          Documents cannot be stored because the database is not reachable. Start it with{" "}
          <code>docker compose up -d db</code> and restart the backend.
        </p>
      )}

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
        <span className="dropzone__title">{dragging ? "Release to add" : "Drop documents here"}</span>
        <span className="dropzone__sub">
          or <span className="link-like">browse your computer</span>
        </span>
        <span className="dropzone__hint">
          PDF, DOCX and scans (PNG, JPEG, TIFF) · up to {limits.max_file_mb} MB each · stored on this machine so
          answers can open the cited page
        </span>
      </label>

      {notice && <p className="notice">{notice}</p>}
      {listError && !dbDown && <p className="notice notice--warn">{listError}</p>}

      {count > 0 && (
        <section className="results" aria-live="polite">
          <div className="results__head">
            <h2>
              Documents <span className="count">{docs?.length ?? 0}</span>
            </h2>
          </div>
          <ol className="cards">
            {uploads.map((u, i) => (
              <li key={u.id} className="card" style={{ "--i": Math.min(i, 8) } as React.CSSProperties}>
                {u.state === "rejected" ? (
                  <ResultCard result={u.detection} onDismiss={() => setUpload(u.id, null)} />
                ) : (
                  <Pending item={u} onDismiss={() => setUpload(u.id, null)} />
                )}
              </li>
            ))}
            {(docs ?? []).map((d, i) => (
              <li key={d.id} className="card" style={{ "--i": Math.min(i + uploads.length, 8) } as React.CSSProperties}>
                <DocumentCard doc={d} busy={busy.has(d.id)} onDelete={() => remove(d)} onReindex={() => reindex(d)} />
              </li>
            ))}
          </ol>
        </section>
      )}

      {askable.length > 0 && <Chat docs={askable} llm={health?.llm} onOpen={openCitation} />}
    </>
  );
}

function Pending({ item, onDismiss }: { item: UploadItem; onDismiss: () => void }) {
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
            <p className="meta">Checking the file type…</p>
          </>
        )}
        {item.state === "error" && (
          <>
            <p className="error">{item.message}</p>
            <button type="button" className="btn-text" onClick={onDismiss}>
              Dismiss
            </button>
          </>
        )}
      </div>
    </div>
  );
}
