"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { clearAllDocuments, deleteDocument, getHealth, listDocuments, reindexDocument, uploadDocument } from "@/lib/api";
import { useDetails, useMedia } from "@/lib/details";
import { isPending } from "@/lib/format";
import { createLimiter } from "@/lib/limit";
import type { Citation, DocSummary, Health, UploadItem } from "@/lib/types";
import Chat from "./Chat";
import DocDetails from "./DocDetails";
import Library, { Dropzone } from "./Library";
import PdfViewer, { type ViewerTarget } from "./PdfViewer";

const DEFAULT_LIMITS = { max_file_mb: 200, max_files: 20 };
const POLL_MS = 1500;

// Large files are held in memory while the backend reads them, so upload two at a time.
const uploadSlot = createLimiter(2);

// The app: documents on the left, the conversation in the middle, the cited page on the
// right (on narrower screens the page opens over the conversation).
export default function Workspace() {
  const { on: details, toggle: toggleDetails } = useDetails();
  const wide = useMedia("(min-width: 1180px)");
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [docs, setDocs] = useState<DocSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [viewer, setViewer] = useState<ViewerTarget | null>(null);
  const [detailsDoc, setDetailsDoc] = useState<DocSummary | null>(null);

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

  // Poll while any document is still being read.
  const anyPending = docs?.some((d) => isPending(d)) ?? false;
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
        setNotice(`Only the first ${limits.max_files} files were taken. Add the rest in another batch.`);
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
      if (!window.confirm(`Delete “${doc.filename}” and its conversations?`)) return;
      await deleteDocument(doc.id);
      setDocs((prev) => (prev ?? []).filter((d) => d.id !== doc.id));
      if (viewer?.docId === doc.id) setViewer(null);
    });

  const reindex = (doc: DocSummary) =>
    withBusy(doc.id, async () => {
      const next = await reindexDocument(doc.id);
      setDocs((prev) => (prev ?? []).map((d) => (d.id === doc.id ? next : d)));
      setDetailsDoc(null);
    });

  const clearAll = async () => {
    const n = docs?.length ?? 0;
    if (!n || !window.confirm(`Delete all ${n} documents and their conversations? This cannot be undone.`)) return;
    try {
      await clearAllDocuments();
      setDocs([]);
      setSelectedId(null);
      setViewer(null);
      setNotice(null);
    } catch (e) {
      setNotice((e as Error).message);
    }
  };

  // A page chip, a nearest page or "View document" opens the page with the passage highlighted.
  const openCitation = useCallback((c: Citation) => {
    setViewer({
      docId: c.doc_id,
      filename: c.filename,
      page: c.page_no,
      anchor: c.anchor,
      boxes: c.bbox ?? [],
      label: c.anchor === "paragraph" ? `¶ ${c.para_idx + 1}` : `p. ${c.page_no}`,
      paraIdx: c.para_idx,
      citedText: c.cited_text,
      checkPage: c.check_page,
      ocrConf: c.ocr_conf,
    });
  }, []);
  const openDocument = (d: DocSummary) =>
    setViewer({ docId: d.id, filename: d.filename, page: 1, anchor: d.anchor, boxes: [], label: "p. 1" });

  const all = docs ?? [];
  const indexed = all.filter((d) => d.status === "indexed");
  const selected = indexed.find((d) => d.id === selectedId) ?? indexed[0] ?? null;
  const empty = docs !== null && all.length === 0 && uploads.length === 0;
  const dbDown = health ? !health.database.available : false;

  const status = offline
    ? { text: "Offline", tone: "off", title: "The backend is not reachable on port 8000." }
    : !health
      ? { text: "Connecting…", tone: "", title: "" }
      : dbDown
        ? { text: "Database offline", tone: "off", title: "Start it with: docker compose up -d db" }
        : {
            text: "Ready",
            tone: "on",
            title: [
              `${health.detector.name} ${health.detector.version}`,
              health.ocr.available ? `OCR: ${health.ocr.engine} ${health.ocr.version} (${health.ocr.languages.join("+")})` : "OCR unavailable",
              "Database connected",
            ].join(" · "),
          };

  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <span className="wordmark">Sakshya</span>
          <span className="stamp stamp--brand" lang="hi" aria-label="साक्ष्य, evidence">
            साक्ष्य
          </span>
        </div>
        <nav className="app__nav">
          <span className={`status status--${status.tone}`} title={status.title} role="status">
            <span className={`dot ${status.tone === "on" ? "dot--on" : status.tone === "off" ? "dot--off" : ""}`} aria-hidden />
            {status.text}
            {details && health && status.tone === "on" && <span className="status__more"> · {status.title}</span>}
          </span>
          <label className="switch" title="Show how documents were read and how each answer was found">
            <input type="checkbox" checked={details} onChange={toggleDetails} />
            <span className="switch__track" aria-hidden />
            Technical details
          </label>
          <Link className="btn-text" href="/annotate">
            Sakshya-Bench
          </Link>
        </nav>
      </header>

      {(notice || (listError && !dbDown) || dbDown) && (
        <div className="app__notices">
          {dbDown && (
            <p className="notice notice--warn">
              Documents cannot be stored because the database is not reachable. Start it with{" "}
              <code>docker compose up -d db</code> and restart the backend.
            </p>
          )}
          {listError && !dbDown && <p className="notice notice--warn">{listError}</p>}
          {notice && (
            <p className="notice">
              {notice}
              <button type="button" className="btn-text notice__close" onClick={() => setNotice(null)} aria-label="Dismiss">
                ×
              </button>
            </p>
          )}
        </div>
      )}

      {empty ? (
        <main className="welcome">
          <p className="eyebrow">English · हिंदी · मराठी · PDF, Word and scans</p>
          <h1>
            Ask your documents. <span className="nowrap">Check the page.</span>
          </h1>
          <p className="lede">
            Add your files. Sakshya reads every page, including old scans, and every answer points to the page it
            comes from.
          </p>
          <Dropzone compact={false} maxFileMb={limits.max_file_mb} onFiles={handleFiles} />
        </main>
      ) : (
        <main className={`shell${wide ? " shell--wide" : ""}`}>
          <Library
            docs={all}
            loading={docs === null}
            uploads={uploads}
            selectedId={selected?.id ?? null}
            busy={busy}
            maxFileMb={limits.max_file_mb}
            onFiles={handleFiles}
            onSelect={(d) => setSelectedId(d.id)}
            onOpen={openDocument}
            onDelete={remove}
            onDetails={setDetailsDoc}
            onDismissUpload={(id) => setUpload(id, null)}
            onClearAll={clearAll}
          />
          {selected ? (
            <Chat key={selected.id} doc={selected} llm={health?.llm} onOpen={openCitation}
                  onOpenDocument={() => openDocument(selected)} />
          ) : (
            <section className="conversation conversation--waiting">
              <div className="conversation__empty">
                <p className="conversation__lead">Reading your documents…</p>
                <p>You can ask questions as soon as a document is ready. Scanned pages take longer.</p>
              </div>
            </section>
          )}
          {wide && (
            <aside className="evidence" aria-label="Cited page">
              {viewer ? (
                <PdfViewer target={viewer} onClose={() => setViewer(null)} inline />
              ) : (
                <div className="evidence__empty">
                  <span className="stamp evidence__stamp" aria-hidden>
                    p. 4
                  </span>
                  <p className="evidence__title">The cited page opens here</p>
                  <p>Click a page chip in an answer to see the original page with the passage highlighted.</p>
                </div>
              )}
            </aside>
          )}
        </main>
      )}

      {viewer && !wide && <PdfViewer target={viewer} onClose={() => setViewer(null)} />}
      {detailsDoc && <DocDetails doc={detailsDoc} onClose={() => setDetailsDoc(null)} onReindex={() => reindex(detailsDoc)} />}
    </div>
  );
}
