"use client";

import { useId, useRef, useState } from "react";
import { formatBytes, isPending, percent, statusText } from "@/lib/format";
import { useDetails } from "@/lib/details";
import type { DocSummary, UploadItem } from "@/lib/types";
import ResultCard from "./ResultCard";

// The left sidebar: add documents, see them being read, pick the one to ask about.
export default function Library({
  docs,
  loading,
  uploads,
  selectedId,
  busy,
  maxFileMb,
  onFiles,
  onSelect,
  onOpen,
  onDelete,
  onDetails,
  onDismissUpload,
  onClearAll,
}: {
  docs: DocSummary[];
  loading: boolean;
  uploads: UploadItem[];
  selectedId: string | null;
  busy: Set<string>;
  maxFileMb: number;
  onFiles: (files: FileList | null) => void;
  onSelect: (doc: DocSummary) => void;
  onOpen: (doc: DocSummary) => void;
  onDelete: (doc: DocSummary) => void;
  onDetails: (doc: DocSummary) => void;
  onDismissUpload: (id: string) => void;
  onClearAll: () => void;
}) {
  const [filter, setFilter] = useState("");
  const shown = filter.trim()
    ? docs.filter((d) => d.filename.toLowerCase().includes(filter.trim().toLowerCase()))
    : docs;

  return (
    <aside className="library" aria-label="Your documents">
      <Dropzone compact maxFileMb={maxFileMb} onFiles={onFiles} />
      {loading && <p className="library__none">Loading your documents…</p>}

      {docs.length + uploads.length > 0 && (
        <div className="library__head">
          <h2>
            Documents <span className="count">{docs.length}</span>
          </h2>
          {docs.length > 0 && (
            <button type="button" className="btn-text btn-text--danger" onClick={onClearAll}>
              Clear all
            </button>
          )}
        </div>
      )}

      {docs.length > 8 && (
        <input
          className="library__filter"
          type="search"
          placeholder="Find a document…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Find a document"
        />
      )}

      <ol className="doclist">
        {uploads.map((u) => (
          <li key={u.id}>
            {u.state === "rejected" ? (
              <div className="docrow docrow--static">
                <ResultCard result={u.detection} onDismiss={() => onDismissUpload(u.id)} />
              </div>
            ) : (
              <UploadRow item={u} onDismiss={() => onDismissUpload(u.id)} />
            )}
          </li>
        ))}
        {shown.map((d) => (
          <li key={d.id}>
            <DocRow
              doc={d}
              selected={d.id === selectedId}
              busy={busy.has(d.id)}
              onSelect={() => onSelect(d)}
              onOpen={() => onOpen(d)}
              onDelete={() => onDelete(d)}
              onDetails={() => onDetails(d)}
            />
          </li>
        ))}
      </ol>
      {filter && shown.length === 0 && <p className="library__none">No document matches “{filter}”.</p>}
    </aside>
  );
}

export function Dropzone({
  compact,
  maxFileMb,
  onFiles,
}: {
  compact: boolean;
  maxFileMb: number;
  onFiles: (files: FileList | null) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0);
  const inputId = useId();
  return (
    <label
      htmlFor={inputId}
      className={`dropzone${compact ? " dropzone--compact" : ""}${dragging ? " dropzone--active" : ""}`}
      onDragEnter={(e) => {
        e.preventDefault();
        depth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => {
        depth.current -= 1;
        if (depth.current <= 0) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        depth.current = 0;
        setDragging(false);
        onFiles(e.dataTransfer.files);
      }}
    >
      <input
        id={inputId}
        type="file"
        multiple
        className="visually-hidden"
        onChange={(e) => {
          onFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <span className="dropzone__icon" aria-hidden>
        <svg viewBox="0 0 64 64" width={compact ? 30 : 56} height={compact ? 30 : 56} fill="none" stroke="currentColor" strokeWidth="2">
          <path className="sheet sheet--back" d="M20 10h20l10 10v30H20z" />
          <path className="sheet sheet--front" d="M14 16h20l10 10v30H14z" fill="var(--surface)" />
          <path className="sheet sheet--front" d="M34 16v10h10" />
          <path className="sheet sheet--front" d="M21 36h16M21 42h16M21 48h10" strokeLinecap="round" />
        </svg>
      </span>
      <span className="dropzone__text">
        <span className="dropzone__title">{dragging ? "Release to add" : compact ? "Add documents" : "Drop documents here"}</span>
        <span className="dropzone__sub">
          {compact ? "Drop files or " : "or "}
          <span className="link-like">browse</span>
        </span>
        {!compact && <span className="dropzone__hint">PDF, DOCX or scanned images · up to {maxFileMb} MB each</span>}
      </span>
    </label>
  );
}

function DocRow({
  doc,
  selected,
  busy,
  onSelect,
  onOpen,
  onDelete,
  onDetails,
}: {
  doc: DocSummary;
  selected: boolean;
  busy: boolean;
  onSelect: () => void;
  onOpen: () => void;
  onDelete: () => void;
  onDetails: () => void;
}) {
  const { on: details } = useDetails();
  const pending = isPending(doc);
  const ready = doc.status === "indexed";
  const { done, total } = doc.progress;
  const scanned = (doc.stats.ocr_share ?? 0) >= 0.5;
  const meta = ready
    ? [doc.anchor === "paragraph" ? "Word document" : doc.page_count ? `${doc.page_count} ${doc.page_count === 1 ? "page" : "pages"}` : null,
       scanned ? "scanned" : null].filter(Boolean).join(" · ")
    : doc.status === "failed"
      ? "Could not be read"
      : statusText(doc);
  return (
    <div className={`docrow${selected ? " docrow--selected" : ""}${doc.status === "failed" ? " docrow--failed" : ""}`}>
      <button
        type="button"
        className="docrow__main"
        onClick={onSelect}
        disabled={!ready}
        aria-pressed={selected}
        title={doc.status === "failed" && doc.error ? doc.error : doc.filename}
      >
        <span className={`stamp docrow__stamp${pending ? " stamp--ghost" : ""}`} aria-hidden>
          {pending ? "···" : doc.stamp}
        </span>
        <span className="docrow__text">
          <span className="docrow__name">{doc.filename}</span>
          <span className="docrow__meta">{meta}</span>
        </span>
      </button>
      <span className="docrow__actions">
        {ready && (doc.has_viewer || doc.anchor === "paragraph") && (
          <button type="button" className="icon-btn" onClick={onOpen} aria-label={`Open ${doc.filename}`} title="Open">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </button>
        )}
        {details && (
          <button type="button" className="icon-btn" onClick={onDetails} aria-label={`Details of ${doc.filename}`} title="Details">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <circle cx="12" cy="12" r="9" />
              <path d="M12 11v6M12 7.5v.5" strokeLinecap="round" />
            </svg>
          </button>
        )}
        <button type="button" className="icon-btn icon-btn--danger" onClick={onDelete} disabled={busy}
                aria-label={`Delete ${doc.filename}`} title="Delete">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
            <path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </span>
      {pending && (
        <div className="docrow__progress">
          {total > 0 ? (
            <div className="bar bar--progress" aria-label={statusText(doc)}>
              <span style={{ width: percent(done / total) }} />
            </div>
          ) : (
            <div className="bar bar--indeterminate" aria-label={statusText(doc)}>
              <span />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function UploadRow({ item, onDismiss }: { item: UploadItem; onDismiss: () => void }) {
  return (
    <div className="docrow docrow--static">
      <div className="docrow__main">
        <span className="stamp stamp--ghost docrow__stamp" aria-hidden>
          {item.state === "error" ? "!" : "···"}
        </span>
        <span className="docrow__text">
          <span className="docrow__name" title={item.name}>
            {item.name}
          </span>
          <span className="docrow__meta">
            {item.state === "queued" && "Waiting to upload…"}
            {item.state === "uploading" && `Uploading ${percent(item.progress)}`}
            {item.state === "analysing" && "Checking the file…"}
            {item.state === "error" && <span className="error">{item.message}</span>}
            {item.state !== "error" && ` · ${formatBytes(item.size)}`}
          </span>
        </span>
      </div>
      {item.state === "error" && (
        <span className="docrow__actions docrow__actions--always">
          <button type="button" className="icon-btn" onClick={onDismiss} aria-label="Dismiss" title="Dismiss">
            ×
          </button>
        </span>
      )}
      {(item.state === "uploading" || item.state === "analysing") && (
        <div className="docrow__progress">
          {item.state === "uploading" ? (
            <div className="bar bar--progress">
              <span style={{ width: percent(item.progress) }} />
            </div>
          ) : (
            <div className="bar bar--indeterminate">
              <span />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
