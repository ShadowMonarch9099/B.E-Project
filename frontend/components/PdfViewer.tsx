"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { fileUrl, getPages } from "@/lib/api";
import { percent } from "@/lib/format";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";

// What the viewer opens: a document, a page, and the boxes to highlight on it.
export interface ViewerTarget {
  docId: string;
  filename: string;
  page: number;
  anchor: "page" | "paragraph";
  boxes: number[][]; // [x0, y0, x1, y1] in PDF points, top-left origin (backend/app/ingest/pages.py)
  label: string; // "p. 4" or "¶ 12"
  paraIdx?: number;
  citedText?: string;
  checkPage?: boolean;
  ocrConf?: number | null;
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3;

// One pdf.js document per document id, kept while the page is open.
const docCache = new Map<string, Promise<PDFDocumentProxy>>();

async function loadPdf(docId: string): Promise<PDFDocumentProxy> {
  let p = docCache.get(docId);
  if (!p) {
    p = import("pdfjs-dist").then((pdfjs) => {
      pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
      return pdfjs.getDocument({ url: fileUrl(docId) }).promise;
    });
    docCache.set(docId, p);
    p.catch(() => docCache.delete(docId));
  }
  return p;
}

function PdfPage({ target }: { target: ViewerTarget }) {
  const [pageNo, setPageNo] = useState(target.page);
  const [zoom, setZoom] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [size, setSize] = useState<{ w: number; h: number; k: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const markRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => setPageNo(target.page), [target]);

  // Fit the page to the panel's width; re-render when the panel is resized.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!width) return;
    let task: RenderTask | null = null;
    let live = true;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const pdf = await loadPdf(target.docId);
        if (!live) return;
        setNumPages(pdf.numPages);
        const page = await pdf.getPage(Math.min(Math.max(1, pageNo), pdf.numPages));
        const base = page.getViewport({ scale: 1 });
        const cssWidth = Math.max(240, (width - 24) * zoom);
        const k = cssWidth / base.width; // CSS pixels per PDF point
        const dpr = window.devicePixelRatio || 1;
        const viewport = page.getViewport({ scale: k * dpr });
        const canvas = canvasRef.current;
        if (!canvas || !live) return;
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        canvas.style.width = `${Math.floor(viewport.width / dpr)}px`;
        canvas.style.height = `${Math.floor(viewport.height / dpr)}px`;
        task = page.render({ canvas, viewport });
        await task.promise;
        if (!live) return;
        setSize({ w: viewport.width / dpr, h: viewport.height / dpr, k });
        setLoading(false);
      } catch (e) {
        if (!live || (e as Error).name === "RenderingCancelledException") return;
        setError(`Could not open the PDF (${(e as Error).message}).`);
        setLoading(false);
      }
    })();
    return () => {
      live = false;
      task?.cancel();
    };
  }, [target.docId, pageNo, zoom, width]);

  const onTargetPage = pageNo === target.page;

  // Bring the highlighted passage into view once the page is drawn.
  useEffect(() => {
    if (size && onTargetPage && target.boxes.length) markRef.current?.scrollIntoView({ block: "center" });
  }, [size, onTargetPage, target]);

  const go = useCallback(
    (delta: number) => setPageNo((p) => Math.min(Math.max(1, p + delta), numPages ?? p + delta)),
    [numPages],
  );

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "PageDown") {
      e.preventDefault();
      go(1);
    } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
      e.preventDefault();
      go(-1);
    } else if (e.key === "+" || e.key === "=") {
      setZoom((z) => Math.min(MAX_ZOOM, z + 0.25));
    } else if (e.key === "-") {
      setZoom((z) => Math.max(MIN_ZOOM, z - 0.25));
    }
  };

  return (
    <div className="viewer__body" onKeyDown={onKey}>
      <div className="viewer__tools" role="toolbar" aria-label="Page controls">
        <button type="button" className="viewer__btn" onClick={() => go(-1)} disabled={pageNo <= 1} aria-label="Previous page">
          ‹
        </button>
        <span className="viewer__pageno" aria-live="polite">
          p. {pageNo}
          {numPages ? ` of ${numPages}` : ""}
        </span>
        <button
          type="button"
          className="viewer__btn"
          onClick={() => go(1)}
          disabled={numPages !== null && pageNo >= numPages}
          aria-label="Next page"
        >
          ›
        </button>
        <span className="viewer__sep" aria-hidden />
        <button type="button" className="viewer__btn" onClick={() => setZoom((z) => Math.max(MIN_ZOOM, z - 0.25))} aria-label="Zoom out">
          −
        </button>
        <span className="viewer__zoom">{Math.round(zoom * 100)}%</span>
        <button type="button" className="viewer__btn" onClick={() => setZoom((z) => Math.min(MAX_ZOOM, z + 0.25))} aria-label="Zoom in">
          +
        </button>
        {!onTargetPage && (
          <button type="button" className="btn-text viewer__back" onClick={() => setPageNo(target.page)}>
            Back to {target.label}
          </button>
        )}
      </div>
      <div className="viewer__scroll" ref={scrollRef} tabIndex={0} aria-label={`Page ${pageNo} of ${target.filename}`}>
        {error ? (
          <p className="error">{error}</p>
        ) : (
          <div className="viewer__page" style={size ? { width: size.w, height: size.h } : undefined}>
            <canvas ref={canvasRef} className={loading ? "viewer__canvas viewer__canvas--loading" : "viewer__canvas"} />
            {size && onTargetPage && (
              <div className="viewer__marks" aria-hidden>
                {target.boxes.map(([x0, y0, x1, y1], i) => (
                  <div
                    key={i}
                    ref={i === 0 ? markRef : undefined}
                    className="viewer__mark"
                    style={{
                      left: x0 * size.k - 2,
                      top: y0 * size.k - 2,
                      width: (x1 - x0) * size.k + 4,
                      height: (y1 - y0) * size.k + 4,
                      animationDelay: `${Math.min(i, 12) * 40}ms`,
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        )}
        {loading && !error && (
          <div className="bar bar--indeterminate viewer__loading" aria-label="Loading the page">
            <span />
          </div>
        )}
      </div>
    </div>
  );
}

// A DOCX that could not be converted to PDF: show its paragraphs and mark the cited one.
function ParagraphView({ target }: { target: ViewerTarget }) {
  const [paras, setParas] = useState<string[] | null>(null);
  const markRef = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    getPages(target.docId)
      .then((pages) => setParas((pages[0]?.text ?? "").split("\n\n")))
      .catch(() => setParas([]));
  }, [target.docId]);
  useEffect(() => markRef.current?.scrollIntoView({ block: "center" }), [paras]);
  if (!paras) {
    return (
      <div className="bar bar--indeterminate" aria-label="Loading">
        <span />
      </div>
    );
  }
  return (
    <div className="viewer__scroll viewer__paras" tabIndex={0}>
      {paras.map((p, i) => (
        <p key={i} ref={i === target.paraIdx ? markRef : undefined} className={i === target.paraIdx ? "para para--mark" : "para"}>
          <span className="para__no">¶ {i + 1}</span>
          {p}
        </p>
      ))}
    </div>
  );
}

export default function PdfViewer({ target, onClose }: { target: ViewerTarget; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const returnFocus = useRef<Element | null>(null);

  // Move focus into the dialog, and back to what opened it on close.
  useEffect(() => {
    returnFocus.current = document.activeElement;
    panelRef.current?.focus();
    return () => (returnFocus.current as HTMLElement | null)?.focus?.();
  }, []);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      onClose();
      return;
    }
    if (e.key !== "Tab") return;
    const focusable = panelRef.current?.querySelectorAll<HTMLElement>(
      'button:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])',
    );
    if (!focusable || focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  return (
    <div className="viewer" onKeyDown={onKeyDown}>
      <div className="viewer__backdrop" onClick={onClose} aria-hidden />
      <div
        ref={panelRef}
        className="viewer__panel"
        role="dialog"
        aria-modal="true"
        aria-label={`${target.filename}, ${target.label}`}
        tabIndex={-1}
      >
        <header className="viewer__head">
          <div className="viewer__title">
            <span className="stamp viewer__stamp" aria-hidden>
              {target.label}
            </span>
            <span className="viewer__file" title={target.filename}>
              {target.filename}
            </span>
          </div>
          <div className="viewer__actions">
            {target.anchor === "page" && (
              <a className="btn-text" href={`${fileUrl(target.docId)}#page=${target.page}`} target="_blank" rel="noreferrer">
                Original
              </a>
            )}
            <button type="button" className="viewer__close" onClick={onClose} aria-label="Close the viewer">
              ×
            </button>
          </div>
        </header>
        {target.checkPage && (
          <p className="viewer__check">
            <span className="check-badge">Check the page</span>
            This page was read from a scan
            {target.ocrConf != null && ` (OCR confidence ${percent(target.ocrConf)})`}: compare names and figures with
            the image.
          </p>
        )}
        {target.citedText && (
          <blockquote className="viewer__quote">
            {target.citedText}
          </blockquote>
        )}
        {target.anchor === "paragraph" ? (
          <ParagraphView target={target} />
        ) : (
          <PdfPage target={target} />
        )}
      </div>
    </div>
  );
}
