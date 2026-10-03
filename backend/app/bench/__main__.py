"""Sakshya-Bench command line (plan section 8). Run from backend/:

    .venv/bin/python -m app.bench fetch            # download the manifest's documents to data/bench/docs
    .venv/bin/python -m app.bench upload           # add the fetched documents to the running app
    .venv/bin/python -m app.bench facts            # record each indexed document's language mix and OCR share
    .venv/bin/python -m app.bench splits           # assign document splits (50/15/35, by language and scan)
    .venv/bin/python -m app.bench freeze           # freeze the splits before any tuning
    .venv/bin/python -m app.bench export out.jsonl # verified questions with gold pages and splits
    .venv/bin/python -m app.bench stats            # counts, agreement (kappa, token-F1), manifest summary

`fetch` records each document's SHA-256, size, page count and text layer (native, scanned,
mixed, legacy, broken) in the manifest. `facts`, run once the documents are indexed, replaces
the languages expected from the listing page with the mix IndicLID found, and records the
share of pages read by OCR; a document counts as scanned when at least half its pages were.
The documents themselves never go into git.
"""

import argparse
import hashlib
import json
import sys
import urllib.request

from . import manifest as man

USER_AGENT = "Sakshya-Bench fetcher (B.E. project, MMCOE Pune)"


def fetch(args) -> None:
    from ..detection import analyse

    m = man.load()
    man.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for d in m["documents"]:
        if args.only and d["id"] not in args.only:
            continue
        ext = d.get("ext", "pdf")
        path = man.DOCS_DIR / f"{d['id']}.{ext}"
        if not path.exists() or args.force:
            print(f"fetch  {d['id']}  {d['url']}", flush=True)
            try:
                req = urllib.request.Request(d["url"], headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
            except Exception as exc:
                d["fetch_error"] = f"{type(exc).__name__}: {exc}"[:200]
                print(f"  failed: {d['fetch_error']}")
                continue
            path.write_bytes(data)
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if d.get("sha256") and d["sha256"] != sha:
            print(f"  WARNING {d['id']}: the file changed at the source (sha256 differs from the manifest)")
        det = analyse(data, path.name)
        facts = {f["label"]: f["value"] for f in det["facts"]}
        d.update(sha256=sha, size_bytes=len(data), pages=int(facts.get("Pages", "0") or 0) or None,
                 text_layer=det.get("pdf_text_layer"), detected_type=det["type"]["label"])
        if d.get("scanned") is None and det.get("pdf_text_layer"):
            d["scanned"] = det["pdf_text_layer"] == "scanned"
        d.pop("fetch_error", None)
        print(f"  ok  {len(data) // 1024} KB, {d['pages']} pages, text layer {d['text_layer']}")
    man.save(m)


def upload(args) -> None:
    import mimetypes
    import uuid

    m = man.load()
    for d in m["documents"]:
        path = man.DOCS_DIR / f"{d['id']}.{d.get('ext', 'pdf')}"
        if not path.exists():
            continue
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
                f"Content-Type: {mimetypes.guess_type(path.name)[0] or 'application/pdf'}\r\n\r\n").encode() + \
            path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{args.api}/api/documents", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                res = json.loads(r.read())
            print(f"{d['id']}: {'already there' if res.get('duplicate') else 'added'} ({res['document']['status']})")
        except urllib.error.HTTPError as exc:
            print(f"{d['id']}: {exc.code} {exc.read()[:200]!r}")


SCANNED_SHARE = 0.5  # a document whose pages were at least half OCR'd counts as scanned


def facts(args) -> None:
    """Replace the languages expected from the listing page with the mix that language ID
    found in the indexed chunks (any language holding at least 10% of the tokens), and record
    the share of pages read by OCR (scanned, legacy-font or broken text layers)."""
    from sqlalchemy import func, select

    from ..db import Chunk, Document, Page, init_db, session_scope

    if not init_db():
        sys.exit("database not reachable")
    m = man.load()
    with session_scope() as s:
        for d in m["documents"]:
            if not d.get("sha256"):
                continue
            doc = s.scalar(select(Document).where(Document.sha256 == d["sha256"]))
            if doc is None or doc.status != "indexed":
                print(f"{d['id']}: not indexed yet")
                continue
            rows = s.execute(select(Chunk.lang, func.sum(Chunk.token_count)).where(Chunk.doc_id == doc.id)
                             .group_by(Chunk.lang)).all()
            total = sum(n or 0 for _, n in rows)
            if not total:
                print(f"{d['id']}: no text")
                continue
            mix = {lang or "unknown": round((n or 0) / total, 3) for lang, n in rows}
            found = [lang for lang, share in sorted(mix.items(), key=lambda x: -x[1]) if share >= 0.10]
            if found != d.get("languages"):
                print(f"{d['id']}: {d.get('languages')} -> {found}  {mix}")
            sources = dict(s.execute(select(Page.source, func.count(Page.id)).where(Page.doc_id == doc.id)
                                     .group_by(Page.source)).all())
            ocr_share = round(sources.get("ocr", 0) / max(1, sum(sources.values())), 3)
            d.update(languages=found, language_mix=mix,
                     languages_source="IndicLID over the indexed chunks (share of tokens, at least 10%)",
                     ocr_share=ocr_share, scanned=ocr_share >= SCANNED_SHARE)
    man.save(m)


def splits(args) -> None:
    m = man.assign_splits(man.load())
    man.check_frozen(m)
    man.save(m)
    print(json.dumps(man.summary(m), indent=2))


def freeze(args) -> None:
    m = man.freeze(man.load())
    man.save(m)
    print(f"Frozen at {m['frozen']['at']}: {len(m['frozen']['test'])} test documents, hash {m['frozen']['hash']}")


def export(args) -> None:
    from ..db import init_db
    from . import store

    if not init_db():
        sys.exit("database not reachable")
    text = store.jsonl(args.include_drafts)
    with open(args.out, "w", encoding="utf8") as f:
        f.write(text)
    print(f"{text.count(chr(10))} questions written to {args.out}")


def stats(args) -> None:
    from ..db import init_db
    from . import store

    out = {"manifest": man.summary(man.load())}
    if init_db():
        out["questions"] = store.stats()
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m app.bench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--only", nargs="*")
    f.add_argument("--force", action="store_true")
    u = sub.add_parser("upload")
    u.add_argument("--api", default="http://127.0.0.1:8000")
    sub.add_parser("facts")
    sub.add_parser("splits")
    sub.add_parser("freeze")
    e = sub.add_parser("export")
    e.add_argument("out")
    e.add_argument("--include-drafts", action="store_true")
    sub.add_parser("stats")
    args = ap.parse_args()
    {"fetch": fetch, "upload": upload, "facts": facts, "splits": splits, "freeze": freeze, "export": export, "stats": stats}[args.cmd](args)


if __name__ == "__main__":
    main()
