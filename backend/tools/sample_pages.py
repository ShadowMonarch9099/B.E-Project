"""Cut a few pages out of a large PDF so it can be uploaded to Sakshya (50 MB limit).

    python tools/sample_pages.py big.pdf                       # first 3, middle 2, last 2 pages
    python tools/sample_pages.py big.pdf --pages 1-3,120,400-402
    python tools/sample_pages.py big.pdf --pages 5-9 -o sample.pdf

Run it from the backend folder with the backend's Python (.venv\\Scripts\\python on Windows).
"""

import argparse
from pathlib import Path

import pymupdf


def parse_pages(spec: str, count: int) -> list[int]:
    """'1-3,50' -> [0, 1, 2, 49] (0-based), ignoring pages past the end."""
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start, _, end = part.partition("-")
        for n in range(int(start), int(end or start) + 1):
            if 1 <= n <= count and n - 1 not in pages:
                pages.append(n - 1)
    return pages


def default_pages(count: int) -> list[int]:
    mid = count // 2
    picks = [0, 1, 2, mid - 1, mid, count - 2, count - 1]
    return sorted({p for p in picks if 0 <= p < count})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", help="pages to keep, e.g. 1-3,50,120-121 (1-based)")
    ap.add_argument("-o", "--out", type=Path, help="output file (default: <name>_sample.pdf)")
    args = ap.parse_args()

    pymupdf.TOOLS.mupdf_display_errors(False)
    with pymupdf.open(args.pdf) as src:
        total = src.page_count
        pages = parse_pages(args.pages, src.page_count) if args.pages else default_pages(src.page_count)
        if not pages:
            raise SystemExit(f"No valid pages; the file has {src.page_count}.")
        out = args.out or args.pdf.with_name(f"{args.pdf.stem}_sample.pdf")
        with pymupdf.open() as dst:
            for p in pages:
                dst.insert_pdf(src, from_page=p, to_page=p)
            dst.save(out, garbage=3, deflate=True)
    size = out.stat().st_size / 1e6
    print(f"Wrote {out} with pages {', '.join(str(p + 1) for p in pages)} of {total} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
