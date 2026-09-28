# B.E. Final Year Project: Multilingual Document Intelligence (RAG)

Final year B.E. project, Department of Artificial Intelligence & Data Science, MMCOE Pune (SPPU).

A citation-grounded Retrieval-Augmented Generation system for private English, Hindi and Marathi document collections (native PDFs, DOCX and scanned documents), with page-level citations and abstention when evidence is insufficient.

**Team:** Tanmay Gosavi, Kush Honkalse
**Guide:** Mr. Rishikesh C. Yeolekar

## Contents

| File | Description |
| --- | --- |
| `Abstract.pdf` | Project abstract |
| `Multilingual_RAG_Project_Synopsis_Enhanced.pdf` | Project synopsis (current) |
| `Multilingual_RAG_Project_Synopsis.pdf` | Project synopsis (earlier draft) |
| `Multilingual_RAG_Research_Gaps_Enhanced.pdf` | Literature review and research gaps (current) |
| `Multilingual_RAG_Research_Gaps.pdf` | Literature review and research gaps (earlier draft) |
| `RAG_Project_Detailed_Plan_v2.pdf` | Detailed implementation plan |
| `BE_RAG_Review1_Redesigned.pptx` | Stage I (Review 1) presentation |
| `BE_RAG_Review2.pptx` | Review 2 presentation (Sakshya), rebuilt from the Review 1 deck using the guide in `docs/Review2_Plan.md` §14 |
| `create_rag_project_pdfs.ps1` | Script that generates the synopsis and research-gap PDFs |
| `docs/Review2_Plan.md` | **Current project plan (Sakshya) for Review 2.** Supersedes the plan v2 PDF and Review 1 deck where they disagree |
| `docs/prebuilt_models.md` | Prebuilt models for detection and extraction, mapped to the plan's stages |
| `docs/sakshya.html` | Review of the plan and the "Sakshya" counter-proposal (open in a browser) |

## The app (Sakshya v0.2: type detection and OCR)

The first slices of the ingestion pipeline (plan §6.1 steps 1-2):

- **Type detection:** upload any file and see what type of document it is, detected from its content rather than its name, plus how Sakshya v1 would ingest it (parse, OCR, or detect only).
- **OCR:** scanned PDFs and images are read automatically with Tesseract (Marathi, Hindi, English). Each page shows its text, its mean word confidence, and a "Check the page" badge when the scan is hard to read. PDF pages that already have a text layer are read directly.
- **Legacy fonts:** Devanagari typed in pre-Unicode fonts (Kruti Dev, DVB-TT Surekh, Shivaji, ...) has a text layer that is nonsense ("भारत" stored as "Hkkjr"). Sakshya detects it and OCRs those pages instead; Word files in such fonts get a warning.

No ML is trained here: type detection uses the prebuilt [Google Magika](https://github.com/google/magika) model and OCR uses [Tesseract](https://github.com/tesseract-ocr/tesseract). See `docs/prebuilt_models.md`.

| Folder | What it is |
| --- | --- |
| `backend/` | FastAPI service: `POST /api/detect` (multipart `files`), `POST /api/ocr` (multipart `file`), `GET /api/health`. Files are read in memory and never stored |
| `frontend/` | Next.js + TypeScript site: drag-and-drop upload, per-file results, monochrome theme |

### Run it locally

**Install Tesseract first** (needed for OCR; type detection works without it):

- Windows: run the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki). On the components page, under *Additional language data*, tick **Hindi** and **Marathi**. The backend finds the default install path (`C:\Program Files\Tesseract-OCR\tesseract.exe`) by itself; for another location set `SAKSHYA_TESSERACT_CMD` to the full path of `tesseract.exe`.
- Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar`
- macOS: `brew install tesseract tesseract-lang`

The status line above the upload box says whether OCR is ready.

```bash
# 1. Backend (Python 3.11+)
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt      # Windows: .venv\Scripts\pip
.venv/bin/uvicorn app.main:app --port 8000

# 2. Frontend (Node 20+), in a second terminal
cd frontend
npm install
npm run dev                                         # open http://localhost:3000
```

The frontend forwards `/api/*` to `BACKEND_URL` (default `http://localhost:8000`). Or run both with `docker compose up --build`.

OCR settings (environment variables for the backend): `SAKSHYA_OCR_LANGS` (default `mar+hin+eng`), `SAKSHYA_OCR_MAX_PAGES` (default `10` pages per file), `SAKSHYA_TESSERACT_CMD` (path to the Tesseract binary).

Tests: `cd backend && .venv/bin/python -m pytest`.

**Testing with a large PDF:** uploads are limited to 50 MB, so cut a few pages out first:

```bash
cd backend
.venv/bin/python tools/sample_pages.py big.pdf                    # first 3, middle 2 and last 2 pages
.venv/bin/python tools/sample_pages.py big.pdf --pages 1-3,120    # or choose pages
```

## Research questions

1. Which retrieval strategy (direct multilingual, translation-based, hybrid dense + BM25) gives the best quality and cost for each language and document type?
2. Does hybrid retrieval with cross-encoder reranking improve page-cited answer faithfulness?
3. Can evidence-sufficiency checks reduce unsupported answers without unacceptable latency?
4. How consistently does the system answer in the requested language?

## Planned stack

FastAPI, Next.js, PostgreSQL + pgvector (HNSW), multilingual-E5, BM25 (IndicNLP tokens + character n-grams), fine-tuned multilingual cross-encoder reranker, Tesseract OCR (hin, mar, eng), AI4Bharat IndicLID / IndicXlit / IndicTrans2, a small local 4-bit instruct LLM, and a multilingual NLI verifier. See `docs/Review2_Plan.md` for details.
