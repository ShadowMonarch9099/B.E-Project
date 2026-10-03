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

## The app (Sakshya v0.3)

What works now, by plan section:

- **Type detection** (§6.1 step 1): any file is identified from its content (Google Magika), not its name.
- **Stored documents** (§6.2, §6.8): PDF, DOCX and scans (PNG, JPEG, TIFF) are stored under `data/uploads/<document id>/` (gitignored), keyed by their SHA-256 so a second upload of the same file is recognised. Documents, pages and their text live in PostgreSQL 16 + pgvector and survive a restart of both servers.
- **Background ingestion** (§6.2 steps 1–3): every page is read, from the text layer when it is usable and with OCR otherwise. Status moves *queued → processing → indexed* (or *failed*), and each card shows "Indexing… page 12 of 40". Unfinished documents are picked up again when the backend restarts.
- **OCR** (§6.2 step 2): Tesseract (Marathi, Hindi, English) at 300 DPI on every page, with per-word confidence and boxes converted to PDF points. Low-confidence pages get a "Check the page" badge.
- **Text layers that cannot be trusted** go to OCR instead: Devanagari typed in legacy fonts (Kruti Dev, DVB-TT Surekh, Shivaji, …: "भारत" stored as "Hkkjr"), and Unicode Devanagari whose conjuncts were written as stray characters ("राष्ट्रीय" stored as "राʊȄीय"). The second kind was found in the team's sample GRs.
- **DOCX** (§6.2 step 3) is converted to PDF with LibreOffice headless so citations point to real pages. Without LibreOffice, a DOCX is read paragraph by paragraph and its card says so.

No ML is trained here: type detection uses the prebuilt [Google Magika](https://github.com/google/magika) model and OCR uses [Tesseract](https://github.com/tesseract-ocr/tesseract). See `docs/prebuilt_models.md`.

| Folder | What it is |
| --- | --- |
| `backend/` | FastAPI service (endpoints below); Alembic migrations in `backend/migrations` |
| `frontend/` | Next.js + TypeScript site: upload, document cards with ingestion progress, extracted pages |
| `data/` | Uploaded originals (created at runtime, gitignored) |

| Endpoint | What it does |
| --- | --- |
| `GET /api/health` | Detector, OCR, database and DOCX-converter status |
| `POST /api/detect` | Detect file types (multipart `files`); nothing is stored |
| `POST /api/ocr` | OCR one PDF or image (multipart `file`, first 10 pages); nothing is stored |
| `POST /api/documents` | Store a PDF, DOCX or image (multipart `file`) and start ingesting it: `201` new, `200` duplicate, `422` detected but not ingestible |
| `GET /api/documents` | All documents with status and progress |
| `GET /api/documents/{id}` | One document: status, facts, page list |
| `GET /api/documents/{id}/pages` | Every page's text, source and OCR confidence |
| `GET /api/documents/{id}/file` | The PDF the viewer opens (`?original=true` for the uploaded file) |
| `POST /api/documents/{id}/reindex` | Detect and ingest the stored file again |
| `DELETE /api/documents/{id}` | Delete the document, its pages and its files |

### Run it locally

You need **Python 3.11+**, **Node 20+**, **Docker** (for the database) and **Tesseract**.

**1. Tesseract** (OCR; type detection works without it):

- Windows: run the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki). On the components page, under *Additional language data*, tick **Hindi** and **Marathi**. The backend finds the default install path (`C:\Program Files\Tesseract-OCR\tesseract.exe`) by itself; for another location set `SAKSHYA_TESSERACT_CMD` to the full path of `tesseract.exe`.
- Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar`
- macOS: `brew install tesseract tesseract-lang`

**2. LibreOffice** (optional, for DOCX page citations): Windows `winget install -e --id TheDocumentFoundation.LibreOffice`, Ubuntu `sudo apt install libreoffice-writer`, macOS `brew install --cask libreoffice`.

**3. The database** (PostgreSQL 16 + pgvector, on port **5433** so it does not clash with a PostgreSQL you may already run on 5432):

```bash
docker compose up -d db
```

**4. Backend and frontend:**

```bash
# Backend (Python 3.11+)
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt      # Windows: .venv\Scripts\pip
cp .env.example .env                               # optional: settings (see the file)
.venv/bin/uvicorn app.main:app --port 8000         # Windows: .venv\Scripts\uvicorn

# Frontend (Node 20+), in a second terminal
cd frontend
npm install
npm run dev                                         # open http://localhost:3000
```

The backend creates and upgrades its tables on startup. If the database is not running, the site still detects and OCRs files, and says that documents cannot be stored. The frontend forwards `/api/*` to `BACKEND_URL` (default `http://localhost:8000`). Or run everything with `docker compose up --build`.

Settings are environment variables, listed with their defaults in `backend/.env.example`. The main ones: `SAKSHYA_DATABASE_URL`, `SAKSHYA_DATA_DIR`, `SAKSHYA_MAX_FILE_MB`, `SAKSHYA_OCR_LANGS` (default `mar+hin+eng`), `SAKSHYA_TESSERACT_CMD`, `SAKSHYA_LIBREOFFICE_CMD`.

**Tests:** `cd backend && .venv/bin/python -m pytest -q`. They use a throwaway SQLite database, so they need no server. To run them against PostgreSQL, create an empty database whose name ends in `_test` and set `SAKSHYA_TEST_DATABASE_URL`, e.g. `postgresql+psycopg://sakshya:sakshya@localhost:5433/sakshya_test`.

**Upload limit:** 200 MB per file by default. To change it, set `SAKSHYA_MAX_FILE_MB` in **both** terminals before starting the backend and the frontend, because the frontend's forwarding limit uses the same value:

```powershell
$env:SAKSHYA_MAX_FILE_MB=500     # Windows PowerShell; on macOS/Linux: export SAKSHYA_MAX_FILE_MB=500
```

**Testing with a PDF over the limit:** cut a few pages out first:

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
