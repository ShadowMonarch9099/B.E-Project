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

## The app (Sakshya v0.8)

What works now, by plan section:

- **Type detection** (§6.1 step 1): any file is identified from its content (Google Magika), not its name.
- **Stored documents** (§6.2, §6.8): PDF, DOCX and scans (PNG, JPEG, TIFF) are stored under `data/uploads/<document id>/` (gitignored), keyed by their SHA-256 so a second upload of the same file is recognised. Documents, pages and their text live in PostgreSQL 16 + pgvector and survive a restart of both servers.
- **Background ingestion** (§6.2 steps 1–3): every page is read, from the text layer when it is usable and with OCR otherwise. Status moves *queued → processing → indexed* (or *failed*), and each card shows "Indexing… page 12 of 40". Unfinished documents are picked up again when the backend restarts.
- **OCR** (§6.2 step 2): Tesseract (Marathi, Hindi, English) at 300 DPI on every page, with per-word confidence and boxes converted to PDF points. Low-confidence pages get a "Check the page" badge.
- **Text layers that cannot be trusted** go to OCR instead: Devanagari typed in legacy fonts (Kruti Dev, DVB-TT Surekh, Shivaji, …: "भारत" stored as "Hkkjr"), and Unicode Devanagari whose conjuncts were written as stray characters ("राष्ट्रीय" stored as "राʊȄीय"). The second kind was found in the team's sample GRs.
- **DOCX** (§6.2 step 3) is converted to PDF with LibreOffice headless so citations point to real pages. Without LibreOffice, a DOCX is read paragraph by paragraph and its card says so.
- **Page-anchored chunks** (§6.2 steps 5–7): each page is split on paragraph and sentence boundaries into passages of about 200–350 tokens (counted with multilingual-E5's own tokenizer), with a one-sentence overlap inside the page. **A chunk never crosses a page**: a paragraph that runs onto the next page becomes two chunks, each citing its own page. Every chunk stores its page, paragraph and position, the boxes of its lines (PDF points, from PyMuPDF words or from Tesseract word boxes), its source (native / ocr / docx), mean OCR confidence and script. The sentence splitter knows common abbreviations ("Survey No. 142", "क्र. १४२", "दि.") so it does not cut there.
- **Normalization** (§6.2 step 5): the IndicNLP normalizer for Devanagari, Unicode NFC, and Devanagari digits mapped to ASCII (१४२ → 142) in the indexed text; the original text is kept for display.
- **Language of each passage** (§6.2 step 4): IndicLID labels every chunk English, Hindi or Marathi and keeps both Hindi and Marathi scores (they share a script and are often close on short or OCR'd text); each page takes the most common language of its chunks, and each card shows the document's language mix.
- **Indexes over the chunks** (§6.2 step 8):
  - *Lexical*: BM25 (the `bm25s` library) over IndicNLP word tokens, and a second BM25 over character 3–5-grams of each word, which still match OCR-damaged or inflected words. Built in memory from the database and rebuilt when the indexed chunks change. PostgreSQL full-text search is not used, because it has no Marathi configuration.
  - *Dense*: [multilingual-E5-base](https://huggingface.co/intfloat/multilingual-e5-base) with its `passage: ` / `query: ` prefixes, vectors in pgvector with an HNSW index (cosine). `SAKSHYA_EMBED_MODEL=intfloat/multilingual-e5-large` switches to the large model (a budget variable in the plan); each model keeps its own vectors and HNSW index.
  - *English shadow* (for the translate route): every Hindi or Marathi chunk is translated to English (IndicTrans2, or NLLB-200 until IndicTrans2 is available) and embedded with the same E5 model; English and Latin-script chunks are copied as they are. It is built right after a document becomes searchable, because translation is the slowest step on a CPU, and its progress shows on the card. `SAKSHYA_SHADOW_INDEX=off` skips it.
  - Documents indexed before a stage existed are brought up to date in the background at startup (language, vectors for the current model, English index).

### Asking questions

Pick an indexed document in the ask bar at the bottom of the page and ask in English, Hindi, Marathi or romanized text. For each question:

1. **Understand the question** (§6.3, N2): IndicLID tells the language and whether a Latin-script question is English or romanized Hindi/Marathi; IndicXlit writes a romanized question in Devanagari ("RUSA yojanet kendra ani rajya hissa kiti aahe?" → "RUSA योजनेत केंद्र अनी राज्य हिस्सा किती आहे?"); a Hindi or Marathi question is also translated to English ("How much is the Union State's share in RUSA?"); numbers, dates and IDs are pulled out with Devanagari digits mapped to ASCII (१४२ → 142) for exact matching. These *views* and the code-mix ratio are stored in `query_logs`.
2. **Choose a route and retrieve** (§6.4, §6.5). Three routes exist: *direct* (E5 over the chunks as written, with every view), *translate* (the question in English against the English shadow index; it falls back to direct when a document has none) and *hybrid* (E5 once per view, BM25 over words and over character n-grams with all views, and, when the question has numbers or IDs, the chunks that contain them as whole tokens; 3,00,000 and 300000 count as the same number; fused with weighted reciprocal rank fusion: dense 1.0, words 1.0, characters 0.6, exact 0.8). Each returns its top 40 within the chosen document.
   **The rule-based router** (a stand-in for the plan's learned router N1) chooses the route from the cheap features the plan lists: the question's script, language and IndicLID confidence, romanized flag, code-mix ratio, length, numbers/IDs and question type; the searched documents' language mix, share of scanned pages and OCR confidence; the OCR confidence of the top dense hits; and whether the English index is ready. Ordered rules: numbers or IDs → hybrid; romanized or code-mixed → hybrid; best matches on low-confidence scans → hybrid; question and documents in different languages → translate; same language, no IDs → direct; otherwise hybrid. Every route's top three candidates are also reranked (the *probe*); if another route's evidence beats the rule's choice by 0.30 or more, that route is used instead. Under each answer, **"Why this route"** shows the rule in words, the views, the features and each route's probe score. All of it goes to `query_logs` as training data for a learned router. A route can be forced with `{"route": "translate"}` in the message body (for evaluation and demos).
   **OCR-aware fusion** (N3, §6.6): in the hybrid route, a chunk from a page below 85% OCR confidence gets up to 2.5× the weight from the character n-gram list and down to half from the word list.
3. **Rerank** with the pretrained multilingual cross-encoder [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) (used as-is; each passage, as written, is scored against the question as typed and its Devanagari view, keeping the higher score) and keep the top 6. Every route uses the same reranker, so the routes differ only in the candidates they find.
4. **Evidence gate** (§6.7 step 1, N4): if the best reranker score is below **τ = 0.15** (`SAKSHYA_GATE_TAU`), Sakshya **abstains without calling the answer model** and shows the nearest pages.
5. **Answer** with the Claude API (`claude-opus-5-5`, effort `medium`): each of the 6 passages is sent as its own document block with the API's citations feature, so every cited sentence carries a stamp-style chip ("p. 4"), with a dashed *check* mark when the page was a low-confidence scan. Answers stream to the browser over Server-Sent Events. Follow-up questions see the earlier turns (and their passages), which are replayed append-only so the API's prompt cache reuses them. Server-side refusal fallbacks are switched on (`fallbacks: "default"`), so if the model's safeguards decline a question, the API retries it on the model Anthropic recommends; a refusal that survives is shown as a polite "cannot answer", and an answer that hits the token limit is marked as cut short.

**Opening the evidence** (§6.8): a citation chip, a nearest page in an abstention, or a document's *Open* button opens the original page in a side panel (full screen on phones) with the cited passage highlighted. The page is drawn with [pdf.js](https://mozilla.github.io/pdf.js/) (`pdfjs-dist`), and the highlight boxes are the chunk's line boxes in PDF points (from PyMuPDF words on native pages, from Tesseract word boxes on scanned, legacy-font and broken-text pages, and from the converted PDF for DOCX and images). A DOCX that could not be converted shows its paragraphs with the cited one marked. Keyboard: `Esc` closes, `←`/`→` change page, `+`/`−` zoom; focus moves into the panel and back to the chip on close. Pages from low-confidence scans carry a "Check the page" note.

Every question is written to the `query_logs` table: the question, its views and features, the route and the router's reason, every route's probe scores, the candidate lists with each source's rank and score, reranker scores, the gate decision, whether the model was called, the model's usage, and the latency of each stage. These rows are the training data for a future learned router and the raw material for the evaluation.

**How τ was chosen (a starting value, not a result).** `backend/tools/gate_probe.py` runs 24 hand-written questions over two of the sample GRs (`backend/tools/gate_probe_questions.json`). With the question as typed only (phase 4), the 14 answerable questions' best reranker scores were 0.077–1.000 and the 10 unanswerable ones' 0.002–0.063. With the phase 6 views, romanized answerable questions rose sharply ("RUSA yojanet kendra ani rajya hissa kiti aahe?" 0.077 → 0.992, "Model Degree College Hingoli la kiti anudan milale?" 0.586 → 0.984), but so did some romanized off-topic ones: "RUSA 3.0 madhe kiti nidhi aahe?" (the GR covers RUSA 1.0 and 2.0) went from 0.063 to 0.401. Answerable now span 0.171–1.000 and unanswerable 0.002–0.401, so no threshold separates them; τ = 0.15 answers 14 of 14 and abstains on 9 of 10, and the near-miss is left to the answer model's "the documents do not say" rule. τ will be re-chosen from the risk–coverage curve on the Sakshya-Bench dev split.

**The API key.** Answers need an Anthropic API key: copy `backend/.env.example` to `backend/.env`, set `ANTHROPIC_API_KEY=...`, and restart the backend. `backend/.env` is gitignored; the key is read by the SDK on the server and never sent to the browser or logged. Without a key everything else works (upload, OCR, search, abstention, the viewer), and an answered question explains how to add the key. **Privacy:** for each question that passes the evidence gate, the question and its top 6 passages are sent to the Anthropic API (decision log, `docs/Review2_Plan.md` §16); nothing else leaves the machine.

**Cost (estimate, not yet measured).** About 3,000–4,000 input tokens (6 passages + instructions) and a few hundred output tokens per first question: roughly US$0.02–0.04 at Claude Opus 5.5's $4 / $20 per million tokens. Follow-ups re-read the earlier turns from the prompt cache at $0.20 per million tokens. Abstained questions cost nothing. The real numbers come from `query_logs.llm.usage` once a key is set.

**Latency on this laptop** (x64 Python under emulation on a Snapdragon X Elite, models loaded): retrieval takes about 0.7 s for an English question and 1.2–2.7 s for a Marathi or romanized one (IndicXlit about 0.5 s, translation about 0.7 s, reranking 0.6–1.1 s). The models load in the background when the backend starts (about a minute; `/api/health` → `models`). An earlier version set `OMP_THREAD_LIMIT=1` for the whole process to keep parallel Tesseract runs single-threaded, which also capped PyTorch and CTranslate2 at one thread and made retrieval about five times slower; the limit now applies to Tesseract only.

**Models** (all used as-is; see `docs/prebuilt_models.md` and the decision log in `docs/Review2_Plan.md` §16): Google Magika (type detection), Tesseract (OCR), multilingual-E5-base (dense retrieval), mmarco-mMiniLMv2 (reranking), [AI4Bharat IndicLID](https://github.com/AI4Bharat/IndicLID) FastText models (language ID; 169 + 229 MB, downloaded on first use to `data/models/indiclid`), [AI4Bharat IndicXlit](https://huggingface.co/ai4bharat/IndicXlit) (transliteration; run in plain PyTorch because the fairseq package does not install on Python 3.11 / Windows), and for translation [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) when `HF_TOKEN` is set for an account that accepted its terms, otherwise [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC 4.0), converted to CTranslate2 int8 on first use (2.5 GB download). Nothing is trained.

| Folder | What it is |
| --- | --- |
| `backend/` | FastAPI service (endpoints below); Alembic migrations in `backend/migrations`; `backend/tools/` small command-line tools |
| `frontend/` | Next.js + TypeScript site: upload, document cards, the conversation thread, the PDF viewer |
| `data/` | Uploaded originals and downloaded models (created at runtime, gitignored) |

| Endpoint | What it does |
| --- | --- |
| `GET /api/health` | Detector, OCR, database, DOCX converter and model status |
| `POST /api/detect` | Detect file types (multipart `files`); nothing is stored |
| `POST /api/ocr` | OCR one PDF or image (multipart `file`, first 10 pages); nothing is stored |
| `POST /api/documents` | Store a PDF, DOCX or image (multipart `file`) and start ingesting it: `201` new, `200` duplicate, `422` detected but not ingestible |
| `GET /api/documents` | All documents with status and progress |
| `GET /api/documents/{id}` | One document: status, facts, page list |
| `GET /api/documents/{id}/pages` | Every page's text, source and OCR confidence |
| `GET /api/documents/{id}/file` | The PDF the viewer opens (`?original=true` for the uploaded file) |
| `POST /api/documents/{id}/reindex` | Detect and ingest the stored file again |
| `DELETE /api/documents/{id}` | Delete the document, its pages and its files |
| `GET /api/search?q=…&doc_id=…&k=10` | Top hits from each index (BM25 words, BM25 character n-grams, E5 dense), for checking retrieval |
| `GET /api/search/route?q=…&doc_id=…&route=…` | One route (`direct`, `translate` or `hybrid`) end to end without answering: views, candidates, reranking, gate |
| `POST /api/conversations` | Start a conversation about a document (`{"doc_id": …}`) |
| `GET /api/conversations?doc_id=…` · `GET /api/conversations/{id}` | List conversations; one conversation with its messages |
| `POST /api/conversations/{id}/messages` | Ask (`{"question": …}`, optionally `"route"`); the answer streams back as Server-Sent Events (`status`, `retrieval`, `delta`, `answer`, `done`) |

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
