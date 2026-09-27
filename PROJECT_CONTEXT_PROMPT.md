# Project Context Prompt: Multilingual Document Intelligence (RAG) → "Sakshya"

> Paste this whole document into a new AI session to give it full context on our B.E. final year project, the analysis already done, the proposed solution, the artifact built, and the repository state.

---

## 1. Who we are and what you are helping with

You are helping a two-person B.E. final year team (Department of Artificial Intelligence & Data Science, Marathwada Mitra Mandal's College of Engineering (MMCOE), Pune, SPPU).

- **Team:** Tanmay Gosavi (BA18), Kush Honkalse (BA20)
- **Guide:** Mr. Rishikesh C. Yeolekar
- **Project title:** Multilingual Document Intelligence (RAG): a Resource-Efficient, Citation-Grounded Multilingual Document Intelligence system for private English and Indian-language document archives
- **Status:** Stage I (Review 1) done. Implementation has not started. Review 2 (implementation) and Review 3 (testing and demo) are next.
- **Hardware budget:** Google Colab T4 GPU and CPU laptops. The team is two people with roughly 12 weeks of build time.

In an earlier session, an AI assistant read all the project documents, critiqued the plan, designed an improved solution called **Sakshya**, published it as an interactive web page (the "artifact"), and set up a Git repository. This prompt records everything from that session.

---

## 2. Project documents (in the repo root)

| File | What it is |
| --- | --- |
| `Abstract.pdf` | Project abstract (the most mature framing) |
| `Multilingual_RAG_Project_Synopsis_Enhanced.pdf` | Current synopsis: aim, objectives, methodology, evaluation, scope |
| `Multilingual_RAG_Project_Synopsis.pdf` | Earlier synopsis draft |
| `Multilingual_RAG_Research_Gaps_Enhanced.pdf` | Literature review of 7 papers, consolidated gap, RQ1–RQ4 |
| `Multilingual_RAG_Research_Gaps.pdf` | Earlier gaps draft |
| `RAG_Project_Detailed_Plan_v2.pdf` | Detailed 12-week plan (SaaS-oriented, Claude API, ML tracks, future roadmap) |
| `BE_RAG_Review1_Redesigned.pptx` | Review 1 presentation (11 slides) |
| `create_rag_project_pdfs.ps1` | PowerShell script that generated the synopsis/gaps PDFs |
| `docs/sakshya.html` | Local copy of the Sakshya artifact (open in a browser) |
| `README.md`, `.gitignore`, `.gitattributes` | Added during repo setup |

**Important:** The six research-paper PDFs in the folder were **0 bytes (empty)**, and were not committed: the Persian hybrid RAG paper, `2025.findings-acl.295.pdf`, the Multilingual IDP paper, the translation-based embeddings paper, LegalRAG, and M3-RAG. All paper-level analysis relied on the team's research-gaps review summaries. They need to be re-downloaded.

---

## 3. The original proposal (as written by the team)

### 3.1 Abstract and synopsis (the strongest framing)
- End-to-end RAG pipeline: ingestion (PDF, DOCX, scans), OCR (Tesseract), language identification, page-aware chunking, semantic indexing, retrieval, reranking, evidence-grounded generation with **page-level citations**.
- **Core research contribution:** a controlled comparison of three retrieval strategies:
  1. Direct multilingual embeddings (multilingual-E5)
  2. Translation-to-English then embed
  3. Hybrid dense + BM25 with cross-encoder reranking
- An "adaptive router **may** select the route" (optional, not specified).
- Evidence-sufficiency and **abstention** when evidence is insufficient.
- Stack: multilingual-E5, PostgreSQL + pgvector (HNSW), BM25, multilingual cross-encoder, FastAPI, Next.js.
- Evaluation: Recall@k, MRR@10, nDCG@10, per-language results, human correctness, citation precision/recall, faithfulness, abstention precision, latency, GPU/CPU memory, OCR robustness, cross-lingual pairs. Baselines: vanilla dense, translation-only, hybrid without reranking/abstention.
- Languages: English, Hindi, Marathi (plus one or two more, only if evaluated).

### 3.2 Research gaps review (7 papers)
1. **Investigating Language Preference of Multilingual RAG Systems** (Park & Lee, Findings of ACL 2025): retrievers and generators prefer high-resource/query language and Latin script; proposes DKM-RAG. Gap: no private OCR documents, page provenance, or Indian deployment.
2. **Resource-Efficient Multilingual RAG for Indian Languages** (2026): English embedding pipeline plus output translation for 22 Indian languages. Gap: doesn't test when direct vs translation vs hybrid is best.
3. **M3-RAG** (Xu, Zhu, Ba, 2025): multimodal + multilingual RAG with modality routing. Gap: no page-cited QA over private PDF/DOCX/scans on limited hardware.
4. **Translation-Based Embedding Strategies for Non-English Documents** (Iscan, Ozara, Akbulut, 2024): translate to English before embedding. Gap: no adaptive policy for which queries benefit from translation.
5. **A Multilingual Intelligent Document Processing System** (2025): functional multilingual upload + QA. Gap: no retrieval comparison, faithfulness, out-of-scope behaviour, or evaluation across formats.
6. **LegalRAG** (Kabir et al., 2025): Bangla–English legal gazette QA with a relevance-check LLM and query refinement. Gap: domain-specific; struggles with out-of-context questions.
7. **Hybrid Layered Retrieval and Task-Aware Embeddings for Persian RAG** (Zoughi et al., 2026): FAQ + dense + BM25 + LLM fallback. Gap: Persian/FAQ-specific; latency rises on limited GPUs.

**Consolidated gap:** no study evaluates an end-to-end private-document RAG that jointly compares direct, translation and hybrid retrieval for English and Indian languages while measuring citation correctness, answer-language fidelity, OCR robustness, abstention, latency and cost.

**Research questions:**
- RQ1: Which retrieval strategy gives the best quality and cost per language and document type?
- RQ2: Does hybrid + cross-encoder reranking improve page-cited answer faithfulness?
- RQ3: Can evidence-sufficiency checks reduce unsupported answers without unacceptable latency?
- RQ4: How consistently does the system answer in the requested language?

### 3.3 Detailed plan v2 (drifts toward a SaaS product)
- Multi-tenant SaaS: JWT auth, rate limiting, AWS S3, Redis, analytics dashboard, Sentry, Vercel + Railway/Render.
- Ingestion: pdfplumber, python-docx, TXT (no OCR listed), langdetect, LangChain RecursiveCharacterTextSplitter (512 tokens, 64 overlap), multilingual-E5-large (1024-dim), pgvector HNSW.
- Query: dense-only top-5, Claude API (claude-haiku) with "answer ONLY from context" prompt, SSE streaming, citations appended.
- Claimed novelty: cross-lingual retrieval via multilingual-E5 in a multi-tenant SaaS.
- ML tracks: (1) fine-tune E5 with InfoNCE on 2–5k LLM-generated triplets; (2) train a BERT-base-multilingual-cased cross-encoder from scratch on 3–8k pairs; (3) QLoRA fine-tune Llama-3.2-3B / Mistral-7B. Predicted "~12% P@5 improvement".
- Datasets: AI4Bharat IndicCorp, NPTEL PDFs, Government gazettes, Wikipedia dumps, 15–20 custom documents.
- Large future roadmap (GraphRAG, multimodal, agentic RAG, voice, WhatsApp bot, RBAC, SSO, billing, ONNX, etc.).

### 3.4 Review 1 deck
- Claims 100+ languages; generator is QLoRA LLaMA-3.2-3B via Ollama; reranker trained on multilingual MS-MARCO; 6-month timeline.
- Slide 2 uses unsourced statistics ("73% of enterprise data is unstructured", "3.5 hrs avg. daily time spent searching").
- Literature survey lists only 2020–2023 foundation papers (RAG, DPR, E5, multilingual SBERT, LoRA, QLoRA), not the 7 multilingual RAG papers.

---

## 4. Critique of the original proposal

**Strengths to keep:** RQ1–RQ4 framing; page-level metadata on chunks; PostgreSQL + pgvector + HNSW; cross-encoder reranking; the synopsis metric list; two-track team split.

**Findings (by severity):**
- **Critical: three documents describe three different systems** (SaaS + Claude in plan v2; QLoRA LLaMA + 100+ languages + 6 months in the deck; retrieval comparison study in the synopsis). Adopt the synopsis framing as the single source of truth.
- **Critical: weak novelty.** Cross-lingual retrieval is a built-in property of multilingual-E5. The adaptive router, the one original idea, is only "may".
- **Critical: hallucination control is just a prompt instruction**, yet the plan claims it "eliminates fabrication". There is no mechanism or metric.
- **High: citations can be wrong by construction.** The recursive splitter can create chunks spanning two pages. OCR is missing from plan v2's ingestion steps.
- **High: langdetect fails** on short queries, Hindi vs Marathi (same script), and romanized input.
- **High: three heavy training jobs on a T4.** A cross-encoder trained from base BERT on a few thousand pairs will likely underperform an existing multilingual reranker. Llama-3.2's official languages include Hindi but not Marathi.
- **High: evaluation leakage.** LLM-generated queries are used for both training and "held-out" evaluation. IndicCorp/Wikipedia have no QA labels, and there is no unanswerable-question test.
- **Medium:** unsourced statistics and a predicted gain before any experiment. PostgreSQL full-text search has no Marathi configuration. Scans often use Devanagari digits (१४२), which never match "142".

---

## 5. The proposed solution: Sakshya (साक्ष्य, "evidence")

**One-line pitch:** We don't stop at comparing three retrieval strategies. We learn which one each question needs, answer in the script it was asked in, and check every sentence against the page it cites.

**What it does:** answers questions over an office's own English, Hindi and Marathi files (including old scans). It replies in the language *and script* the user typed, cites the exact page for every sentence, and abstains with the nearest pages when the documents don't support an answer.

### 5.1 Architecture
**Ingest (once per document):**
Upload (PDF/DOCX/scan) → Parse + OCR (Tesseract hin+mar, keep word confidence) → Script + language ID per block (IndicLID, Hindi/Marathi check) → Page-anchored chunks (never cross a page; store page, bounding box, paragraph, OCR confidence) → Embed + index into three indexes:
- Dense: multilingual-E5 in pgvector (HNSW), used by the direct and hybrid routes
- Lexical: BM25 on IndicNLP-normalized tokens + character n-grams, used by the hybrid route
- English shadow: IndicTrans2 translations of chunks, used by the translate route

**Ask (per question):**
Question (any language/script) → Understand query (IndicLID, IndicXlit transliteration, IndicTrans2 translation, intent, entities, digit normalization १४२↔142) → **Learned router** (picks direct / translate / hybrid + fusion weights) → retrieve top-40 → fine-tuned multilingual cross-encoder rerank → top-6 → **Evidence gate** (calibrated sufficiency score ≥ τ; below τ it abstains and skips the LLM) → Generate (small local 4-bit LLM, one citation per sentence) → **Verify** (multilingual NLI per claim + language/script check) → Cited answer (opens the PDF page with the passage highlighted) **or** Abstain (says what is missing, shows nearest pages).

### 5.2 Novelty (each tied to a paper gap and a hypothesis)
- **N1: Learned retrieval router.** Run all three routes on each training question, label which found the gold page best (oracle labels), and train a small gradient-boosted model on cheap features: script, romanization, code-mix ratio, numbers/IDs, query length, corpus language mix, share of scanned pages. This turns the RQ1 comparison into a deployable policy. Closes the gaps in papers 2 and 4. Tested by H1, H2.
- **N2: Romanized and code-mixed queries** (Hinglish, romanized Marathi). Three query views (as typed, transliterated to Devanagari, translated to English) plus digit normalization. None of the 7 papers evaluate this. Tested by H3.
- **N3: OCR-aware retrieval and citation.** Per-chunk OCR confidence feeds the router, shifts fusion weight toward character n-grams on noisy pages, and adds "check the page" badges. Closes the gaps in papers 3 and 5. Tested by H4.
- **N4: Claim-level verification + calibrated abstention.** A pre-generation evidence gate, then a per-sentence NLI check. Unsupported sentences are dropped; if too few survive, the system abstains. The threshold is set from a risk–coverage curve. Addresses LegalRAG's out-of-context failure. Tested by H5.
- **N5: Script-faithful answering.** Romanized in, romanized out (with a Devanagari toggle); mismatches are repaired with transliteration or IndicTrans2. Addresses the language/Latin-script bias from the ACL 2025 paper and gives RQ4 a mechanism. Tested by H6.
- **C6: Sakshya-Bench.** A small public benchmark (below).

**Honest framing for examiners:** hybrid search, NLI checking and transliteration exist individually. The novelty is (a) a router trained on the outcome of our own controlled comparison, so the study yields a working policy, and (b) evaluation under conditions the papers skip together: romanized queries, scanned Marathi, page-level citations, unanswerable questions and a T4 budget.

### 5.3 Models
- **Trained by the team (ML core):** the router (gradient-boosted trees); a fine-tuned pretrained multilingual cross-encoder (e.g. mMiniLM trained on mMARCO) with hard negatives; evidence-gate calibration (logistic); optionally a fine-tuned mDeBERTa XNLI verifier on Hindi/Marathi claim pairs.
- **Used as-is:** multilingual-E5 (base vs large as a budget variable); AI4Bharat IndicLID, IndicXlit, IndicTrans2; Tesseract (hin, mar); a 2–4B instruct model in 4-bit via llama.cpp/Ollama, chosen by a Hindi/Marathi pilot; optional BGE-M3 ablation.
- **Stretch only:** QLoRA generator fine-tune, E5 contrastive fine-tune.

### 5.4 Delivery (kept small)
FastAPI + PostgreSQL/pgvector; a Next.js single-workspace app with upload, ask, a PDF viewer with highlighted evidence, and a "why this route" panel. Multi-tenancy, JWT, S3, Redis, analytics and billing move to future work.

### 5.5 Evaluation
| ID | Hypothesis | Measured by | Baseline |
| --- | --- | --- | --- |
| H1 | No single fixed route wins across language × doc type × query type (RQ1) | nDCG@10, Recall@20 per slice; oracle bound | 3 fixed routes |
| H2 | Learned router ≥ best fixed route at lower cost (N1) | nDCG@10, ms/query, % sent to translation | best fixed, always-hybrid, rules |
| H3 | Transliteration lifts recall on romanized/code-mixed queries (N2) | Recall@20 on romanized slice | raw query, langdetect |
| H4 | OCR-aware weighting holds recall as scan quality drops (N3) | Recall@20 by OCR bucket; synthetic 5–30% CER | no OCR features |
| H5 | Gate + claim checks cut unsupported sentences at modest latency (RQ2, RQ3, N4) | unsupported-claim rate, citation P/R, abstention P/R, AURC, added ms | prompt-only guard (plan v2) |
| H6 | Script-faithful answering raises language + script match (RQ4, N5) | fidelity by query script, human fluency | no fidelity check |

**Sakshya-Bench:** ~60 public documents (Maharashtra GRs, university circulars, gazettes, NPTEL notes; ~40% scanned); ~400 human-checked questions with gold document/page/span. Composition: 40% same-language, 30% cross-language, 15% romanized/code-mixed, 15% unanswerable. 15% is double-annotated for agreement. Sanity checks on IndicQA (includes Marathi) and MIRACL-Hindi. Budget reporting: p50/p95 latency and peak RAM/VRAM on a T4 and a 16 GB CPU laptop, plus a quality-vs-cost Pareto chart.

### 5.6 12-week plan (Member 1 = retrieval & models; Member 2 = data, generation & UI)
- **W1–2:** M1 page-anchored ingestion with OCR confidence and the 3 indexes. M2 collects ~60 docs, writes the annotation guideline/tool and a minimal UI. Scope frozen.
- **W3–4:** M1 builds the 3 fixed routes, eval harness and query understanding. M2 annotates ~400 questions (M1 double-annotates 15%) and builds the PDF viewer with bbox highlight. Bench v1.
- **W5–6:** M1 runs the RQ1 comparison, derives oracle labels and trains the router (document-level CV). M2 fine-tunes the reranker and builds the "why this route" panel. N1 + N2 done.
- **W7–8:** M1 builds the evidence gate, NLI verifier, risk–coverage calibration and OCR-aware fusion. M2 runs the generator pilot and builds the script-faithful output path. **Research core done for Review 2.**
- **W9–10:** Both run ablations H1–H6, latency/RAM on T4 and CPU, the OCR-noise stress test, and error analysis.
- **W11–12:** Both write the report and a short paper draft, build a demo with 4 scenarios, and release the code and Sakshya-Bench. Review 3.

### 5.7 Risks and mitigations
Annotation time (start W1, cap at 400, human-verify LLM drafts); router overfitting (shallow model, ~15 features, CV by document, report oracle gap); weak Marathi generation (generate in Hindi/English and translate with IndicTrans2); NLI weaker in Marathi (XNLI lacks Marathi, so fine-tune on 1–2k checked pairs); Tesseract on old scans (report it as the H4 robustness curve); T4 memory (E5-base vs large, ONNX int8, 4-bit GGUF).

### 5.8 Fixes needed in the current documents
1. Re-download the six empty paper PDFs.
2. Pick one generator story across plan, deck and synopsis.
3. Drop "100+ languages" and "eliminates hallucination".
4. Cite or remove the slide-2 statistics.
5. Replace "~12% P@5 improvement" with a hypothesis.
6. Add the 7 multilingual RAG papers to the deck's literature survey.
7. Add OCR and page-bounded chunking to plan v2 ingestion.

**Cut or defer:** multi-tenancy, JWT, rate limits, S3, Redis, Sentry, the Vercel/Railway split, the analytics dashboard, training a cross-encoder from base BERT, and QLoRA as a core deliverable.

---

## 6. Real-life use cases (Maharashtra)
1. **Gram Panchayat / Tehsil office:** scheme eligibility from scanned Marathi GRs + English guidelines; romanized Marathi queries (N2, N3, N5).
2. **University exam cell:** Hinglish questions about ATKT/backlog rules over English circulars (N1, N2, N5).
3. **Law firm / district court:** English questions over scanned Marathi land-acquisition orders + English Acts; Devanagari digit matching (N1, N3, N4).
4. **Primary health centre:** Hindi protocol questions; the system must abstain when protocols don't cover the topic. It supports search and does not give medical advice (N4, N5).
5. **Co-operative bank compliance:** RBI circulars in English/Hindi; dates and clause numbers; an audit trail of cited pages (N1, N4).
6. **Krishi Vigyan Kendra:** romanized Marathi crop-advisory questions answered with the advisory page; WhatsApp is future work (N2, N5).

---

## 7. The artifact

- **What:** an interactive, single-page HTML report titled **"Sakshya"**, published as a private Claude artifact: https://claude.ai/artifact/1UAUh7WkzKHaBxzH9eCw2H (share it from the page's Share menu for teammates or the guide to open it). A local copy is in the repo at `docs/sakshya.html`.
- **Design:** Mukta (display) and Hind (body) fonts, both with Devanagari support, plus IBM Plex Mono for data. The accent colour is "rubber-stamp violet", echoing official Indian office stamps; citations render as stamp-like chips. Light and dark themes; responsive down to phone width.
- **Sections:**
  1. **Hero:** the name with a साक्ष्य stamp, a one-paragraph description, and an example case card (romanized Marathi question → romanized Marathi answer with GR page citations, route and verification pills, OCR warning).
  2. **01 Plan review:** Keep / At risk / Change summary and 9 findings with severity pills and fixes.
  3. **02 Solution:** SVG architecture diagram (ingest row, three index bars, ask row, right-to-left answer row with an abstain branch; novelty-tagged boxes N1–N5) plus three columns: models you train, models used as-is, delivery.
  4. **03 Walkthrough (interactive):** four scenario tabs, each showing six pipeline stages (understand, route with probability bars, retrieve candidates with OCR badges, rerank + evidence gate meter, generate + per-claim NLI verification, final answer or abstention), with a "Play through the stages" button and a dashed box contrasting what plan v2 would likely do:
     - Gram Panchayat clerk: romanized Marathi → hybrid route → answer in romanized Marathi
     - Engineering student: Hinglish → translate route → Hinglish answer
     - Junior advocate: English question over Marathi scans → hybrid with digit normalization; plan v2 would cite the wrong survey number (१३८)
     - PHC health worker: Hindi question on a topic not in the files → direct route → gate below τ → abstention in Hindi
  5. **04 Use cases:** six cards (who, sample query, files, today's pain, what Sakshya does, novelty tags).
  6. **05 Novelty:** cards N1–N5 + C6, the "what is new" answer for examiners, and a novelty matrix (7 papers + current proposal + Sakshya × 9 capabilities, ●/◐/—).
  7. **06 Comparison:** 17-row table, current proposal vs Sakshya, with Keep/Strengthen/Change verdicts.
  8. **07 Evaluation:** H1–H6 table and the Sakshya-Bench composition bar.
  9. **08 12-week plan:** table by member and milestone.
  10. **09 Risks & fixes:** six risks, the fix list, the cut/defer list, and the one-line pitch.
- **Caveat:** all example documents, amounts (e.g. ₹3,00,000, ₹18,40,000/hectare) and scores in the artifact are **illustrative**, not measured. The novelty matrix is based on the gaps review, not the original papers. Both must be verified or replaced with real results.

---

## 8. Repository state

- GitHub: **https://github.com/ShadowMonarch9099/B.E-Project** (private), branch `main`, tracking `origin/main`.
- Commits: `bcb439a` Initial commit (documents, plan, deck, script, README, .gitignore, docs/sakshya.html) and `0e916c1` Mark PDFs and Office files as binary (`.gitattributes`: `*.pdf`, `*.pptx`, `*.docx`, `*.xlsx`, images as binary; `* text=auto`).
- The six empty paper PDFs are untracked on purpose.
- GitHub CLI 2.101.0 is installed at `C:\Program Files\GitHub CLI\gh.exe` but not logged in; the branch was published by the user directly.
- Git identity: Kush Honkalse.

---

## 9. Open next steps

1. Re-download the 7 papers and verify the novelty matrix cells.
2. Rewrite the plan and deck around the Sakshya / synopsis framing for Review 2.
3. Start W1–2 work: page-anchored ingestion with OCR confidence; document collection and annotation guideline for Sakshya-Bench.
4. Add collaborators (teammate, guide) to the GitHub repo and share the artifact.

**When helping further:** keep claims limited to what we evaluate. Treat the example numbers as illustrative. Prioritize the research core (router, query normalization, OCR-aware retrieval, verification/abstention, script fidelity, benchmark) over SaaS features.
