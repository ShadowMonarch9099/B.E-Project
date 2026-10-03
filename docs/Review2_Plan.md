# Sakshya: Project Plan for Review 2

**Multilingual Document Intelligence (RAG): a resource-efficient, citation-grounded question-answering system for private English, Hindi and Marathi document archives**

B.E. Final Year Project · Department of Artificial Intelligence & Data Science · MMCOE Pune (SPPU)
Team: Tanmay Gosavi (BA18), Kush Honkalse (BA20) · Guide: Mr. Rishikesh C. Yeolekar

> **Status of this document.** This is the single source of truth for the project from Review 2 onward. It follows the framing of the abstract and the enhanced synopsis, and adds the Sakshya design (`docs/sakshya.html`). Where `RAG_Project_Detailed_Plan_v2.pdf` or `BE_RAG_Review1_Redesigned.pptx` disagree with this document, this document wins. Section 13 lists every change and the reason for it.

---

## Contents

1. [Summary](#1-summary)
2. [Problem statement](#2-problem-statement)
3. [Scope](#3-scope)
4. [Aim and objectives](#4-aim-and-objectives)
5. [Research questions and hypotheses](#5-research-questions-and-hypotheses)
6. [System design](#6-system-design)
7. [Models](#7-models)
8. [Data: Sakshya-Bench](#8-data-sakshya-bench)
9. [Evaluation protocol](#9-evaluation-protocol)
10. [12-week plan](#10-12-week-plan)
11. [Risks and mitigations](#11-risks-and-mitigations)
12. [Literature survey](#12-literature-survey)
13. [Changes from earlier documents](#13-changes-from-earlier-documents)
14. [Deck revision guide for Review 2](#14-deck-revision-guide-for-review-2)
15. [Open items](#15-open-items)
16. [Decision log](#16-decision-log)

---

## 1. Summary

Sakshya (साक्ष्य, "evidence") answers questions over an office's own English, Hindi and Marathi files, including old scans. It answers in the language and script the user typed, cites the page behind every sentence, and abstains, showing the nearest pages, when the documents do not support an answer.

The research core is a controlled comparison of three retrieval strategies (direct multilingual embeddings, translate-to-English, and hybrid dense + BM25 with reranking). We go one step further than the comparison: we use its outcomes to **train a router that picks the strategy for each question**, so the study produces a working policy rather than only a table.

**One-line pitch:** we don't stop at comparing three retrieval strategies. We learn which one each question needs, answer in the script it was asked in, and check every sentence against the page it cites.

**What is new (stated carefully for examiners).** Hybrid search, NLI checking and transliteration each exist already. Our contribution is:

1. a retrieval router trained on the outcome of our own controlled comparison, and
2. an evaluation under conditions the surveyed papers do not test together: romanized and code-mixed queries, scanned Marathi, page-level citations, unanswerable questions, and a T4 / CPU-laptop budget.

---

## 2. Problem statement

Government offices, colleges, courts, banks and health centres in Maharashtra keep their records as a mix of English and Marathi or Hindi files: native PDFs, Word documents and scanned paper. Finding one answer means reading many files by hand. Three things make existing tools unreliable here:

- **Keyword search misses paraphrase and cross-language matches.** A question about "contract penalty" can miss a clause about "liquidated damages". A Marathi question can miss the English circular that answers it.
- **General-purpose LLMs answer without evidence.** They produce fluent answers with no page to check, which is not acceptable for legal, medical, academic or compliance questions.
- **Real queries and documents are messy.** People type Hinglish or Marathi in Roman script, scans carry OCR errors, and numbers appear in Devanagari digits (१४२) that never match "142". The best retrieval strategy changes with the language, the query and the document.

> The Review 1 deck opened with two statistics ("73% of enterprise data is unstructured", "3.5 hrs average daily time spent searching"). They had no source and have been removed. If we want a figure on this slide, we must cite a published source for it; otherwise we use the concrete examples above.

---

## 3. Scope

| | In scope | Out of scope (future work) |
| --- | --- | --- |
| **Languages** | English, Hindi, Marathi, in Devanagari and Roman script, including code-mixed (Hinglish, romanized Marathi) queries | Any other language, unless we add it *and* evaluate it |
| **Formats** | Native PDF, DOCX, scanned PDF / images (OCR) | Tables, charts and images as answer sources (needs a vision-language model) |
| **Users** | One workspace, one small team, one document collection | Multi-tenancy, organisations, roles, billing |
| **Deployment** | Runs on a Colab T4 GPU and a 16 GB CPU laptop; everything local | Cloud hosting, CDN, autoscaling |
| **Claims** | Only what Sections 5 and 9 measure | "100+ languages", "eliminates hallucination", predicted gains |

The system helps people **find and read** evidence. It does not give legal, medical or compliance advice.

---

## 4. Aim and objectives

**Aim.** To design and evaluate a citation-grounded multilingual RAG system for private English, Hindi and Marathi documents that chooses its retrieval strategy per question, answers in the user's language and script, and abstains when evidence is insufficient, within a T4 / CPU budget.

| # | Objective | Output |
| --- | --- | --- |
| O1 | **Page-anchored ingestion** of PDF, DOCX and scans, with OCR, per-block language and script ID, and chunks that never cross a page | Ingestion pipeline and chunk store |
| O2 | **Controlled comparison** of direct, translate and hybrid retrieval by language, document type and query type | RQ1 results table |
| O3 | **Learned router** that picks the route per question from cheap features | Trained router + H2 results |
| O4 | **Evidence gate and claim verification** that reduce unsupported sentences and abstain when needed | Calibrated gate, NLI verifier, H5 results |
| O5 | **Script-faithful answering** for Devanagari, romanized and code-mixed input | Output path + H6 results |
| O6 | **Sakshya-Bench**: a public, human-checked benchmark with gold pages and unanswerable questions | ~60 documents, ~400 questions, annotation guideline |

---

## 5. Research questions and hypotheses

The four research questions from the synopsis stay. Each now has testable hypotheses and a mechanism (N1–N5) that the system builds.

| RQ | Question |
| --- | --- |
| RQ1 | Which retrieval strategy gives the best retrieval quality and cost for each language and document type? |
| RQ2 | Does hybrid dense + lexical retrieval with cross-encoder reranking improve page-cited answer faithfulness? |
| RQ3 | Can evidence-sufficiency checks reduce unsupported answers without unacceptable latency? |
| RQ4 | How consistently does the system answer in the requested language? |

| ID | Hypothesis | Answers | Measured by | Compared against |
| --- | --- | --- | --- | --- |
| H1 | No single fixed route wins across language × document type × query type | RQ1 | nDCG@10, Recall@20 per slice; oracle upper bound | the 3 fixed routes |
| H2 | The learned router matches or beats the best fixed route at lower cost | RQ1 (N1) | nDCG@10, ms/query, % of queries sent to translation | best fixed route, always-hybrid, hand-written rules |
| H3 | Transliteration and digit normalization raise recall on romanized and code-mixed queries | RQ1 (N2) | Recall@20 on the romanized/code-mixed slice | raw query; `langdetect`-based handling |
| H4 | OCR-aware fusion keeps recall steadier as scan quality drops | RQ1 (N3) | Recall@20 by OCR-confidence bucket; synthetic 5–30% character error rate | same pipeline without OCR features |
| H5 | The evidence gate plus claim checks cut unsupported sentences at modest added latency | RQ2, RQ3 (N4) | unsupported-claim rate, citation precision/recall, abstention precision/recall, AURC, added ms | prompt-only guard ("answer only from the context") |
| H6 | Script-faithful answering raises language and script match | RQ4 (N5) | language + script match by query script; human fluency rating | no fidelity check |

We do **not** predict effect sizes in advance. Each hypothesis is reported as supported, not supported or inconclusive, with confidence intervals (Section 9.4).

---

## 6. System design

### 6.1 Overview

```
INGEST (once per document)
  Upload (PDF / DOCX / scan)
    → Parse + OCR           Tesseract hin+mar+eng; keep per-word confidence
    → Script + language ID  per block (IndicLID); Hindi/Marathi check
    → Page-anchored chunks  never cross a page; keep page, bbox, paragraph, OCR confidence
    → Three indexes
        Dense     multilingual-E5 in pgvector (HNSW)      used by direct + hybrid
        Lexical   BM25 on normalized tokens + char n-grams used by hybrid
        English   IndicTrans2 translation of each chunk   used by translate

ASK (per question)
  Question (any language, any script)
    → Understand   IndicLID, IndicXlit, IndicTrans2; digits १४२ ↔ 142; numbers/IDs
    → Router (N1)  picks direct / translate / hybrid (+ fusion weights)
    → Retrieve top-40 → cross-encoder rerank → top-6
    → Evidence gate (N4)  sufficiency score ≥ τ ? continue : ABSTAIN (no LLM call)
    → Generate     local 4-bit LLM, one citation per sentence
    → Verify (N4, N5)  NLI per sentence; language + script check and repair
    → Cited answer (opens the PDF page with the passage highlighted)
      or Abstain (says what is missing, shows the nearest pages)
```

### 6.2 Ingestion (Objective O1)

This replaces the ingestion steps in plan v2, which had no OCR and used a splitter that could produce chunks spanning two pages.

1. **Detect the file type.** Native PDF pages with a usable text layer go to the PDF parser (`pdfplumber` or PyMuPDF). Pages with no or broken text layer (checked by character count and the share of valid Devanagari/Latin characters) go to OCR. DOCX goes to `python-docx`.
2. **OCR.** Tesseract with `hin`, `mar` and `eng` language packs, page image at 300 DPI. Keep word-level confidence and bounding boxes from the TSV output. Store a page-level OCR confidence (mean word confidence) on every chunk from that page.
3. **Pages for DOCX.** DOCX has no fixed pages. We convert DOCX to PDF (LibreOffice headless) so citations point to a page the user can open, and keep the paragraph index as a secondary anchor.
4. **Script and language ID per block.** IndicLID per text block, not per page and not `langdetect`. `langdetect` fails on short text, confuses Hindi and Marathi (same script) and cannot handle romanized text. For Devanagari blocks where IndicLID is unsure between Hindi and Marathi, we keep both labels with scores.
5. **Normalization.** IndicNLP normalizer for Devanagari (nukta, matra and punctuation variants). Keep the original text for display; index the normalized text.
6. **Page-anchored chunking.** Split within a page only, on paragraph and sentence boundaries, to roughly 200–350 tokens with a one-sentence overlap *inside the same page*. A chunk never crosses a page. If a paragraph continues onto the next page, the two halves become two chunks, each citing its own page.
7. **Chunk record.** Every chunk stores:

   | Field | Purpose |
   | --- | --- |
   | `doc_id`, `page_no`, `para_idx`, `chunk_idx` | citation |
   | `bbox` (list of boxes) | highlight in the PDF viewer |
   | `text_raw`, `text_norm`, `text_en` | display, lexical index, translate route |
   | `lang`, `lang_scores`, `script` | router features, fidelity |
   | `source` (`native` / `ocr` / `docx`) | router features, reporting |
   | `ocr_conf` | router features, OCR-aware fusion, "check the page" badge |
   | `emb_e5` (vector) | dense index |

8. **Indexes.**
   - **Dense:** multilingual-E5 embeddings in PostgreSQL + pgvector with an HNSW index. E5-base vs E5-large is a budget variable we report.
   - **Lexical:** BM25 over IndicNLP-tokenized normalized text plus character 3–5-grams, computed in Python (for example with the `bm25s` library). We do not use PostgreSQL full-text search, because PostgreSQL ships no Marathi text-search configuration.
   - **English shadow:** IndicTrans2 translation of each non-English chunk, embedded with the same E5 model, for the translate route.

### 6.3 Query understanding (N2)

For each question we build up to three **views**: as typed; transliterated to Devanagari (IndicXlit) if it is romanized; translated to English (IndicTrans2) if it is not English. We also normalize digits (Devanagari ↔ ASCII), extract numbers, dates and IDs (survey numbers, GR numbers, clause numbers) for exact matching, and compute the router features.

### 6.4 Retrieval routes

| Route | What it does |
| --- | --- |
| **Direct** | Embed the question (as typed, plus the Devanagari view if romanized) with E5; search the dense index |
| **Translate** | Translate the question to English; search the English shadow index |
| **Hybrid** | Dense + BM25 (over all views), fused with weighted reciprocal rank fusion |

Each route returns the top 40 candidates. A fine-tuned multilingual cross-encoder reranks them and keeps the top 6.

### 6.5 Learned router (N1)

1. For every training question, run all three routes and record which one ranks the gold page highest (ties broken by cost). This is the **oracle label**.
2. Features (about 15, all cheap): query script, romanized flag, code-mix ratio, query length, has number/ID, question type, IndicLID confidence, collection language mix, share of scanned pages, mean OCR confidence of the top dense hits, score gap between routes' top hits.
3. Model: gradient-boosted trees (LightGBM or scikit-learn), shallow, with **cross-validation grouped by document** so no document appears in both train and test.
4. Output: a route and fusion weights. The UI shows the reason in a "why this route" panel.

### 6.6 OCR-aware retrieval (N3)

Page OCR confidence is a router feature. On low-confidence pages the hybrid route shifts fusion weight toward character n-grams, which tolerate OCR errors better than whole-word tokens. Citations from low-confidence pages carry a "check the page" badge.

### 6.7 Evidence gate, generation and verification (N4, N5)

This replaces "answer only from the context" as the only safeguard.

1. **Evidence gate (before generation).** A logistic model over reranker scores, score margins, route agreement and OCR confidence gives a sufficiency score. The threshold τ is picked from a risk–coverage curve on the development split. Below τ the system abstains **without calling the LLM** and shows the nearest pages.
2. **Generation.** A small local instruct model (Section 7.3) gets only the top chunks, each labelled with its page, and must end every sentence with a citation marker.
3. **Claim verification (after generation).** Each sentence is checked against the passage it cites with a multilingual NLI model. Sentences that are not entailed are removed. If too few supported sentences remain, the system abstains.
4. **Script fidelity.** The answer's language and script are compared with the question's. Romanized in, romanized out (with a Devanagari toggle). Mismatches are repaired with IndicXlit (script) or IndicTrans2 (language). If the generator is weak in Marathi, it writes in Hindi or English and IndicTrans2 translates the result.

### 6.8 Delivery

- **Backend:** FastAPI; PostgreSQL + pgvector; background ingestion with FastAPI background tasks (no Celery or Redis).
- **Frontend:** Next.js, one workspace: upload, ask, a PDF viewer that opens the cited page with the passage highlighted, the "why this route" panel, and abstention messages with the nearest pages.
- **Run:** Docker Compose on a laptop; model experiments on Colab T4.

---

## 7. Models

### 7.1 Trained or fine-tuned by the team (the ML core)

| Model | What we do | Data | Why |
| --- | --- | --- | --- |
| **Retrieval router** | Train gradient-boosted trees | Oracle labels from running all 3 routes on training questions | Turns the RQ1 comparison into a deployable policy (N1) |
| **Cross-encoder reranker** | Fine-tune a pretrained multilingual cross-encoder (for example an mMiniLM model trained on mMARCO) with hard negatives from our routes | Training split of Sakshya-Bench + mined hard negatives | A pretrained multilingual reranker is a much stronger start than BERT-base trained from scratch on a few thousand pairs |
| **Evidence-gate calibrator** | Logistic regression; τ from risk–coverage | Development split | Makes abstention a measured decision, not a prompt instruction (N4) |
| **NLI verifier** (optional) | Fine-tune a multilingual NLI model (for example mDeBERTa-v3 XNLI) on 1–2k checked Hindi/Marathi claim–passage pairs | Pairs drawn from our generated answers, human-checked | XNLI has Hindi but not Marathi |

### 7.2 Used as-is

multilingual-E5 (base and large, as a budget variable); AI4Bharat IndicLID, IndicXlit and IndicTrans2 (distilled variants for CPU); IndicNLP library; Tesseract 5 with `hin`, `mar`, `eng`; optional BGE-M3 as an embedding ablation.

### 7.3 Generator: one decision

Earlier documents gave three answers (Claude API in plan v2, QLoRA LLaMA-3.2-3B in the deck, "lightweight generator" in the synopsis). **Decision:** a **2–4B open instruct model, 4-bit GGUF, run locally with llama.cpp or Ollama**. No hosted LLM API in the system, because the target users' documents are private.

The exact model is chosen in weeks 7–8 by a **pilot** on 50 development questions (Hindi, Marathi, English, romanized), scoring citation-format compliance, human-rated fluency in each language, and latency on T4 and CPU. Candidates include Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct and Gemma-family 2–4B instruct models. Llama-3.2 officially supports Hindi but not Marathi, which is why the pilot and the translate-back fallback in 6.7 exist.

### 7.4 Stretch only (after the research core is done)

QLoRA fine-tuning of the generator; contrastive (InfoNCE) fine-tuning of E5 on our data. Neither is a Review 2 or Review 3 deliverable.

---

## 8. Data: Sakshya-Bench

Plan v2 proposed IndicCorp, Wikipedia dumps and LLM-generated queries used for both training and evaluation. IndicCorp and Wikipedia have no question–answer labels, and using the same LLM-generated queries for training and "held-out" testing leaks. Sakshya-Bench replaces this.

### 8.1 Documents

About 60 public documents: Maharashtra government resolutions (GRs), university circulars, gazette notifications, NPTEL notes. Target about 40% scanned. Mix of English, Hindi and Marathi, with some bilingual documents. We record the source URL and licence of every document.

### 8.2 Questions

About 400 human-checked questions, each with gold document, page and evidence span.

| Slice | Share | Example |
| --- | --- | --- |
| Same language as the document | 40% | Marathi question over a Marathi GR |
| Cross-language | 30% | English question over a Marathi order |
| Romanized or code-mixed | 15% | "ATKT ke liye kitne subject allowed hai?" |
| Unanswerable from the collection | 15% | A topic the documents do not cover |

- An LLM may draft candidate questions, but **a human writes or edits and verifies every question and gold span**.
- 15% of questions are annotated by both members; we report agreement (Cohen's κ on the gold page, token-F1 on spans).
- **Splits are by document** (roughly 50 / 15 / 35% of documents for train / dev / test), so no test document is seen by the router, reranker or calibrator during training. The test split is frozen before any tuning.

### 8.3 Sanity checks on public sets

IndicQA (includes Marathi) and MIRACL-Hindi, to check that our retrieval numbers are in a reasonable range against published baselines.

### 8.4 Release

Documents list with URLs, questions, gold labels, annotation guideline and evaluation scripts, released with the code.

---

## 9. Evaluation protocol

### 9.1 Metrics

| Area | Metrics |
| --- | --- |
| Retrieval | Recall@k (k = 5, 20), MRR@10, nDCG@10, each per language, document type (native / scanned) and query type; oracle upper bound for routing |
| Answers | Human correctness (3-point scale); citation precision and recall at page level; unsupported-claim rate (share of sentences not entailed by their cited passage, human-checked on a sample) |
| Abstention | Abstention precision and recall on the unanswerable slice; risk–coverage curve and AURC |
| Language fidelity | Language match and script match by query script; human fluency rating |
| Robustness | Recall@20 by OCR-confidence bucket; Recall@20 under synthetic OCR noise (5–30% character error rate) |
| Cost | p50 / p95 latency per stage; peak RAM and VRAM; on a Colab T4 and a 16 GB CPU laptop; quality-vs-cost Pareto chart |

### 9.2 Baselines

1. Vanilla dense multilingual RAG (E5 only, no reranker, prompt-only guard). This is essentially plan v2's pipeline with a local generator.
2. Translate-only RAG.
3. Hybrid without reranking and without the evidence gate.
4. Each fixed route with the full answer stack, to isolate the router.

### 9.3 Ablations

Remove one component at a time: router → best fixed route; transliteration view; digit normalization; OCR-aware weighting; reranker; evidence gate; NLI verifier; script repair; E5-base vs E5-large.

### 9.4 Reporting

- Paired bootstrap confidence intervals (1,000 resamples over questions) for every comparison; we call a difference real only if the 95% interval excludes zero.
- Results reported per slice, not only overall averages.
- Every number in the report comes from the frozen test split, with the command that produced it.

---

## 10. 12-week plan

**Member 1 (M1):** retrieval and models. **Member 2 (M2):** data, generation and UI.

| Weeks | M1 | M2 | Milestone |
| --- | --- | --- | --- |
| 1–2 | Page-anchored ingestion with OCR confidence and IndicLID; the 3 indexes | Collect ~60 documents; annotation guideline and tool; minimal UI (upload, list) | Scope frozen |
| 3–4 | The 3 fixed routes; evaluation harness; query understanding (views, digits) | Annotate ~400 questions (M1 double-annotates 15%); PDF viewer with bounding-box highlight | Sakshya-Bench v1; splits frozen |
| 5–6 | RQ1 comparison; oracle labels; train router (grouped CV) | Fine-tune reranker; "why this route" panel | N1 + N2 done |
| 7–8 | Evidence gate; NLI verifier; risk–coverage calibration; OCR-aware fusion | Generator pilot; script-faithful output path; abstention UI | **Research core done → Review 2** |
| 9–10 | Ablations H1–H6; latency and memory on T4 and CPU; OCR-noise stress test | Human evaluation (correctness, fluency); error analysis | All results in |
| 11–12 | Report and short paper draft; code release | Demo with 4 scenarios; benchmark release | **Review 3** |

### 10.1 What we show at Review 2

- Working ingestion on native, DOCX and scanned Marathi documents, with page-correct citations.
- Sakshya-Bench v1 statistics (documents, questions by slice, agreement).
- RQ1 results table on the dev split: three routes by language × document type × query type, plus the oracle bound.
- The trained router's first results against the best fixed route.
- The evidence gate's risk–coverage curve.
- A live end-to-end demo of one answered and one abstained question.

### 10.2 What we show at Review 3

Final H1–H6 results on the frozen test split, ablations, cost measurements, error analysis, the four demo scenarios, and the released code and benchmark.

---

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Annotation takes longer than planned | Start in week 1; cap at 400 questions; LLM drafts, human verifies; M1 helps in weeks 3–4 |
| Router overfits a small training set | Shallow model, ~15 features, CV grouped by document; report the gap to the oracle |
| Weak Marathi generation | Pilot in weeks 7–8; fall back to generating in Hindi or English and translating with IndicTrans2 |
| NLI is weaker in Marathi (XNLI has no Marathi) | Fine-tune on 1–2k checked pairs; report per-language verifier accuracy |
| Tesseract struggles on old scans | Report it as the H4 robustness curve rather than hiding it; flag low-confidence pages in the UI |
| T4 memory and CPU latency | E5-base vs large; distilled IndicTrans2; ONNX int8 for encoders; 4-bit GGUF generator |
| Scope creep toward SaaS features | Section 3 is fixed; anything in the "future work" column needs both members and the guide to agree |

---

## 12. Literature survey

The Review 1 deck listed only foundation papers (2020–2023). The Review 2 deck must include the seven multilingual RAG papers from the research-gaps review, because they define our gap. Citation details below come from the team's gaps review; they must be checked against the papers once the PDFs are re-downloaded (Section 15).

### 12.1 Multilingual RAG papers (the gap)

| # | Paper | What it does | Gap it leaves | Our response |
| --- | --- | --- | --- | --- |
| 1 | Park & Lee (2025). *Investigating Language Preference of Multilingual RAG Systems.* Findings of ACL 2025 | Shows retrievers and generators prefer high-resource / query language and Latin script; proposes DKM-RAG | No private OCR documents, page provenance or Indian deployment | Script-faithful answering (N5), H6 |
| 2 | *Resource-Efficient Multilingual RAG for Indian Languages* (2026) | English embedding pipeline, translate only the output, 22 Indian languages | Does not test when direct vs translate vs hybrid is best | RQ1 comparison and learned router (N1), H1–H2 |
| 3 | Xu, Zhu & Ba (2025). *M3-RAG: Unified Multimodal and Multilingual RAG* | Text + image retrieval with modality routing | No page-cited QA over private PDF/DOCX/scans on limited hardware | OCR-aware retrieval and page citations (N3), H4 |
| 4 | Iscan, Ozara & Akbulut (2024). *Enhancing RAG Pipeline Performance with Translation-Based Embedding Strategies for Non-English Documents* | Translate to English before embedding | No policy for which queries benefit from translation | Learned router (N1), H2 |
| 5 | *A Multilingual Intelligent Document Processing System* (2025) | Multilingual upload and QA | No retrieval comparison, faithfulness, out-of-scope handling or format-wise evaluation | Full evaluation protocol, Sakshya-Bench (C6) |
| 6 | Kabir et al. (2025). *LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval.* arXiv:2504.16121 | Bangla–English legal gazette QA with relevance checking and query refinement | Domain-specific; struggles with out-of-context questions | Evidence gate and abstention (N4), H5 |
| 7 | Zoughi et al. (2026). *Hybrid Layered Retrieval and Task-Aware Embeddings for Efficient Persian RAG Systems* | FAQ + dense + BM25 + LLM fallback | Persian/FAQ-specific; latency rises on limited GPUs | Hybrid route for Indian languages under a T4/CPU budget, cost reporting |

**Consolidated gap.** No surveyed study evaluates an end-to-end private-document RAG system that jointly compares direct, translation-based and hybrid retrieval for English and Indian languages while measuring citation correctness, answer-language fidelity, OCR robustness, abstention, latency and cost.

### 12.2 Foundation papers (keep, but as background)

Lewis et al. 2020 (RAG, NeurIPS); Karpukhin et al. 2020 (DPR, EMNLP); Wang et al. 2022 (E5; and the multilingual-E5 technical report, 2024); Reimers & Gurevych 2020 (multilingual SBERT, EMNLP). LoRA and QLoRA move to a "stretch" note because generator fine-tuning is no longer a core deliverable. Add the tools we depend on: IndicTrans2 (Gala et al., 2023), IndicLID (Madhani et al., 2023), IndicXlit (Madhani et al., 2023), and Nogueira & Cho 2019 (BERT passage reranking).

---

## 13. Changes from earlier documents

### 13.1 Fixes applied (items 2–7 of the Sakshya fix list)

| # | Fix | Before | Now | Where |
| --- | --- | --- | --- | --- |
| 2 | One generator story | Claude API (plan v2); QLoRA LLaMA-3.2-3B via Ollama (deck); unspecified (synopsis) | Local 2–4B instruct model, 4-bit, chosen by a pilot; QLoRA is stretch only | §7.3 |
| 3 | Drop overclaims | "100+ languages"; "eliminates hallucination / fabrication" | English, Hindi, Marathi only; unsupported claims are *measured and reduced*, with abstention | §3, §5, §6.7 |
| 4 | Unsourced statistics | "73% of enterprise data…", "3.5 hrs…" on slide 2 | Removed; replaced by concrete examples (cite a source if a figure is reintroduced) | §2 |
| 5 | Predicted gain | "~12% P@5 improvement" | Hypotheses H1–H6 with no predicted effect size; bootstrap CIs | §5, §9.4 |
| 6 | Literature survey | Only 2020–2023 foundation papers in the deck | Seven multilingual RAG papers + foundation papers | §12 |
| 7 | Ingestion | No OCR in plan v2; 512-token recursive splitter could cross pages; `langdetect` | Tesseract OCR with confidence; page-anchored chunks; IndicLID per block | §6.2 |

Item 1 (re-download the six empty paper PDFs) is still open; see Section 15.

### 13.2 Other corrections

| Topic | Before | Now | Reason |
| --- | --- | --- | --- |
| Novelty claim | Cross-lingual retrieval with multilingual-E5 in a multi-tenant SaaS | Learned router + evaluation under combined real conditions | Cross-lingual retrieval is a built-in property of multilingual-E5 |
| Router | "An adaptive router *may* select the route" | Specified and trained (§6.5) | It was the one original idea and was left optional |
| Reranker | BERT-base-multilingual-cased trained from scratch on 3–8k pairs | Fine-tune a pretrained multilingual cross-encoder | A from-scratch reranker on so little data is likely to underperform existing ones |
| Embedding fine-tuning | Core track (InfoNCE on 2–5k LLM triplets) | Stretch only | T4 budget; LLM-generated training data risks leakage |
| Lexical search | PostgreSQL `tsvector` (plan v2 future work) | BM25 in Python on IndicNLP tokens + char n-grams | PostgreSQL has no Marathi text-search configuration |
| Numbers | Not handled | Devanagari ↔ ASCII digit normalization; exact ID matching | Scans use १४२; plain matching misses "142" |
| Evaluation data | IndicCorp, Wikipedia, LLM queries for train and test | Sakshya-Bench with human-checked gold pages, document-level splits, unanswerable slice | No QA labels in IndicCorp/Wikipedia; train/test leakage |
| Timeline | 6 months (deck) vs 12 weeks (plan v2) | 12 weeks (§10) | One timeline |
| Evaluation metrics | Precision@5, BLEU, user study | §9 metric list, following the synopsis | BLEU does not measure faithfulness or citation correctness |

### 13.3 Cut or deferred to future work

Multi-tenancy and organisation schema; JWT auth and rate limiting; AWS S3; Redis and Celery; Sentry; the Vercel / Railway split; the analytics dashboard; conversation memory; training a cross-encoder from base BERT; QLoRA as a core deliverable; everything in plan v2's "Future Enhancements" section (GraphRAG, multimodal, agentic RAG, voice, WhatsApp, RBAC, SSO, billing, and so on).

---

## 14. Deck revision guide for Review 2

Slide-by-slide changes to turn `BE_RAG_Review1_Redesigned.pptx` into the Review 2 deck.

| Slide | Review 1 content | Review 2 change |
| --- | --- | --- |
| 1 Title | "Multilingual Document Intelligence (RAG)" | Add the name: "Sakshya: citation-grounded multilingual document intelligence" |
| 2 Introduction | Unsourced 73% / 3.5 hrs; "100+ languages" | Problem from §2 with the three concrete failure examples; languages: English, Hindi, Marathi |
| 3 Literature survey | 6 foundation papers | The 7 papers in §12.1 (contribution / gap / our response) + consolidated gap; foundation papers as one line |
| 4 Problem statement | "50+ languages", French, German, Arabic | §2 problem statement; Maharashtra use cases |
| 5 Objectives | Fine-tune E5, BERT reranker, QLoRA, multi-tenant SaaS | Objectives O1–O6 (§4) |
| 6 Proposed system | Celery/Redis, QLoRA LLaMA, multi-tenant | Architecture from §6.1 (reuse the SVG from `docs/sakshya.html`) |
| 7 Architecture | Dense top-20 → BERT → LLaMA | Router, evidence gate and verifier shown explicitly |
| 8 Methodology | 512-token chunks, `langdetect` | Page-anchored ingestion (§6.2) and the query pipeline |
| 9 Tech stack | S3, Celery, Redis, LangChain, Vercel | §6.8 and §7 (drop S3, Celery, Redis, LangChain) |
| 10 Work division | 6 months | 12-week table (§10) |
| 11 Conclusion | "Solves the gap… any language" | Review 2 results (§10.1) and what remains for Review 3 |
| New | — | RQ/hypothesis table (§5); Sakshya-Bench composition (§8); first RQ1 results; risk–coverage curve |

---

## 15. Open items

1. **Re-download the six empty paper PDFs** (Persian hybrid RAG, `2025.findings-acl.295.pdf`, Multilingual IDP, translation-based embeddings, LegalRAG, M3-RAG) and the Indian-languages paper, then verify every row of §12.1 and every cell of the novelty matrix in `docs/sakshya.html` against the papers themselves. Fill in missing author names for papers 2 and 5.
2. **Illustrative numbers.** All example documents, amounts and scores in `docs/sakshya.html` are illustrative. Replace them with real results from Sakshya-Bench before showing them as findings.
3. **Rebuild the deck** from §14.
4. **Regenerate the synopsis PDF** (`create_rag_project_pdfs.ps1`) if the guide wants the synopsis updated to match this plan; the enhanced synopsis is already mostly consistent, except the reference list (add papers 2, 5 and the tools in §12.2).
5. **Add collaborators** (teammate and guide) to the GitHub repository.
6. **Annotate Sakshya-Bench** (§8.2): about 400 questions at `/annotate`, by hand, with the 15% second annotation. None exist yet, so no benchmark number exists yet. Then run `python -m app.bench facts`, `splits` and `freeze` **before** any tuning, and re-choose τ on dev (`python -m eval tau`).
7. **Document gaps in the collection** (§8.1): no English university circulars, gazette notifications or NPTEL notes yet (the SPPU site's notice links were dead or scanned); 67 documents from GRs, collectorate notices, NHM ASHA modules and the Constitution. The Hindi-English Constitution's Hindi pages have no Unicode text layer.
8. **Answer providers:** add the Groq / OpenRouter / NVIDIA NIM keys to `backend/.env`, confirm the model on each with `python -m app.llm models`, and run the live smoke test (`SAKSHYA_LIVE_LLM=1`). Then `python -m eval cost` and, after agreeing the spend, `python -m eval answers` on dev for citation-format compliance per model (§7.3's pilot).
9. **Speed on CPU:** on the team laptop (x64 Python emulated on Snapdragon X Elite) indexing the 67 documents (2,283 pages) is slow, mostly OCR and NLLB translation for the English shadow index (which now runs only when nothing else is being indexed). Measure it per stage, and consider IndicTrans2-distilled or a smaller beam for the shadow index.
10. **Optional installs** on the team laptop: LibreOffice (DOCX page citations; the Docker image has it) and an `HF_TOKEN` for IndicTrans2.


---

## 16. Decision log

Decisions that change something in this plan, newest last. Each says what changed, why, and what it costs us.

### 2026-10-03 · Build every component without training; answers from the Claude API for now

**Decision.** Model training may not be possible in the time and on the hardware we have, so every component of §6 is built *without training*, and the generator is a hosted API for now:

| Plan component | Plan (§6–7) | Built now | What makes it trainable later |
| --- | --- | --- | --- |
| Generator (§7.3) | Local 2–4B instruct model, 4-bit | **Claude API (Anthropic)**, `claude-opus-5-5`, effort `medium`, with the API's citations feature | All calls go through one `AnswerProvider` interface (`backend/app/llm/`); a local model is a second provider, with no change to retrieval or the UI |
| Learned router (N1, §6.5) | Gradient-boosted trees on oracle labels | Rule-based router over the same cheap features | Every question's features, chosen route and per-route scores are written to `query_logs`: the router's training data |
| Cross-encoder reranker (§7.1) | Fine-tuned multilingual cross-encoder | Pretrained `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, used as-is | Hard negatives can be mined from `query_logs` |
| Evidence-gate calibrator (N4) | Logistic regression; τ from risk–coverage | Hand-set threshold τ on the best reranker score, chosen by sweeping τ on the dev split and reading the risk–coverage curve (no training) | The same curve and features feed a calibrator later |
| NLI verifier (§7.1) | Fine-tuned mDeBERTa-v3 XNLI | Pretrained multilingual mDeBERTa-v3 XNLI, used as-is, optional, on top of a citation check | Claim–passage pairs from logged answers |

**Out of scope now, recorded as future work:** QLoRA fine-tuning of a generator, E5 contrastive fine-tuning, reranker fine-tuning, the trained router, and the trained gate.

**Why.** It keeps the research questions answerable: RQ1 (route comparison) needs no trained component, and the plan's hypotheses about the router (H2) and the gate (H5) can still be tested with transparent rules as the baseline the trained versions must beat. A hosted generator removes the riskiest part of §7.3 (weak Marathi generation from a small local model) while the rest of the pipeline is built and measured.

**What it costs.** This deviates from §7.3's "no hosted LLM API in the system". For each question that passes the evidence gate, **the top 6 retrieved passages (never a whole document) and the question are sent to the Anthropic API**. The ask bar says so on screen. Questions below the gate never leave the machine; neither does anything else (OCR, retrieval, the indexes). Answers cost money per question (see the README), and need an API key, read from `ANTHROPIC_API_KEY`; without a key, detection, OCR, search and abstention keep working and the UI explains how to add one. Claims in the report about privacy must say that passages are sent to a hosted model while this decision stands.

**Also decided on the same day.**
- *IndicTrans2 is gated on Hugging Face* (it needs an account that accepts its terms and an `HF_TOKEN`). It stays the plan's translator and is used whenever such a token is set. Every translation records which model made it, so results from different translators are never mixed silently.
- *IndicXlit* ships only as fairseq checkpoints, and fairseq does not install on Python 3.11 / Windows. The official IndicXlit weights are run with a small PyTorch re-implementation of fairseq's transformer inference (`backend/app/ml/xlit.py`): the same dictionaries, language tokens and beam search (beam 5). The official engine's optional rescoring with word-frequency tables (850 MB) is not used.

### 2026-10-03 · Translation stand-in: NLLB-200, not OPUS-MT

**Decision.** Until IndicTrans2 is available, translation to English uses Meta's **NLLB-200 distilled 600M**, run with CTranslate2 in int8.

**Why.** Helsinki-NLP OPUS-MT (the first stand-in) produced unrelated English for all 5 Marathi test sentences we tried (for example "घरकुल योजना साठी उत्पन्नाची मर्यादा किती आहे?" → "What's the price of a gift?"); its multilingual `mul-en` model was no better. A wrong English view would actively mislead retrieval. NLLB translated the same sentences sensibly ("What is the income limit for household schemes?"). In PyTorch it took 5–11 s per sentence on our laptop; with CTranslate2 int8, 0.5–0.9 s, which makes both the question's English view and the English shadow index of chunks practical. Small samples, checked by reading; the translate route's real quality is measured in the evaluation (RQ1).

**What it costs.** NLLB-200's licence is CC-BY-NC 4.0 (non-commercial). That suits this academic project, but anything beyond it must switch to IndicTrans2 (MIT). Translations are approximate ("राष्ट्रीय उच्चतर शिक्षा अभियान" came out as "National Higher Penalty Mission"), which is one reason the translate route is compared, not assumed.

### 2026-10-03 · Language ID: IndicLID's FastText models only

**Decision.** IndicLID-FTN and IndicLID-FTR are used as-is; IndicLID's third stage (an IndicBERT classifier for romanized text the FastText model is unsure about, 969 MB) is not. Low-confidence romanized questions are settled by a short list of romanized Hindi and Marathi function words, and the scores are also reported restricted to English / Hindi / Marathi.

**Why.** FTR alone misread code-mixed questions ("third year me ATKT ke liye kitne backlog allowed hai?" as 56% English; "KYC update ki last date kya hai …" as Assamese). With the restriction and the word list, all 12 probe questions got the right language and romanized flag. The BERT stage remains an option if the evaluation shows romanized language ID errors.

### 2026-10-03 · Answer providers: Groq, OpenRouter and NVIDIA NIM (the Claude API kept as an option)

**Decision.** The team has API keys for Groq, OpenRouter and NVIDIA NIM, so answers now come from these three OpenAI-compatible chat APIs, through one more `AnswerProvider` (`backend/app/llm/openai_compat.py`, official `openai` Python SDK pointed at each provider's base URL). They are tried in order (`SAKSHYA_LLM_ORDER`, default Groq → OpenRouter → NVIDIA NIM); if one fails before it has streamed anything (rate limit, outage, unknown model, no credit), the next answers, and the answer says which provider wrote it. The default model on all three is `openai/gpt-oss-120b`, set per provider (`SAKSHYA_GROQ_MODEL`, …) and to be confirmed with `python -m app.llm models` once the keys are in `backend/.env`. The Claude provider stays and is used when only `ANTHROPIC_API_KEY` is set. This replaces the first entry's "answers from the Claude API" for the default configuration; everything else in that entry (no training, rule-based router, pretrained reranker, hand-set τ, optional NLI) is unchanged.

**What changes in the pipeline.** These APIs have no citations feature. The six passages are numbered in the user message ("[3] (file.pdf · p. 4)", numbered across the conversation so earlier turns never change), and the model must end each sentence with the numbers it relies on. The markers are parsed back into per-sentence citations, so the verifier (§6.7 step 3, N4) works exactly as before: a sentence without a valid marker is removed, and too few cited sentences make the answer an abstention. A model that ignores the format therefore produces abstentions, not uncited answers. Citation-format compliance per provider and model becomes something the evaluation measures (`python -m eval answers`), as §7.3's pilot intended.

**What it costs.** For each question that passes the evidence gate, the question and its top 6 passages are sent to whichever provider answers it (one of three companies, depending on the order and on failures); the ask bar names the provider and model. Keys live only in `backend/.env` (gitignored), are read on the server and never logged or sent to the browser. Free tiers have tight rate limits (tokens per minute), so evaluation runs must be paced, and their per-question cost depends on the provider and model (estimated before any run).
