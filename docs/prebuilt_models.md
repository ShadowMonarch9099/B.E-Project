# Prebuilt models for document detection and extraction

This note lists prebuilt, open models that fit the stages of the Review 2 plan (`docs/Review2_Plan.md`). The plan's choices come first; alternatives are marked as such and are **not** adopted unless the team and the guide agree (plan §11: no scope creep).

## In use now

| Stage | Model | What it does | Notes |
| --- | --- | --- | --- |
| File type detection (§6.1 step 1) | [Google Magika](https://github.com/google/magika) `standard_v3_3` | Deep-learning content-type detector: ~1 MB model, 200+ types, runs on CPU in milliseconds ([paper](https://arxiv.org/pdf/2409.13768)) | Used as-is from `pip install magika`. **Finding:** it labels plain Devanagari (Hindi/Marathi) text as "unknown", so the backend adds a UTF-8 text check before falling back to the file extension. |
| Dense retrieval (§6.2 step 8) | [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) (MIT) | 278M-parameter multilingual encoder, 768-dim vectors, used with the "query: " / "passage: " prefixes | Used as-is (no fine-tuning). Vectors in pgvector with an HNSW index. The large model is a setting (`SAKSHYA_EMBED_MODEL`). |
| Reranking (§6.4) | [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) (Apache-2.0) | Multilingual MiniLM cross-encoder trained on mMARCO; scores (question, passage) pairs | Used as-is (no fine-tuning); sigmoid of its logit is the evidence gate's score. |
| Language + script ID (§6.2 step 4, §6.3) | [AI4Bharat IndicLID](https://github.com/AI4Bharat/IndicLID) FTN + FTR (MIT) | FastText classifiers for Indian languages and English: FTN has 26 native-script labels, FTR 22 romanized labels | Used as-is; the BERT stage is not used (decision log). **Finding:** FTR alone misreads code-mixed Hinglish; restricting to en/hi/mr plus a function-word list fixed all 12 probe questions. |
| Transliteration (§6.3) | [AI4Bharat IndicXlit](https://huggingface.co/ai4bharat/IndicXlit) en→indic v1.0 (MIT) | Character-level transformer, roman → Devanagari | The fairseq checkpoint run in plain PyTorch (`backend/app/ml/xlit.py`), beam 5, no rescoring; about 80 ms per word on the team's laptop. |
| Translation to English (§6.3, §6.4) | IndicTrans2 indic-en dist-200M (gated) when `HF_TOKEN` is set; otherwise [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC 4.0) via CTranslate2 int8 | Neural MT | **Finding:** OPUS-MT mr→en produced unrelated English for 5/5 Marathi test sentences; NLLB was sensible. CTranslate2 int8 cut NLLB from 5–11 s to 0.5–0.9 s per sentence on CPU. |
| OCR for scans (§6.1 step 2) | [Tesseract](https://github.com/tesseract-ocr/tesseract) 5 with `eng`, `hin`, `mar` language data, via `pytesseract` | LSTM OCR; per-word confidence and boxes; PDF pages rendered at 300 DPI | **Findings:** (1) Language order matters: with `hin` first, the digit 1 was read as a danda ("142" → "42"). On 7 scanned pages of real Maharashtra GRs, `mar+hin+eng` (the default) had 19.3% character error and read 60% of amounts exactly; `eng+hin+mar` had 19.6% and 56%. (2) Tables are the weak spot: prose pages had 3-14% character error, table pages 40-49%, mostly from cell reading order and Devanagari digits misread as Latin look-alikes (१ → 9, ५ → 4). Page-mode 3 (Tesseract's default) beat modes 4 and 6. Small samples; re-check on Sakshya-Bench. |

Everything else in the current app is a plain parser, not a model: PyMuPDF (PDF pages and text-layer check), python-docx, Pillow and `zipfile`.

**Finding (text-layer check):** the team's two "native" sample GRs have a Unicode Devanagari text layer whose conjuncts are stored as stray Latin Extended-B / IPA characters ("राष्ट्रीय" → "राʊȄीय", "प्रकल्प" → "ɓकȧप"); 7–50% of the Devanagari words on each page were affected. PDFs that PyMuPDF itself writes in Lohit Devanagari show the same defect. Such pages are now classed as *broken* and OCR'd, which read them at 87% and 93% mean word confidence.

## Next stages, from the plan

| Plan stage | Model named in the plan | Link |
| --- | --- | --- |
| Claim verification (§6.7 step 3) | A multilingual NLI model, used as-is (decision log) | [MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) |

## Alternatives worth evaluating (not in the plan)

| Model | What it could replace or add | Caveats |
| --- | --- | --- |
| [Docling](https://github.com/docling-project/docling) (IBM, MIT licence) | PDF/DOCX/PPTX/image → structured Markdown/JSON with layout, reading order and tables; plugs in Tesseract, EasyOCR or RapidOCR ([report](https://arxiv.org/pdf/2408.09869)) | Heavier than PyMuPDF; would change the ingestion design in §6.1, so it needs the guide's approval |
| [Surya](https://github.com/datalab-to/surya) | OCR + layout + reading order in 90+ languages, including Hindi / Devanagari | Model weights have a modified OpenRAIL-M licence; recent versions expect a GPU (vLLM) or llama.cpp; compare against Tesseract on the H4 OCR curve before switching |
| [GLiNER](https://github.com/urchade/GLiNER) (multilingual variant) | Zero-shot entity extraction (names, dates, survey/GR numbers) for the "extract info" idea | Outside the plan's research core; results on Hindi/Marathi are unverified |
