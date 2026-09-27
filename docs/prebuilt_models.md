# Prebuilt models for document detection and extraction

This note lists prebuilt, open models that fit the stages of the Review 2 plan (`docs/Review2_Plan.md`). The plan's choices come first; alternatives are marked as such and are **not** adopted unless the team and the guide agree (plan §11: no scope creep).

## In use now

| Stage | Model | What it does | Notes |
| --- | --- | --- | --- |
| File type detection (§6.1 step 1) | [Google Magika](https://github.com/google/magika) `standard_v3_3` | Deep-learning content-type detector: ~1 MB model, 200+ types, runs on CPU in milliseconds ([paper](https://arxiv.org/pdf/2409.13768)) | Used as-is from `pip install magika`. **Finding:** it labels plain Devanagari (Hindi/Marathi) text as "unknown", so the backend adds a UTF-8 text check before falling back to the file extension. |

Everything else in the current app is a plain parser, not a model: PyMuPDF (PDF pages and text-layer check), python-docx, Pillow and `zipfile`.

## Next stages, from the plan

| Plan stage | Model named in the plan | Link |
| --- | --- | --- |
| OCR for scans (§6.1 step 2) | Tesseract 5 with `hin`, `mar`, `eng` | [tesseract-ocr](https://github.com/tesseract-ocr/tesseract) |
| Language + script ID per block (§6.1 step 4) | AI4Bharat IndicLID | [AI4Bharat/IndicLID](https://github.com/AI4Bharat/IndicLID) |
| Transliteration of romanized queries (§6.2) | AI4Bharat IndicXlit | [AI4Bharat/IndicXlit](https://github.com/AI4Bharat/IndicXlit) |
| English shadow index, translate route (§6.1, §6.3) | AI4Bharat IndicTrans2 (distilled for CPU) | [AI4Bharat/IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) |
| Dense retrieval (§6.1) | multilingual-E5 (base / large) | [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) |

## Alternatives worth evaluating (not in the plan)

| Model | What it could replace or add | Caveats |
| --- | --- | --- |
| [Docling](https://github.com/docling-project/docling) (IBM, MIT licence) | PDF/DOCX/PPTX/image → structured Markdown/JSON with layout, reading order and tables; plugs in Tesseract, EasyOCR or RapidOCR ([report](https://arxiv.org/pdf/2408.09869)) | Heavier than PyMuPDF; would change the ingestion design in §6.1, so it needs the guide's approval |
| [Surya](https://github.com/datalab-to/surya) | OCR + layout + reading order in 90+ languages, including Hindi / Devanagari | Model weights have a modified OpenRAIL-M licence; recent versions expect a GPU (vLLM) or llama.cpp; compare against Tesseract on the H4 OCR curve before switching |
| [GLiNER](https://github.com/urchade/GLiNER) (multilingual variant) | Zero-shot entity extraction (names, dates, survey/GR numbers) for the "extract info" idea | Outside the plan's research core; results on Hindi/Marathi are unverified |
