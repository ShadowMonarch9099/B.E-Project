# OCR-noise stress test

> Queries are automatic known-item sentences (SANITY CHECK, NOT SAKSHYA-BENCH).

- **Command:** `python -m eval noise --queries known-item --n 60 --max-chunks 1200 --levels 0,0.05,0.1,0.2,0.3`
- **Commit:** f40402b+uncommitted · 2026-10-03T19:23:12+00:00 · 48 questions, 1085 chunks

| Target CER | Measured CER | dense | word | char | hybrid | hybrid+ocr-aware |
| --- | --- | --- | --- | --- | --- | --- |
| 0% | 0.0% | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5% | 4.2% | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10% | 8.3% | 1.000 | 0.979 | 1.000 | 1.000 | 1.000 |
| 20% | 16.5% | 1.000 | 0.979 | 1.000 | 1.000 | 1.000 |
| 30% | 24.8% | 1.000 | 0.958 | 1.000 | 1.000 | 1.000 |

Recall@20 at page level, first stage only (no reranker), each question within its own document. OCR-aware weighting gives every chunk an OCR confidence of 1 - CER.
