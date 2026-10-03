# Evaluation reports

Reports produced by `backend/eval` (plan §9), each with the exact command that produced it. Every table in a report is captioned with its split, its number of questions and that command.

**No Sakshya-Bench results exist yet.** The benchmark's questions have not been annotated (plan §8.2, open item 6). The reports here are **sanity checks** of the harness, the indexes and the routes, run on automatic queries. They are not findings and must not be quoted as Sakshya's accuracy.

| Report | What it is | Command |
| --- | --- | --- |
| `known-item-sanity.md` (+ `.risk_coverage.svg`) | 100 sentences copied from random pages of the 67 indexed documents; each must retrieve its own page. Copied sentences favour lexical matching, so retrieval numbers are optimistic, and there are no unanswerable queries, so abstention cannot be judged. The translate route was left out because the English shadow index was still being built. | `python -m eval known-item --n 100 --systems sakshya,route:direct,route:hybrid,base:dense,base:hybrid,abl:-reranker,abl:-ocr-aware,abl:-digits,abl:-gate` |
| `ocr-noise-sanity.md` | Known-item queries from native pages (48 that fit in the chunk cap, from whole documents) against those pages with synthetic OCR noise at 0–30% target character error rate (4–25% measured), re-indexed in memory. Long copied sentences keep enough intact text at every level, so only word BM25 drops (1.00 → 0.96 Recall@20); this shows the test works, not that OCR-aware weighting helps, which needs short real questions. | `python -m eval noise --queries known-item --n 60 --max-chunks 1200 --levels 0,0.05,0.1,0.2,0.3` |

Both ran on the team laptop (Windows 11 on Snapdragon X Elite, x64 Python under emulation, 16 GB RAM) on 3–4 October 2026 (commit `f40402b` plus the noise-cap fix in this commit), **while the backend was translating the English shadow index in the background**, so the latencies are upper bounds.

When the questions exist: `python -m eval run --split dev` (all systems), `python -m eval tau eval/results/<run>.json`, then, once the splits are frozen, `python -m eval run --split test`. Copy the report here with its command.
