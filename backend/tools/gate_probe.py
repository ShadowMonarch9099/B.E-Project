"""Print the best reranker score of each probe question, to pick the evidence gate's tau.

Usage (from backend/, with the backend's database and the documents named in the file):
    .venv/bin/python tools/gate_probe.py                       # tools/gate_probe_questions.json
    .venv/bin/python tools/gate_probe.py my_questions.json --route direct

Prints one line per question and, at the end, the range of best scores for answerable and
unanswerable questions. This is a starting point only; the plan picks tau from the
risk-coverage curve on the Sakshya-Bench dev split (backend/eval).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db import Document, init_db, session_scope  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=str(Path(__file__).with_name("gate_probe_questions.json")))
    ap.add_argument("--route", default="hybrid")
    args = ap.parse_args()
    spec = json.loads(Path(args.file).read_text(encoding="utf8"))
    if not init_db():
        sys.exit("database not reachable")
    with session_scope() as s:
        ids = {d.filename: d.id for d in s.scalars(select(Document).where(Document.status == "indexed"))}
    best = {True: [], False: []}
    for item in spec["questions"]:
        filename = spec["documents"][item["doc"]]
        r = retrieve(item["q"], [ids[filename]], route=args.route)
        top = r.top[0] if r.top else None
        page = r.chunks[top.chunk_id].page_no if top else None
        score = r.gate.score
        best[item["answerable"]].append(score)
        print(f"{'ANS' if item['answerable'] else 'OFF'}  {score:.3f}  p.{page}  "
              f"{r.latency_ms['retrieval_total']:6.0f} ms  {item['q']}")
    for flag, label in ((True, "answerable"), (False, "unanswerable")):
        scores = best[flag]
        if scores:
            print(f"{label:13} n={len(scores):2}  min={min(scores):.3f}  max={max(scores):.3f}")


if __name__ == "__main__":
    main()
