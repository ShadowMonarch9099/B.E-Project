"""Sakshya evaluation harness (plan section 9). Run from backend/ with the database running:

    .venv/bin/python -m eval run --split dev                 # retrieval: RQ1, baselines, ablations, abstention
    .venv/bin/python -m eval run --split test --systems rq1  # only once the splits are frozen
    .venv/bin/python -m eval tau --split dev --max-risk 0.1  # re-choose the gate threshold from risk-coverage
    .venv/bin/python -m eval noise --split dev               # Recall@20 under synthetic OCR noise (5-30% CER)
    .venv/bin/python -m eval known-item --n 100              # sanity check with sentences copied from pages
    .venv/bin/python -m eval cost --split dev                # what an answer run would send and cost
    .venv/bin/python -m eval answers --split dev --yes       # answer-level metrics (calls the answer model)
    .venv/bin/python -m eval report eval/results/<run>.json  # rewrite a run's Markdown report

Each run writes eval/results/<name>.json (every question, every system), <name>.md (the tables,
each captioned with the split and this command) and <name>.risk_coverage.svg. Retrieval runs
call no external API. Questions come from the database (verified only) or --questions <export>.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from app.db import init_db

from . import data, report, runtime
from .systems import resolve

RESULTS = Path(__file__).resolve().parent / "results"


def _out(name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"


def _meta(split: str | None, title: str, note: str | None = None, **extra) -> dict:
    m = data.check_split(split if split != "all" else None, extra.pop("allow_unfrozen", False))
    return {"title": title, "split": split, "command": runtime.command(), "commit": runtime.git_commit(),
            "date": runtime.now(), "machine": runtime.machine(), "manifest_frozen": (m.get("frozen") or {}).get("hash"),
            "note": note, **extra}


def _save(result: dict, path: Path, resamples: int) -> None:
    result["meta"]["peak_rss_mb"] = runtime.peak_rss_mb()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=list), encoding="utf8")
    summary = report.write(result, path.with_suffix(".md"), resamples)
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=list),
                                                 encoding="utf8")
    print(f"wrote {path}, {path.with_suffix('.md').name}, {path.with_suffix('.risk_coverage.svg').name}")


def _need_db() -> None:
    if not init_db():
        sys.exit("The database is not reachable (docker compose up -d db).")


def cmd_run(args) -> None:
    from . import runner

    _need_db()
    meta = _meta(args.split, f"Sakshya evaluation: {args.split} split", allow_unfrozen=args.allow_unfrozen)
    questions = data.load_questions(args.questions, args.split)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        sys.exit(f"No verified questions in the {args.split} split. Annotate them at /annotate first "
                 "(nothing is reported without real questions).")
    systems = resolve(args.systems)
    print(f"{len(questions)} questions x {len(systems)} systems, scope {args.scope}")
    result = runner.run(questions, systems, args.scope)
    result["meta"] = {**meta, "systems": [s.as_dict() for s in systems]}
    _save(result, _out(f"{args.split}-{args.name}"), args.resamples)


def cmd_known_item(args) -> None:
    from . import known_item, runner

    _need_db()
    docs = data.documents()
    questions = known_item.sample(docs, args.n, args.seed, args.doc_type, args.translate)
    if not questions:
        sys.exit("No indexed Sakshya-Bench documents with usable sentences.")
    systems = resolve(args.systems)
    kind = "machine-translated to English" if args.translate else "copied from the page"
    note = (f"SANITY CHECK, NOT SAKSHYA-BENCH: {len(questions)} automatic known-item queries ({kind}, seed {args.seed}). "
            "A copied sentence must retrieve its own page. These are not questions a person asked and favour "
            "lexical matching; they only show that the harness, indexes and routes work end to end.")
    meta = _meta("all", "Known-item sanity check", note)
    print(f"{len(questions)} known-item queries x {len(systems)} systems")
    result = runner.run(questions, systems, args.scope)
    result["meta"] = {**meta, "systems": [s.as_dict() for s in systems], "known_item": True}
    _save(result, _out("known-item" + ("-mt" if args.translate else "")), args.resamples)


def cmd_noise(args) -> None:
    from sqlalchemy import select

    from app.db import Chunk, session_scope
    from app.retrieval.query import analyse_query

    from . import known_item, metrics, noise

    _need_db()
    docs = data.documents()
    native = {d["doc_id"] for d in docs.values() if d["manifest_id"] and d["doc_type"] == "native"}
    if args.queries == "known-item":
        qs = known_item.sample({k: v for k, v in docs.items() if v["doc_id"] in native}, args.n, args.seed, "native")
        note = "Queries are automatic known-item sentences (SANITY CHECK, NOT SAKSHYA-BENCH)."
    else:
        data.check_split(args.split if args.split != "all" else None, args.allow_unfrozen)
        qs = [q for q in data.load_questions(args.questions, args.split) if q.get("answerable")]
        note = f"Answerable Sakshya-Bench questions of the {args.split} split whose document is native."
    items = []
    for q in qs:
        gold = data.gold_pages(q, docs)
        doc = docs.get(q.get("sha256") or "")
        if gold and doc and doc["doc_id"] in native:
            items.append({"id": q["id"], "question": q["question"], "doc_id": doc["doc_id"], "gold": gold})
    if not items:
        sys.exit("No questions over native documents to corrupt.")
    with session_scope() as s:
        rows = s.execute(select(Chunk.doc_id, Chunk.page_no, Chunk.text_norm)
                         .where(Chunk.doc_id.in_({i["doc_id"] for i in items})).order_by(Chunk.id)).all()
    chunks = [{"doc_id": r.doc_id, "page_no": r.page_no, "text": r.text_norm} for r in rows][:args.max_chunks]
    analyses = {i["question"]: analyse_query(i["question"]) for i in items}
    levels = [float(x) for x in args.levels.split(",")]
    print(f"{len(items)} questions over {len(chunks)} chunks; CER levels {levels}")
    out = noise.stress(items, chunks, levels, analyses, seed=args.seed)
    meta = _meta(args.split if args.queries == "bench" else "all", "OCR-noise stress test", note,
                 allow_unfrozen=args.allow_unfrozen)
    meta["peak_rss_mb"] = runtime.peak_rss_mb()
    path = _out("noise-" + args.queries)
    path.write_text(json.dumps({"meta": meta, "levels": out, "questions": len(items), "chunks": len(chunks)},
                               ensure_ascii=False, indent=1), encoding="utf8")
    names = list(next(iter(out.values()))["recall@20"])
    lines = [f"# OCR-noise stress test", "", f"> {note}", "", f"- **Command:** `{meta['command']}`",
             f"- **Commit:** {meta['commit']} · {meta['date']} · {len(items)} questions, {len(chunks)} chunks", "",
             "| Target CER | Measured CER | " + " | ".join(names) + " |", "| --- | --- | " + " | ".join("---" for _ in names) + " |"]
    for lv in out.values():
        lines.append(f"| {lv['target_cer']:.0%} | {lv['measured_cer']:.1%} | "
                     + " | ".join(f"{lv['recall@20'][n]:.3f}" for n in names) + " |")
    lines += ["", "Recall@20 at page level, first stage only (no reranker), each question within its own document. "
              "OCR-aware weighting gives every chunk an OCR confidence of 1 - CER."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


def cmd_tau(args) -> None:
    from . import metrics

    result = json.loads(Path(args.results).read_text(encoding="utf8"))
    if result["meta"].get("split") == "test":
        sys.exit("tau is chosen on the dev split, never on test.")
    qs = [q for q in result["questions"] if not q.get("skipped") and args.system in q["systems"]]
    conf = [q["systems"][args.system]["gate_score"] for q in qs]
    err = [(not q["answerable"]) or not q["systems"][args.system].get("evidence_in_top") for q in qs]
    best = metrics.choose_tau(conf, err, args.max_risk)
    print(json.dumps({"split": result["meta"].get("split"), "system": args.system, "questions": len(qs),
                      "max_risk": args.max_risk, **best,
                      "current_tau": result["config"]["tau"],
                      "at_current_tau": metrics.operating_point(conf, err, result["config"]["tau"])}, indent=2))


def cmd_cost(args) -> None:
    from . import answers

    _need_db()
    data.check_split(args.split if args.split != "all" else None, args.allow_unfrozen)
    qs = data.load_questions(args.questions, args.split)[:args.limit or None]
    print(json.dumps(answers.estimate(qs, data.documents()), indent=2, ensure_ascii=False))


def cmd_answers(args) -> None:
    from . import answers

    _need_db()
    meta = _meta(args.split, f"Sakshya answers: {args.split} split", allow_unfrozen=args.allow_unfrozen)
    qs = data.load_questions(args.questions, args.split)[:args.limit or None]
    if not qs:
        sys.exit(f"No verified questions in the {args.split} split.")
    est = answers.estimate(qs, data.documents())
    print(json.dumps(est, indent=2, ensure_ascii=False))
    if not est["provider"].get("configured"):
        sys.exit("No answer provider is configured (add a key to backend/.env).")
    if not args.yes:
        sys.exit("Nothing was sent. Check the estimate above, then re-run with --yes to call the answer model.")
    rows = answers.run(qs, args.scope, args.route, args.pause)
    summary = answers.summarize(rows)
    path = _out(f"answers-{args.split}")
    meta["peak_rss_mb"] = runtime.peak_rss_mb()
    path.write_text(json.dumps({"meta": meta, "estimate": est, "summary": summary, "questions": rows},
                               ensure_ascii=False, indent=1), encoding="utf8")
    answers.rating_sheet(rows, path.with_suffix(".rating.csv"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {path} and {path.with_suffix('.rating.csv').name} (for human ratings)")


def cmd_report(args) -> None:
    path = Path(args.results)
    result = json.loads(path.read_text(encoding="utf8"))
    report.write(result, path.with_suffix(".md"), args.resamples)
    print(f"wrote {path.with_suffix('.md')}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m eval", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, split_default="dev"):
        p.add_argument("--split", default=split_default, choices=["train", "dev", "test", "all"])
        p.add_argument("--questions", help="a JSONL export instead of the database")
        p.add_argument("--allow-unfrozen", action="store_true", help="dry runs only; never reported")
        p.add_argument("--limit", type=int)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--systems", default="all")
    p.add_argument("--scope", default="document", choices=["document", "collection"])
    p.add_argument("--name", default="retrieval")
    p.add_argument("--resamples", type=int, default=1000)
    p = sub.add_parser("known-item")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--doc-type", choices=["native", "scanned"])
    p.add_argument("--translate", action="store_true")
    p.add_argument("--systems", default="rq1")
    p.add_argument("--scope", default="document", choices=["document", "collection"])
    p.add_argument("--resamples", type=int, default=1000)
    p = sub.add_parser("noise")
    common(p)
    p.add_argument("--queries", default="bench", choices=["bench", "known-item"])
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--levels", default="0,0.05,0.1,0.2,0.3")
    p.add_argument("--max-chunks", type=int, default=3000)
    p = sub.add_parser("tau")
    p.add_argument("results")
    p.add_argument("--system", default="sakshya")
    p.add_argument("--max-risk", type=float, default=0.1)
    p = sub.add_parser("cost")
    common(p)
    p = sub.add_parser("answers")
    common(p)
    p.add_argument("--scope", default="document", choices=["document", "collection"])
    p.add_argument("--route", choices=["direct", "translate", "hybrid"])
    p.add_argument("--pause", type=float, default=0.0, help="seconds between questions (free-tier rate limits)")
    p.add_argument("--yes", action="store_true", help="really call the answer model")
    p = sub.add_parser("report")
    p.add_argument("results")
    p.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()
    {"run": cmd_run, "known-item": cmd_known_item, "noise": cmd_noise, "tau": cmd_tau, "cost": cmd_cost,
     "answers": cmd_answers, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
