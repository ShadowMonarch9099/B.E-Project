"""Summaries and the Markdown report (plan section 9.4).

Every table is captioned with the split, the number of questions and the command that
produced it. Comparisons use paired bootstrap intervals over questions; a difference is
marked real only when its 95% interval excludes zero.
"""

import html
from collections import Counter
from pathlib import Path

from . import metrics
from .bootstrap import bootstrap_mean, paired_bootstrap
from .systems import SYSTEMS

RETRIEVAL = ("recall@5", "recall@20", "mrr@10", "ndcg@10")
BREAKDOWNS = ("slice", "lang", "doc_type", "qtype")
ROUTES = ("route:direct", "route:translate", "route:hybrid")


def _system_names(questions: list[dict]) -> list[str]:
    names: list[str] = []
    for q in questions:
        for name in q["systems"]:
            if name not in names:
                names.append(name)
    return names


def _error(rec: dict, answerable: bool) -> bool:
    """Retrieval-level risk: answering is an error when the question is unanswerable or the
    gold page is not among the passages that would go to the answer model."""
    return (not answerable) or not rec.get("evidence_in_top")


def summarize(result: dict, resamples: int = 1000) -> dict:
    qs = [q for q in result["questions"] if not q.get("skipped")]
    names = _system_names(qs)
    ans = [q for q in qs if q["answerable"]]
    out = {"n": len(qs), "answerable": len(ans), "unanswerable": len(qs) - len(ans),
           "skipped": Counter(q["skipped"] for q in result["questions"] if q.get("skipped")), "systems": {}}
    for name in names:
        rows = [q for q in qs if name in q["systems"]]
        arows = [q for q in rows if q["answerable"]]
        s: dict = {"label": SYSTEMS[name].label if name in SYSTEMS else name,
                   "group": SYSTEMS[name].group if name in SYSTEMS else "other", "n": len(rows)}
        for m in RETRIEVAL:
            s[m] = bootstrap_mean([q["systems"][name].get(m) for q in arows], resamples)
        s["breakdown"] = {}
        for key in BREAKDOWNS:
            groups: dict[str, list[dict]] = {}
            for q in arows:
                groups.setdefault(str(q.get(key) or "unknown"), []).append(q)
            s["breakdown"][key] = {g: {"n": len(v), **{m: metrics.mean(q["systems"][name].get(m) for q in v)
                                                       for m in ("recall@20", "mrr@10")}}
                                   for g, v in sorted(groups.items())}
        answered = [q["systems"][name]["answered"] for q in rows]
        s["abstention"] = metrics.abstention_pr([not a for a in answered], [q["answerable"] for q in rows])
        conf = [q["systems"][name]["gate_score"] for q in rows]
        err = [_error(q["systems"][name], q["answerable"]) for q in rows]
        rc = metrics.risk_coverage(conf, err)
        s["aurc"] = rc["aurc"]
        s["curve"] = [(round(c, 4), round(r, 4)) for c, r, _ in rc["points"]]
        answered_err = [e for e, a in zip(err, answered) if a]
        s["answered_share"] = sum(answered) / len(answered) if answered else None
        s["answered_risk"] = sum(answered_err) / len(answered_err) if answered_err else None
        s["routes"] = dict(Counter(q["systems"][name]["route"] for q in rows))
        if name != "sakshya" and "sakshya" in names:
            both = [q for q in arows if "sakshya" in q["systems"]]
            s["vs_sakshya"] = {m: paired_bootstrap([q["systems"]["sakshya"].get(m) for q in both],
                                                   [q["systems"][name].get(m) for q in both], resamples)
                               for m in ("recall@20", "mrr@10")}
        out["systems"][name] = s
    present = [r for r in ROUTES if r in names]
    if present:
        oracle = {m: [max(q["systems"][r].get(m) or 0.0 for r in present) for q in ans] for m in RETRIEVAL}
        out["oracle"] = {m: bootstrap_mean(v, resamples) for m, v in oracle.items()}
        out["best_fixed_route"] = max(present, key=lambda r: (out["systems"][r]["mrr@10"]["mean"] or 0.0))
    lat_rows = [q["systems"][names[0]].get("latency_ms") or {} for q in qs if names]
    out["latency_system"] = names[0] if names else None
    out["latency_ms"] = metrics.latency_summary([{k: v for k, v in r.items() if k != "analysis_cached"}
                                                 for r in lat_rows])
    return out


def _f(x, digits: int = 3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _ci(b: dict) -> str:
    if not b or b.get("mean") is None:
        return "–"
    return f"{b['mean']:.3f} [{b['low']:.3f}, {b['high']:.3f}]"


def _diff(b: dict) -> str:
    if not b or b.get("diff") is None:
        return "–"
    mark = " **real**" if b["real"] else ""
    return f"{b['diff']:+.3f} [{b['low']:+.3f}, {b['high']:+.3f}]{mark}"


def risk_coverage_svg(curves: dict[str, list[tuple[float, float]]], title: str, width: int = 560,
                      height: int = 360) -> str:
    """A plain SVG line chart: coverage on x, risk on y (both 0-1)."""
    pad_l, pad_r, pad_t, pad_b = 52, 150, 30, 44
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    dashes = ["", "6 3", "2 3", "8 3 2 3", "1 2", "4 4"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" '
             f'font-size="11"><rect width="{width}" height="{height}" fill="#fff"/>',
             f'<text x="{pad_l}" y="18" font-size="13">{html.escape(title)}</text>']
    for i in range(6):
        v = i / 5
        x, y = pad_l + v * w, pad_t + h - v * h
        parts.append(f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.1f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{pad_t + h + 16}" text-anchor="middle">{v:.1f}</text>')
    parts.append(f'<text x="{pad_l + w / 2}" y="{height - 8}" text-anchor="middle">coverage (share of questions answered)</text>')
    parts.append(f'<text x="14" y="{pad_t + h / 2}" text-anchor="middle" transform="rotate(-90 14 {pad_t + h / 2})">'
                 f'risk (share answered wrongly)</text>')
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{w}" height="{h}" fill="none" stroke="#111"/>')
    for i, (name, pts) in enumerate(curves.items()):
        if not pts:
            continue
        coords = " ".join(f"{pad_l + c * w:.1f},{pad_t + h - r * h:.1f}" for c, r in pts)
        dash = f' stroke-dasharray="{dashes[i % len(dashes)]}"' if dashes[i % len(dashes)] else ""
        parts.append(f'<polyline points="{coords}" fill="none" stroke="#111" stroke-width="1.6"{dash}/>')
        ly = pad_t + 14 + i * 16
        parts.append(f'<line x1="{pad_l + w + 10}" x2="{pad_l + w + 34}" y1="{ly}" y2="{ly}" stroke="#111" '
                     f'stroke-width="1.6"{dash}/>')
        parts.append(f'<text x="{pad_l + w + 38}" y="{ly + 4}">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def markdown(result: dict, summary: dict, svg_name: str | None = None) -> str:
    meta = result["meta"]
    split, cmd = meta.get("split"), meta.get("command")
    cap = f"_{split} split · n = {{n}} · `{cmd}`_"
    S = summary["systems"]
    lines = [f"# {meta.get('title', 'Sakshya evaluation')}", ""]
    if meta.get("note"):
        lines += [f"> {meta['note']}", ""]
    lines += [f"- **Split:** {split} · **scope:** {result['scope']} ({result['collection_size']} documents in the collection)",
              f"- **Questions:** {summary['n']} ({summary['answerable']} answerable, {summary['unanswerable']} unanswerable)"
              + (f"; skipped: {dict(summary['skipped'])}" if summary["skipped"] else ""),
              f"- **Command:** `{cmd}`",
              f"- **Commit:** {meta.get('commit')} · **date:** {meta.get('date')} · **machine:** "
              f"{meta.get('machine', {}).get('platform')} ({meta.get('machine', {}).get('cpus')} CPUs)",
              f"- **Config:** top-k {result['config']['top_k']}, top-n {result['config']['top_n']}, "
              f"tau {result['config']['tau']}, RRF weights {result['config']['weights']}",
              f"- **Peak RAM of the evaluation process:** {meta.get('peak_rss_mb')} MB · run time {result.get('seconds')} s",
              ""]
    n_ans = summary["answerable"]
    rq1 = [r for r in ("route:direct", "route:translate", "route:hybrid", "sakshya") if r in S]
    if rq1:
        lines += ["## RQ1: retrieval by route (answerable questions, page level)", "",
                  "| System | Recall@5 | Recall@20 | MRR@10 | nDCG@10 |", "| --- | --- | --- | --- | --- |"]
        for name in rq1:
            lines.append(f"| {S[name]['label']} | " + " | ".join(_ci(S[name][m]) for m in RETRIEVAL) + " |")
        if "oracle" in summary:
            lines.append("| Oracle (best route per question; upper bound for routing) | "
                         + " | ".join(_ci(summary["oracle"][m]) for m in RETRIEVAL) + " |")
        lines += ["", "Mean with a 95% bootstrap interval. " + cap.format(n=n_ans), ""]
        if summary.get("best_fixed_route"):
            lines += [f"Best fixed route on this split (by MRR@10): **{summary['best_fixed_route']}**.", ""]
        for key, title in (("lang", "question language"), ("doc_type", "document type"), ("slice", "slice"),
                           ("qtype", "question type")):
            groups = sorted({g for name in rq1 for g in S[name]["breakdown"][key]})
            if not groups:
                continue
            lines += [f"### By {title} (Recall@20 / MRR@10)", "",
                      "| System | " + " | ".join(f"{g} (n={S[rq1[0]]['breakdown'][key].get(g, {}).get('n', 0)})"
                                                for g in groups) + " |",
                      "| --- | " + " | ".join("---" for _ in groups) + " |"]
            for name in rq1:
                cells = []
                for g in groups:
                    b = S[name]["breakdown"][key].get(g)
                    cells.append(f"{_f(b['recall@20'], 2)} / {_f(b['mrr@10'], 2)}" if b else "–")
                lines.append(f"| {S[name]['label']} | " + " | ".join(cells) + " |")
            lines += ["", cap.format(n=n_ans), ""]
        if "sakshya" in S:
            share = S["sakshya"]["routes"]
            lines += ["Routes the router chose: " + ", ".join(f"{k} {v}" for k, v in sorted(share.items())) + ".", ""]
    groups = [("baseline", "Baselines (plan §9.2)"), ("ablation", "Ablations (plan §9.3)")]
    for group, title in groups:
        names = [n for n in S if S[n]["group"] == group]
        if not names:
            continue
        lines += [f"## {title}", "", "| System | Recall@20 | MRR@10 | Δ Recall@20 vs Sakshya | Δ MRR@10 vs Sakshya |",
                  "| --- | --- | --- | --- | --- |"]
        for name in names:
            vs = S[name].get("vs_sakshya", {})
            lines.append(f"| {S[name]['label']} | {_ci(S[name]['recall@20'])} | {_ci(S[name]['mrr@10'])} | "
                         f"{_diff(vs.get('recall@20'))} | {_diff(vs.get('mrr@10'))} |")
        lines += ["", "Δ is the paired bootstrap difference (1,000 resamples) with its 95% interval; **real** = "
                  "the interval excludes zero. " + cap.format(n=n_ans), ""]
    lines += ["## Abstention and risk–coverage", "",
              "Retrieval-level risk: answering counts as an error when the question is unanswerable or the gold page "
              "is not among the 6 passages sent to the answer model. Answer-level numbers need the answer model "
              "(`python -m eval answers`).", "",
              "| System | Answered | Risk when answered | Abstention precision | Abstention recall | Wrongly abstained | AURC |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in S:
        a = S[name]["abstention"]
        lines.append(f"| {S[name]['label']} | {_f(S[name]['answered_share'], 2)} | {_f(S[name]['answered_risk'], 2)} | "
                     f"{_f(a['precision'], 2)} | {_f(a['recall'], 2)} | {a['wrongly_abstained']} | {_f(S[name]['aurc'], 3)} |")
    lines += ["", cap.format(n=summary["n"]), ""]
    if svg_name:
        lines += [f"![Risk–coverage curve]({svg_name})", ""]
    if summary.get("latency_ms"):
        lines += [f"## Latency per stage ({summary['latency_system']}, ms)", "", "| Stage | p50 | p95 | n |",
                  "| --- | --- | --- | --- |"]
        for stage, v in summary["latency_ms"].items():
            lines.append(f"| {stage} | {_f(v['p50'], 0)} | {_f(v['p95'], 0)} | {v['n']} |")
        lines += ["", "Measured on the first system run for each question, before any cache for it was warm. "
                  + cap.format(n=summary["n"]), ""]
    return "\n".join(lines)


def write(result: dict, out_md: Path, resamples: int = 1000) -> dict:
    summary = summarize(result, resamples)
    svg_path = out_md.with_suffix(".risk_coverage.svg")
    curves = {name: s["curve"] for name, s in summary["systems"].items()
              if s["group"] in ("sakshya", "route", "baseline") or name == "abl:-gate"}
    svg_path.write_text(risk_coverage_svg(curves, f"Risk–coverage, {result['meta'].get('split')} split"),
                        encoding="utf8")
    out_md.write_text(markdown(result, summary, svg_path.name), encoding="utf8")
    return summary
