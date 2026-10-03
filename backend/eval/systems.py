"""The systems an evaluation compares: Sakshya, each fixed route, the baselines of plan
section 9.2 and the ablations of section 9.3 that can be measured at retrieval level.

Answer-level ablations (NLI verifier, script repair) need the answer model and run through
`python -m eval answers`. "Router -> best fixed route" is read off the per-route rows
(the report names the best fixed route on the split). E5-base vs E5-large needs the
collection re-embedded with SAKSHYA_EMBED_MODEL=intfloat/multilingual-e5-large and a second run.
"""

from dataclasses import asdict, dataclass

from app.retrieval.query import AnalysisOptions


@dataclass(frozen=True)
class System:
    name: str
    group: str  # sakshya / route / baseline / ablation
    label: str
    plan: str
    route: str | None = None  # None: the rule-based router chooses
    reranker: bool = True
    gate: bool = True
    ocr_aware: bool = True
    transliterate: bool = True
    translate: bool = True
    digits: bool = True
    numbers: bool = True

    def options(self, models_on: bool) -> AnalysisOptions:
        return AnalysisOptions(langid=models_on, transliterate=models_on and self.transliterate,
                               translate=models_on and self.translate, digits=self.digits, numbers=self.numbers)

    def as_dict(self) -> dict:
        return asdict(self)


SYSTEMS: dict[str, System] = {s.name: s for s in [
    System("sakshya", "sakshya", "Sakshya (router, all components)", "6.4-6.7"),
    System("route:direct", "route", "Direct route, full stack", "9.2 (4)", route="direct"),
    System("route:translate", "route", "Translate route, full stack", "9.2 (4)", route="translate"),
    System("route:hybrid", "route", "Hybrid route, full stack", "9.2 (4)", route="hybrid"),
    System("base:dense", "baseline", "Vanilla dense: E5 only, no reranker, no gate", "9.2 (1)",
           route="direct", reranker=False, gate=False),
    System("base:translate", "baseline", "Translate-only: no reranker, no gate", "9.2 (2)",
           route="translate", reranker=False, gate=False),
    System("base:hybrid", "baseline", "Hybrid without reranking and without the gate", "9.2 (3)",
           route="hybrid", reranker=False, gate=False),
    System("abl:-xlit", "ablation", "No transliteration view", "9.3", transliterate=False),
    System("abl:-digits", "ablation", "No digit normalization (question side)", "9.3", digits=False),
    System("abl:-ocr-aware", "ablation", "No OCR-aware weighting", "9.3", ocr_aware=False),
    System("abl:-reranker", "ablation", "No reranker (fused order; the gate still scores the top 6)", "9.3",
           reranker=False),
    System("abl:-gate", "ablation", "No evidence gate", "9.3", gate=False),
]}

GROUPS = {
    "all": list(SYSTEMS),
    "rq1": ["sakshya", "route:direct", "route:translate", "route:hybrid"],
    "baselines": ["sakshya", "base:dense", "base:translate", "base:hybrid", "route:direct", "route:translate",
                  "route:hybrid"],
    "ablations": ["sakshya", "route:direct", "route:translate", "route:hybrid", "abl:-xlit", "abl:-digits",
                  "abl:-ocr-aware", "abl:-reranker", "abl:-gate"],
}


def resolve(spec: str) -> list[System]:
    names: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        for name in GROUPS.get(part, [part]):
            if name not in SYSTEMS:
                raise SystemExit(f"unknown system {name!r}; known: {', '.join(SYSTEMS)} or groups {', '.join(GROUPS)}")
            if name not in names:
                names.append(name)
    if "sakshya" in names:  # first, so its latencies are measured before any cache is warm
        names.remove("sakshya")
        names.insert(0, "sakshya")
    return [SYSTEMS[n] for n in names]
