"""The Sakshya-Bench document manifest and its splits (plan sections 8.1 and 8.2).

backend/bench/manifest.json lists the public documents: source URL, publisher, licence,
language(s), whether scanned, and (once fetched) SHA-256 and page count. The documents
themselves stay out of git (data/bench/docs); the manifest is enough to fetch them again.

Splits are by DOCUMENT, so no test document is seen while tuning anything: about 50% train,
15% dev and 35% test, assigned within each (language, scanned) group so every split gets a
similar mix. `freeze` records the assignment and its hash; after that the test split never
changes, and documents added later can only go to train or dev.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import config

MANIFEST = config.BACKEND_DIR / "bench" / "manifest.json"
DOCS_DIR = config.DATA_DIR / "bench" / "docs"
PROPORTIONS = {"train": 0.50, "dev": 0.15, "test": 0.35}
SPLITS = ("train", "dev", "test")


def load(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def save(manifest: dict, path: Path = MANIFEST) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def _key(doc_id: str) -> str:
    return hashlib.sha256(doc_id.encode()).hexdigest()


def _group(doc: dict) -> tuple:
    langs = doc.get("languages") or ["unknown"]
    return (langs[0], bool(doc.get("scanned")))


def assignment_hash(manifest: dict) -> str:
    pairs = sorted((d["id"], d.get("split")) for d in manifest["documents"])
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16]


def assign_splits(manifest: dict) -> dict:
    """Give every document without a split one. Before freezing, all documents are
    (re)assigned; after it, existing splits are kept and new documents go to train or dev."""
    frozen = manifest.get("frozen")
    docs = manifest["documents"]
    if frozen:
        new = [d for d in docs if not d.get("split")]
        for d in new:  # deterministic: about 3 train for each dev, never test
            d["split"] = "dev" if int(_key(d["id"])[:8], 16) % 100 < 23 else "train"
        return manifest
    groups: dict[tuple, list[dict]] = {}
    for d in docs:
        groups.setdefault(_group(d), []).append(d)
    for members in groups.values():
        members.sort(key=lambda d: _key(d["id"]))
        n = len(members)
        n_test = round(n * PROPORTIONS["test"])
        n_dev = round(n * PROPORTIONS["dev"])
        if n >= 3 and n_test == 0:
            n_test = 1
        for i, d in enumerate(members):
            d["split"] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
    return manifest


def freeze(manifest: dict) -> dict:
    if manifest.get("frozen"):
        raise ValueError(f"The splits were frozen on {manifest['frozen']['at']}; they cannot be re-frozen.")
    if any(not d.get("split") for d in manifest["documents"]):
        raise ValueError("Assign splits before freezing.")
    manifest["frozen"] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hash": assignment_hash(manifest),
        "test": sorted(d["id"] for d in manifest["documents"] if d["split"] == "test"),
    }
    return manifest


def check_frozen(manifest: dict) -> None:
    """Raise if a frozen test split was changed afterwards."""
    frozen = manifest.get("frozen")
    if not frozen:
        return
    now = sorted(d["id"] for d in manifest["documents"] if d.get("split") == "test")
    if now != frozen["test"]:
        raise ValueError("The test split differs from the one frozen on " + frozen["at"])


def summary(manifest: dict) -> dict:
    docs = manifest["documents"]
    by_split = {s: [d for d in docs if d.get("split") == s] for s in SPLITS}
    langs: dict[str, int] = {}
    for d in docs:
        for lang in d.get("languages") or ["unknown"]:
            langs[lang] = langs.get(lang, 0) + 1
    known = [d for d in docs if d.get("scanned") is not None]
    return {
        "documents": len(docs),
        "fetched": sum(1 for d in docs if d.get("sha256")),
        "by_split": {s: len(v) for s, v in by_split.items()},
        "unassigned": sum(1 for d in docs if not d.get("split")),
        "languages": langs,
        "scanned_share": round(sum(1 for d in known if d["scanned"]) / len(known), 3) if known else None,
        "frozen": manifest.get("frozen"),
    }


def split_of_sha(manifest: dict) -> dict[str, tuple[str, str | None]]:
    """sha256 -> (manifest id, split), for documents that have been fetched."""
    return {d["sha256"]: (d["id"], d.get("split")) for d in manifest["documents"] if d.get("sha256")}
