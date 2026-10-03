"""Language and script identification with AI4Bharat IndicLID (plan sections 6.2 step 4 and 6.3).

IndicLID has two FastText models, used as-is: IndicLID-FTN for text in native scripts and
IndicLID-FTR for romanized text. Text whose letters are mostly Latin goes to FTR, the rest
to FTN. Sakshya's scope is English, Hindi and Marathi, so the scores are also reported
restricted to those three languages ("scores"); Hindi and Marathi are both kept, which is
what the router and the answer-fidelity check need when the two are close.

IndicLID's third stage, an IndicBERT classifier for low-confidence romanized text (969 MB), is
not used. In its place, romanized questions with a weak FTR prediction are settled by a short
list of romanized Hindi and Marathi function words ("hai", "ke liye", "ahe", "kiti"),
which also decides whether a Latin-script question is English or romanized Hindi/Marathi.

Models: https://github.com/AI4Bharat/IndicLID/releases (v1.0), MIT licence. They are
downloaded to DATA_DIR/models/indiclid on first use (169 MB + 229 MB).
"""

import logging
import re
import threading
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .. import config

log = logging.getLogger("sakshya.langid")

MODEL_DIR = config.MODEL_DIR / "indiclid"
RELEASE = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0"
MODELS = {"ftn": "indiclid-ftn", "ftr": "indiclid-ftr"}
IN_SCOPE = {"eng_Latn": "en", "hin_Deva": "hi", "mar_Deva": "mr", "hin_Latn": "hi", "mar_Latn": "mr"}
FTR_CONFIDENT = 0.6  # IndicLID's own threshold for sending romanized text on to its BERT stage

# Common romanized function words. Overlapping words ("nahi", "ka", "pan") count for both.
HINDI_WORDS = set("""hai hain ho hoga hogi ke ki ka ko se me mein liye kitne kitna kitni kya kaise kab kahan kaun
kyon kyun aur nahi nahin bhi toh tha thi the chahiye wala wali wale sakte sakta sakti karna karne karte diya gaya
raha rahi rahe apne unke uske iska iske yeh ye woh wo agar lekin par tak saath sath milta milti milega jata jati
jate hota hoti hote""".split())
MARATHI_WORDS = set("""ahe aahe aahet ahet kiti ani aani sathi saathi kay kasa kashi kase kuthe kevha konta konti
konte kontya kon madhe madhye mdhe la chi cha che chya tar var pan nahi naahi hote hota hoti kara karave karavi
karayla karaycha lagtat lagte milte miltat milel dyave mhanje pahije asel aste astat karta kartat zala zali jhala
keli kele amhi tumhi aapan yanchi yancha tyanchi tyancha hyachi hyacha""".split())
_LATIN_WORD = re.compile(r"[A-Za-z]+")


@dataclass
class LangResult:
    lang: str  # en / hi / mr / other
    label: str  # IndicLID's own label, e.g. "mar_Latn"
    confidence: float  # IndicLID's probability for `label`
    script: str  # "Deva" or "Latn" (which model was used), or "none"
    romanized: bool  # Latin script, Hindi or Marathi
    scores: dict = field(default_factory=dict)  # restricted to en / hi / mr, summing to 1
    model: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _letters(text: str) -> tuple[int, int]:
    deva = sum(1 for c in text if "ऀ" <= c <= "ॿ" and c.isalpha() or "ा" <= c <= "्")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return deva, latin


def _download(name: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    archive = MODEL_DIR / f"{name}.zip"
    if not archive.exists():
        log.info("Downloading %s from the IndicLID release (first use)", name)
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(f"{RELEASE}/{name}.zip", tmp)
        tmp.rename(archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(MODEL_DIR)


def model_path(kind: str):
    folder = MODEL_DIR / MODELS[kind]
    found = sorted(folder.glob("*.bin")) if folder.exists() else []
    if not found:
        _download(MODELS[kind])
        found = sorted(folder.glob("*.bin"))
    return found[0]


def models_present() -> bool:
    return all((MODEL_DIR / MODELS[k]).exists() and any((MODEL_DIR / MODELS[k]).glob("*.bin")) for k in MODELS)


class IndicLID:
    name = "IndicLID (FTN + FTR, v1.0)"

    def __init__(self) -> None:
        import fasttext

        fasttext.FastText.eprint = lambda *_: None  # silence the load-time warning
        self.ftn = fasttext.load_model(str(model_path("ftn")))
        self.ftr = fasttext.load_model(str(model_path("ftr")))
        self._lock = threading.Lock()

    @staticmethod
    def _all(model, text: str) -> dict[str, float]:
        # model.f.predict is fasttext's own predictor; the Python wrapper's predict() breaks on NumPy 2.
        pairs = model.f.predict(text, -1, 0.0, "replace")
        return {label.replace("__label__", ""): float(p) for p, label in pairs}

    def scores(self, text: str, script: str) -> dict[str, float]:
        clean = " ".join(text.split())
        if script == "Latn":
            clean = clean.lower()
        with self._lock:
            return self._all(self.ftr if script == "Latn" else self.ftn, clean)

    def identify(self, text: str) -> LangResult:
        deva, latin = _letters(text)
        if deva + latin == 0:
            return LangResult("other", "other", 0.0, "none", False, {}, self.name)
        script = "Latn" if latin > deva else "Deva"
        raw = self.scores(text, script)
        label = max(raw, key=raw.get)
        restricted = {code: 0.0 for code in ("en", "hi", "mr")}
        for lab, p in raw.items():
            if lab in IN_SCOPE and (script == "Deva" or lab.endswith("_Latn")):
                restricted[IN_SCOPE[lab]] += p
        total = sum(restricted.values())
        restricted = {k: round(v / total, 4) for k, v in restricted.items()} if total > 0 else restricted

        if script == "Deva":
            lang = max(("hi", "mr"), key=lambda k: restricted[k]) if total > 0 else "other"
            return LangResult(lang, label, round(raw[label], 4), script, False, restricted, self.name)

        words = [w.lower() for w in _LATIN_WORD.findall(text)]
        hi_hits = sum(w in HINDI_WORDS for w in words)
        mr_hits = sum(w in MARATHI_WORDS for w in words)
        confident = raw[label] >= FTR_CONFIDENT and label in IN_SCOPE
        if confident:
            lang = IN_SCOPE[label]
        elif hi_hits or mr_hits:
            lang = "mr" if mr_hits + restricted["mr"] > hi_hits + restricted["hi"] else "hi"
        else:
            lang = max(restricted, key=restricted.get) if total > 0 else "en"
        if lang == "en" and (hi_hits + mr_hits) >= 2:  # "ke liye ... hai": English words, Hindi grammar
            lang = "mr" if mr_hits > hi_hits else "hi"
        return LangResult(lang, label, round(raw[label], 4), script, lang in ("hi", "mr"), restricted, self.name)

    def is_english_word(self, word: str) -> bool:
        """For the code-mix ratio: an English word inside a romanized Hindi/Marathi question."""
        w = word.lower()
        if w in HINDI_WORDS or w in MARATHI_WORDS:
            return False
        if len(word) >= 2 and word.isupper():
            return True  # acronyms: ATKT, KYC, GR
        raw = self.scores(w, "Latn")
        en = raw.get("eng_Latn", 0.0)
        indic = raw.get("hin_Latn", 0.0) + raw.get("mar_Latn", 0.0)
        return en > indic and en >= 0.3


@lru_cache(maxsize=1)
def get_langid() -> IndicLID:
    return IndicLID()


def status() -> dict:
    return {"model": IndicLID.name, "present": models_present(), "loaded": get_langid.cache_info().currsize > 0}
