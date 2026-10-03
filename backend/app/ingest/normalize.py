"""Text normalization for indexing (plan section 6.2 step 5, section 6.3).

The original text (`text_raw`) is kept for display and citations. The index uses
`text_norm`: Unicode NFC, the IndicNLP normalizer for Devanagari (nukta, matra and
punctuation variants), Devanagari digits mapped to ASCII (१४२ -> 142, so a scanned GR's
survey number matches the "142" a user types), and collapsed whitespace.
"""

import re
import unicodedata
from functools import lru_cache

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ASCII_TO_DEVANAGARI = str.maketrans("0123456789", "०१२३४५६७८९")
_DEVA = re.compile(r"[ऀ-ॿ]")
_SPACES = re.compile(r"[ \t ​]+")


@lru_cache(maxsize=4)
def _normalizer(lang: str):
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    return IndicNormalizerFactory().get_normalizer(lang)


def to_ascii_digits(text: str) -> str:
    return text.translate(DEVANAGARI_DIGITS)


def to_devanagari_digits(text: str) -> str:
    return text.translate(ASCII_TO_DEVANAGARI)


def normalize(text: str, lang: str | None = None, digits: bool = True) -> str:
    """Index form of a passage or query. `lang` picks the IndicNLP normalizer (hi or mr)."""
    text = unicodedata.normalize("NFC", text)
    if _DEVA.search(text):
        text = _normalizer("mr" if lang == "mr" else "hi").normalize(text)
    if digits:
        text = to_ascii_digits(text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def script_of(text: str) -> str | None:
    """"devanagari", "latin", "mixed" (each at least 20% of the letters) or None (no letters)."""
    deva = latin = 0
    for ch in text:
        if "ऀ" <= ch <= "ॿ" and not ("।" <= ch <= "९"):  # not dandas or digits
            deva += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    total = deva + latin
    if total == 0:
        return None
    share = deva / total
    if share >= 0.8:
        return "devanagari"
    if share <= 0.2:
        return "latin"
    return "mixed"
