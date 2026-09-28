"""Detect Devanagari text stored in legacy (pre-Unicode) fonts.

Many Indian government PDFs and Word files were typed in fonts such as
Kruti Dev, DVB-TT Surekh or Shivaji. These fonts draw Devanagari glyphs on
ASCII / Latin-1 code points, so the page looks right but its text layer is
nonsense: "भारत का संविधान" is stored as "Hkkjr dk lafo/kku" (Kruti Dev) or
"£ÉÉ®iÉ BÉEÉ ºÉÆÉÊ´ÉvÉÉxÉ" (DVB-TT Surekh). Such a text layer is not usable
(plan section 6.1 step 1), so those pages go to OCR instead.

Three signals, any of which marks text as legacy:
1. the font name is a known legacy Devanagari font;
2. many Latin-1 Supplement characters (É, º, ®, ...) and almost no Unicode Devanagari (DVB/Shree style);
3. Kruti Dev spellings of very common Hindi/Marathi words ("dk" = का, "gS" = है, "esa" = में).
"""

import re

LEGACY_FONT_NAME = re.compile(
    r"kruti|dvb|dv-?tt|surekh|shivaji|yogesh|chanakya|walkman|aps-?dv|shusha|devlys|akruti|"
    r"shree[-_ ]?dev|shree[-_ ]?mar|agra|amar ?ujala|marathi[-_ ]?saral|priyanka|sanskrit ?99|ism-?dv",
    re.IGNORECASE,
)

# Kruti Dev spellings of frequent Hindi and Marathi words. English words that look the
# same (for example "us", "is") are left out on purpose.
KRUTI_WORDS = {
    "dk", "ds", "dh", "dks", "gS", "gSa", "esa", "ls", "vkSj", "fd", "ij", "Fkk", "Fkh", "Fks",
    "Hkh", "rks", "tks", "gks", "gksxk", "fd;k", "fy,", ";g", ";k", "vkfn", "ljdkj", "Hkkjr",
    "gksrs", "djus", "djsa", "vkgs", "vkf.k", "dsys", ";kaps", "gksrh",
}

LATIN1_SHARE = 0.15   # DVB-style text is full of Latin-1 Supplement characters
KRUTI_SHARE = 0.04    # share of words that are Kruti Dev spellings of common words
MIN_KRUTI_HITS = 5
DEVANAGARI_SHARE = 0.05  # text with more real Devanagari than this is already Unicode


def _devanagari_share(chars: list[str]) -> float:
    return sum(1 for c in chars if "ऀ" <= c <= "ॿ") / len(chars) if chars else 0.0


def looks_legacy(text: str) -> bool:
    """True when text looks like Devanagari typed in a legacy font (by content alone)."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 20 or _devanagari_share(chars) > DEVANAGARI_SHARE:
        return False
    latin1 = sum(1 for c in chars if " " <= c <= "ÿ") / len(chars)
    if latin1 >= LATIN1_SHARE:
        return True
    words = text.split()
    hits = sum(1 for w in words if w.rstrip("A").rstrip(",.") in KRUTI_WORDS)
    return hits >= MIN_KRUTI_HITS and hits / len(words) >= KRUTI_SHARE


def is_legacy_font(name: str) -> bool:
    # Drop the "ABCDEF+" subset prefix that PDF writers add to embedded font names.
    return bool(LEGACY_FONT_NAME.search(name.split("+", 1)[-1]))


def legacy_spans(page) -> tuple[int, int, set[str]]:
    """(characters in legacy-font spans, all characters, legacy font names) for a PyMuPDF page.

    A span in a legacy-named font whose text is already mostly Unicode Devanagari is not
    counted: some newer fonts keep the old name but use Unicode.
    """
    legacy_chars = total = 0
    fonts: set[str] = set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"]
                n = len(text.strip())
                total += n
                if n and is_legacy_font(span["font"]):
                    chars = [c for c in text if not c.isspace()]
                    if _devanagari_share(chars) <= 0.5:
                        legacy_chars += n
                        fonts.add(span["font"].split("+", 1)[-1])
    return legacy_chars, total, fonts
