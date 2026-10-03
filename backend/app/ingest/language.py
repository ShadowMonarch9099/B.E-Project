"""Language of each chunk, and the English shadow index (plan section 6.2 steps 4 and 8).

Language ID runs on every chunk, our text block: IndicLID-FTN for Devanagari, IndicLID-FTR
for Latin script. The label is English, Hindi or Marathi, and the scores keep both Hindi and
Marathi (they share a script, and short or OCR'd passages are often close). A page's language
is the most common language of its chunks, weighted by length.

The English shadow index serves the translate route (section 6.4): every Hindi or Marathi
chunk is translated to English (IndicTrans2, or NLLB-200 until IndicTrans2 is available) and
embedded with the same E5 model as view "en"; English chunks are copied as they are. It is
built after the document is already searchable, because translation is the slowest step of
ingestion on a CPU. SAKSHYA_SHADOW_INDEX=off skips it (the translate route then falls back
to the direct route).
"""

import logging
import os
from collections import Counter
from collections.abc import Callable

from sqlalchemy import select

from ..db import Chunk, Page, session_scope
from ..index import dense
from ..retrieval.query import MODELS_ON

log = logging.getLogger("sakshya.language")

SHADOW_ON = os.environ.get("SAKSHYA_SHADOW_INDEX", "on") != "off"
TRANSLATE_BATCH = 2  # chunks per translation call; small, so a question never waits long (app/busy.py)


def _script_only(script: str | None) -> tuple[str | None, dict]:
    """Without IndicLID (unit tests): Latin text is English; Devanagari stays undecided."""
    if script == "latin":
        return "en", {"en": 1.0, "hi": 0.0, "mr": 0.0}
    return None, {}


def identify_chunks(doc_id: str, on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Set lang / lang_scores on every chunk and lang on every page; returns the language mix."""
    with session_scope() as s:
        chunks = s.scalars(select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.seq)).all()
        items = [(c.id, c.text_raw, c.script, c.page_id, c.token_count) for c in chunks]
    lid = None
    if MODELS_ON:
        from ..ml.langid import get_langid

        lid = get_langid()
    results: dict[int, tuple[str | None, dict]] = {}
    for i, (cid, text, script, _, _) in enumerate(items, start=1):
        if lid is not None:
            r = lid.identify(text)
            results[cid] = (r.lang if r.lang != "other" else None, r.scores)
        else:
            results[cid] = _script_only(script)
        if on_progress and (i % 20 == 0 or i == len(items)):
            on_progress(i, len(items))

    mix: Counter = Counter()
    page_votes: dict[int, Counter] = {}
    for cid, _, _, page_id, tokens in items:
        lang, _ = results[cid]
        if lang:
            mix[lang] += tokens or 1
            page_votes.setdefault(page_id, Counter())[lang] += tokens or 1
    with session_scope() as s:
        for c in s.scalars(select(Chunk).where(Chunk.doc_id == doc_id)):
            c.lang, c.lang_scores = results.get(c.id, (None, {}))
        for p in s.scalars(select(Page).where(Page.doc_id == doc_id)):
            votes = page_votes.get(p.id)
            p.lang = votes.most_common(1)[0][0] if votes else None
    total = sum(mix.values())
    return {lang: round(n / total, 3) for lang, n in mix.most_common()} if total else {}


def needs_language(doc_id: str) -> bool:
    with session_scope() as s:
        return s.scalar(select(Chunk.id).where(Chunk.doc_id == doc_id, Chunk.lang.is_(None),
                                               Chunk.script.is_not(None)).limit(1)) is not None


def build_shadow(doc_id: str, on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Translate the document's non-English chunks to English and embed them as view "en"."""
    from ..ml.translate import get_translator

    with session_scope() as s:
        rows = s.execute(select(Chunk.id, Chunk.text_raw, Chunk.lang, Chunk.script)
                         .where(Chunk.doc_id == doc_id).order_by(Chunk.seq)).all()
    # Devanagari (or mixed) Hindi/Marathi chunks are translated; everything else (English, and
    # Latin-script tables or number-only rows, whatever language IndicLID gave them) is copied as it is.
    indic = [(r.id, r.text_raw, r.lang if r.lang in ("hi", "mr") else "hi") for r in rows
             if r.script in ("devanagari", "mixed") and r.lang != "en"]
    indic_ids = {cid for cid, _, _ in indic}
    english = [(r.id, r.text_raw) for r in rows if r.id not in indic_ids]
    translator = get_translator() if indic else None
    done, total = 0, len(indic)
    translated: dict[int, tuple[str, str]] = {cid: (text, "source") for cid, text in english}
    from .. import busy

    for i in range(0, len(indic), TRANSLATE_BATCH):
        busy.wait_until_idle()  # questions first
        batch = indic[i:i + TRANSLATE_BATCH]
        by_lang: dict[str, list] = {}
        for item in batch:
            by_lang.setdefault(item[2], []).append(item)
        for lang, items in by_lang.items():
            outs = translator.to_english([t for _, t, _ in items], lang)
            for (cid, _, _), out in zip(items, outs):
                translated[cid] = (out, translator.label(lang))
        done += len(batch)
        if on_progress:
            on_progress(done, total)
    with session_scope() as s:
        for c in s.scalars(select(Chunk).where(Chunk.doc_id == doc_id)):
            if c.id in translated:
                c.text_en, c.text_en_model = translated[c.id]
    vectors = dense.embed_chunks(doc_id, view="en", texts_of=lambda c: c.text_en)
    models = sorted({m for _, m in translated.values() if m != "source"})
    return {"state": "ready", "translated": len(indic), "copied_english": len(english), "vectors": vectors,
            "model": models[0] if len(models) == 1 else models or None}


def needs_shadow(doc_id: str) -> bool:
    """True when some chunk of the document has no English-view vector for the current model."""
    if not SHADOW_ON:
        return False
    from sqlalchemy import func

    from ..db import Embedding
    from ..ml.embedder import get_embedder

    with session_scope() as s:
        chunks = s.scalar(select(func.count(Chunk.id)).where(Chunk.doc_id == doc_id, Chunk.text_raw != ""))
        vectors = s.scalar(select(func.count()).select_from(Embedding).where(
            Embedding.doc_id == doc_id, Embedding.view == "en", Embedding.model == get_embedder().name))
    return (vectors or 0) < (chunks or 0)
