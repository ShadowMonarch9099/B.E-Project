"""Translation to English (plan sections 6.3 and 6.4): the question's English view, and the
English shadow index of non-English chunks for the translate route.

The plan names AI4Bharat IndicTrans2 (distilled, for CPU). Its Hugging Face weights are
gated: they need an account that accepts the terms and an HF_TOKEN. Until one is set,
Helsinki-NLP OPUS-MT (Apache-2.0, ungated) stands in (decision log, 2026-10-03). Every
translation records which model produced it, so results from the two are never mixed
silently. SAKSHYA_TRANSLATOR: auto (default: IndicTrans2 if it loads, else OPUS-MT),
indictrans2, opus-mt, or off.

Both models translate sentence by sentence: OPUS-MT in particular degrades on long inputs.
"""

import logging
import os
import re
import threading
from functools import lru_cache

log = logging.getLogger("sakshya.translate")

OPUS_MODELS = {"mr": "Helsinki-NLP/opus-mt-mr-en", "hi": "Helsinki-NLP/opus-mt-hi-en"}
INDICTRANS2_MODEL = os.environ.get("SAKSHYA_INDICTRANS2_MODEL", "ai4bharat/indictrans2-indic-en-dist-200M")
INDICTRANS2_TAGS = {"mr": "mar_Deva", "hi": "hin_Deva"}
BATCH = 16
MAX_LENGTH = 256
_SENT = re.compile(r"(?<=[.?!।॥])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for line in text.splitlines() for p in _SENT.split(line)]
    return [p for p in parts if p]


class OpusMtTranslator:
    """Helsinki-NLP OPUS-MT Marian models, one per source language."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self._models: dict = {}
        self._lock = threading.Lock()
        self.name = "opus-mt"

    def model_name(self, lang: str) -> str:
        return OPUS_MODELS.get(lang, OPUS_MODELS["hi"])

    def _load(self, lang: str):
        name = self.model_name(lang)
        if name not in self._models:
            from huggingface_hub import snapshot_download
            from transformers import MarianMTModel, MarianTokenizer

            path = snapshot_download(name, allow_patterns=["*.json", "pytorch_model.bin", "*.spm"])
            self._models[name] = (MarianTokenizer.from_pretrained(path), MarianMTModel.from_pretrained(path).eval())
        return self._models[name]

    def _run(self, sentences: list[str], lang: str) -> list[str]:
        tok, model = self._load(lang)
        out: list[str] = []
        with self._torch.inference_mode():
            for i in range(0, len(sentences), BATCH):
                enc = tok(sentences[i:i + BATCH], return_tensors="pt", padding=True, truncation=True,
                          max_length=MAX_LENGTH)
                gen = model.generate(**enc, num_beams=4, max_new_tokens=MAX_LENGTH)
                out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def to_english(self, texts: list[str], lang: str) -> list[str]:
        with self._lock:
            return _by_sentence(texts, lambda sents: self._run(sents, lang))

    def label(self, lang: str) -> str:
        return self.model_name(lang)


class IndicTrans2Translator:
    """AI4Bharat IndicTrans2 (distilled 200M, Indic -> English). Needs HF access (gated)."""

    def __init__(self, name: str = INDICTRANS2_MODEL) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = snapshot_download(name, token=os.environ.get("HF_TOKEN"))  # raises if not granted
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True).eval()
        self._lock = threading.Lock()
        self.name = "indictrans2"
        self._model_name = name

    @staticmethod
    def _preprocess(sentence: str, lang: str) -> str:
        # The minimal form of IndicTransToolkit's preprocessing for Devanagari sources:
        # IndicNLP normalisation and tokenisation, then the language tags the model expects.
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize.indic_tokenize import trivial_tokenize

        norm = IndicNormalizerFactory().get_normalizer(lang).normalize(sentence)
        return f"{INDICTRANS2_TAGS[lang]} eng_Latn {' '.join(trivial_tokenize(norm, lang))}"

    def _run(self, sentences: list[str], lang: str) -> list[str]:
        from sacremoses import MosesDetokenizer

        detok = MosesDetokenizer(lang="en")
        lang = lang if lang in INDICTRANS2_TAGS else "hi"
        out: list[str] = []
        with self._torch.inference_mode():
            for i in range(0, len(sentences), BATCH):
                batch = [self._preprocess(s, lang) for s in sentences[i:i + BATCH]]
                enc = self.tokenizer(batch, return_tensors="pt", padding="longest", truncation=True,
                                     max_length=MAX_LENGTH)
                gen = self.model.generate(**enc, num_beams=4, max_length=MAX_LENGTH, use_cache=True)
                decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
                out.extend(detok.detokenize(d.split()) for d in decoded)
        return out

    def to_english(self, texts: list[str], lang: str) -> list[str]:
        with self._lock:
            return _by_sentence(texts, lambda sents: self._run(sents, lang))

    def label(self, lang: str) -> str:
        return self._model_name


def _by_sentence(texts: list[str], run) -> list[str]:
    """Translate each text sentence by sentence, in one batch, and join the results."""
    pieces = [split_sentences(t) for t in texts]
    flat = [s for p in pieces for s in p]
    translated = run(flat) if flat else []
    out, pos = [], 0
    for p in pieces:
        out.append(" ".join(translated[pos:pos + len(p)]))
        pos += len(p)
    return out


class NoTranslator:
    name = "off"

    def to_english(self, texts: list[str], lang: str) -> list[str]:
        return ["" for _ in texts]

    def label(self, lang: str) -> str:
        return "off"


@lru_cache(maxsize=1)
def get_translator():
    choice = os.environ.get("SAKSHYA_TRANSLATOR", "auto")
    if choice == "off":
        return NoTranslator()
    if choice in ("auto", "indictrans2"):
        try:
            return IndicTrans2Translator()
        except Exception as exc:
            if choice == "indictrans2":
                raise
            log.info("IndicTrans2 not available (%s); using OPUS-MT. Set HF_TOKEN after accepting the "
                     "model's terms on Hugging Face to use IndicTrans2.", type(exc).__name__)
    return OpusMtTranslator()


def status() -> dict:
    loaded = get_translator.cache_info().currsize > 0
    choice = os.environ.get("SAKSHYA_TRANSLATOR", "auto")
    return {"setting": choice, "loaded": loaded, "engine": get_translator().name if loaded else None,
            "indictrans2_token": bool(os.environ.get("HF_TOKEN"))}
