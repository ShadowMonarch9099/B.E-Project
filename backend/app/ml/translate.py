"""Translation to English (plan sections 6.3 and 6.4): the question's English view, and the
English shadow index of non-English chunks for the translate route.

The plan names AI4Bharat IndicTrans2 (distilled, for CPU). Its Hugging Face weights are
gated: they need an account that accepts the terms and an HF_TOKEN. Until one is set, the
stand-in is Meta's NLLB-200 distilled 600M (CC-BY-NC 4.0: fine for this non-commercial
academic project), run with CTranslate2 in int8. Measured on this laptop (2026-10-03):
OPUS-MT mr->en invented unrelated sentences for 5 of 5 Marathi test sentences, while NLLB
translated them sensibly; in PyTorch NLLB took 5-11 s per sentence, with CTranslate2 int8
0.5-0.9 s (decision log, plan section 16).

Every translation records which model produced it, so results from different translators
are never mixed silently. SAKSHYA_TRANSLATOR: auto (default: IndicTrans2 if it loads, else
NLLB), indictrans2, nllb, or off. Both translate sentence by sentence.
"""

import logging
import os
import re
import threading

from .. import config

from .loading import load_once

log = logging.getLogger("sakshya.translate")

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_DIR = config.MODEL_DIR / "nllb-200-distilled-600M-ct2"
NLLB_TAGS = {"mr": "mar_Deva", "hi": "hin_Deva"}
INDICTRANS2_MODEL = os.environ.get("SAKSHYA_INDICTRANS2_MODEL", "ai4bharat/indictrans2-indic-en-dist-200M")
INDICTRANS2_TAGS = {"mr": "mar_Deva", "hi": "hin_Deva"}
BEAM = int(os.environ.get("SAKSHYA_TRANSLATE_BEAM", "4"))
BATCH = 16
MAX_LENGTH = 256
_SENT = re.compile(r"(?<=[.?!।॥])\s+")


def split_sentences(text: str) -> list[str]:
    """Sentences, for translation. Paragraphs are split at blank lines and their lines joined
    first: an OCR'd or PDF line is usually a piece of a sentence, not a whole one."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        joined = " ".join(line.strip() for line in para.splitlines() if line.strip())
        out.extend(p.strip() for p in _SENT.split(joined) if p.strip())
    return out


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


class NllbTranslator:
    """NLLB-200 distilled 600M with CTranslate2 (int8). Converted from the HF weights on first use."""

    name = "nllb"

    def __init__(self) -> None:
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        hf = snapshot_download(NLLB_MODEL, allow_patterns=["*.json", "pytorch_model.bin", "sentencepiece.bpe.model"])
        if not (NLLB_DIR / "model.bin").exists():
            log.info("Converting %s to CTranslate2 (first use)", NLLB_MODEL)
            NLLB_DIR.parent.mkdir(parents=True, exist_ok=True)
            ctranslate2.converters.TransformersConverter(hf).convert(str(NLLB_DIR), quantization="int8_float32",
                                                                     force=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf)
        self.translator = ctranslate2.Translator(str(NLLB_DIR), device="cpu", compute_type="int8_float32",
                                                 intra_threads=max(1, min(8, os.cpu_count() or 4)))
        self._lock = threading.Lock()

    def _run(self, sentences: list[str], lang: str) -> list[str]:
        tok = self.tokenizer
        tok.src_lang = NLLB_TAGS.get(lang, "hin_Deva")
        sources = [tok.convert_ids_to_tokens(tok.encode(s, truncation=True, max_length=MAX_LENGTH)) for s in sentences]
        results = self.translator.translate_batch(sources, target_prefix=[["eng_Latn"]] * len(sources),
                                                  beam_size=BEAM, max_decoding_length=MAX_LENGTH,
                                                  max_batch_size=BATCH)
        return [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]), skip_special_tokens=True)
                for r in results]

    def to_english(self, texts: list[str], lang: str) -> list[str]:
        with self._lock:
            return _by_sentence(texts, lambda sents: self._run(sents, lang))

    def label(self, lang: str) -> str:
        return f"{NLLB_MODEL} (CTranslate2 int8)"


class IndicTrans2Translator:
    """AI4Bharat IndicTrans2 (distilled 200M, Indic -> English). Needs HF access (gated)."""

    name = "indictrans2"

    def __init__(self, name: str = INDICTRANS2_MODEL) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = snapshot_download(name, token=os.environ.get("HF_TOKEN"))  # raises if access is not granted
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True).eval()
        self._lock = threading.Lock()
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
                gen = self.model.generate(**enc, num_beams=BEAM, max_length=MAX_LENGTH, use_cache=True)
                decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
                out.extend(detok.detokenize(d.split()) for d in decoded)
        return out

    def to_english(self, texts: list[str], lang: str) -> list[str]:
        with self._lock:
            return _by_sentence(texts, lambda sents: self._run(sents, lang))

    def label(self, lang: str) -> str:
        return self._model_name


class NoTranslator:
    name = "off"

    def to_english(self, texts: list[str], lang: str) -> list[str]:
        return ["" for _ in texts]

    def label(self, lang: str) -> str:
        return "off"


@load_once
def get_translator():
    choice = os.environ.get("SAKSHYA_TRANSLATOR", "auto")
    if choice == "off":
        return NoTranslator()
    if choice == "indictrans2" or (choice == "auto" and os.environ.get("HF_TOKEN")):
        try:
            return IndicTrans2Translator()
        except Exception as exc:
            if choice == "indictrans2":
                raise
            log.warning("IndicTrans2 could not be loaded (%s); using NLLB-200. Accept the model's terms on "
                        "Hugging Face for the account whose HF_TOKEN is set.", type(exc).__name__)
    return NllbTranslator()


def status() -> dict:
    loaded = get_translator.cache_info().currsize > 0
    return {"setting": os.environ.get("SAKSHYA_TRANSLATOR", "auto"), "loaded": loaded,
            "engine": get_translator().name if loaded else None, "hf_token_set": bool(os.environ.get("HF_TOKEN"))}
