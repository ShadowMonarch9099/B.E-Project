"""Roman -> Devanagari transliteration with AI4Bharat IndicXlit (plan section 6.3, N2).

IndicXlit's official weights (ai4bharat/IndicXlit on Hugging Face, MIT licence) are a fairseq
checkpoint, and fairseq does not install on Python 3.11 / Windows. This module runs the same
weights in plain PyTorch (decision log, 2026-10-03). The checkpoint is a standard fairseq
`transformer` (6 + 6 layers, d=256, 4 heads, FFN 1024, pre-norm, embedding layer norm,
sinusoidal positions, GELU), trained with `translation_multi_simple_epoch` and
`encoder_langtok=tgt`; it is read exactly as fairseq would:

- dictionaries: fairseq's <s>=0, <pad>=1, </s>=2, <unk>=3, then dict.<lang>.txt in file
  order, then the language tokens __en__, __as__, ... in lang_list.txt order;
- source: "__mr__ g h a r k u l </s>" (the official pre-processing: lower case, one token
  per character, the target language token first);
- decoding: beam search (beam 5, length penalty 1, the checkpoint's generation settings),
  starting from </s>, top hypothesis only. The official engine can also rescore candidates
  with word-frequency tables (850 MB); that step is not used here.
"""

import logging
import math
import re
import threading
from functools import lru_cache

from .loading import load_once

log = logging.getLogger("sakshya.xlit")

REPO = "ai4bharat/IndicXlit"
PREFIX = "indicxlit-en-indic-v1.0"
LANG_LIST = ["en", "as", "bn", "brx", "gom", "gu", "hi", "kn", "ks", "mai", "ml", "mni", "mr", "ne", "or",
             "pa", "sa", "sd", "si", "ta", "te", "ur"]  # models/en2indic/lang_list.txt
BEAM = 5
MAX_LEN = 60
_WORD = re.compile(r"[A-Za-z]+")


def _load_dict(path: str) -> list[str]:
    symbols = ["<s>", "<pad>", "</s>", "<unk>"]
    with open(path, encoding="utf8") as f:
        for line in f:
            sym = line.rsplit(" ", 1)[0] if " " in line.strip() else line.strip()
            if sym:
                symbols.append(sym)
    return symbols + [f"__{lang}__" for lang in LANG_LIST]


class _Transformer:
    """Inference-only fairseq transformer with pre-norm layers."""

    def __init__(self, state: dict, heads: int):
        import torch

        self.t = torch
        self.sd = {k: v.float() for k, v in state.items() if v.dtype.is_floating_point}
        self.heads = heads
        self.dim = self.sd["encoder.embed_tokens.weight"].shape[1]
        self.scale = math.sqrt(self.dim)
        self.layers = 1 + max(int(k.split(".")[2]) for k in self.sd if k.startswith("encoder.layers."))

    def _ln(self, x, name):
        return self.t.nn.functional.layer_norm(x, (self.dim,), self.sd[f"{name}.weight"], self.sd[f"{name}.bias"])

    def _lin(self, x, name):
        return self.t.nn.functional.linear(x, self.sd[f"{name}.weight"], self.sd.get(f"{name}.bias"))

    def _positions(self, length: int, offset: int = 0):
        # fairseq SinusoidalPositionalEmbedding; positions start at padding_idx + 1 = 2.
        t = self.t
        half = self.dim // 2
        freq = t.exp(t.arange(half, dtype=t.float) * -(math.log(10000) / (half - 1)))
        pos = t.arange(2 + offset, 2 + offset + length, dtype=t.float).unsqueeze(1) * freq.unsqueeze(0)
        return t.cat([t.sin(pos), t.cos(pos)], dim=1)

    def _attn(self, x, kv, name, key_pad=None, causal=False):
        t = self.t
        b, lq, d = x.shape
        lk = kv.shape[1]
        h, hd = self.heads, d // self.heads
        q = self._lin(x, f"{name}.q_proj") * hd ** -0.5
        k = self._lin(kv, f"{name}.k_proj")
        v = self._lin(kv, f"{name}.v_proj")
        q = q.view(b, lq, h, hd).transpose(1, 2)
        k = k.view(b, lk, h, hd).transpose(1, 2)
        v = v.view(b, lk, h, hd).transpose(1, 2)
        scores = q @ k.transpose(-1, -2)
        if key_pad is not None:
            scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
        if causal:
            mask = t.triu(t.ones(lq, lk, dtype=t.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, lq, d)
        return self._lin(out, f"{name}.out_proj")

    def _ffn(self, x, name):
        return self._lin(self.t.nn.functional.gelu(self._lin(x, f"{name}.fc1")), f"{name}.fc2")

    def encode(self, tokens):
        x = self.scale * self.sd["encoder.embed_tokens.weight"][tokens] + self._positions(tokens.shape[1])
        x = self._ln(x, "encoder.layernorm_embedding")
        for i in range(self.layers):
            p = f"encoder.layers.{i}"
            y = self._ln(x, f"{p}.self_attn_layer_norm")
            x = x + self._attn(y, y, f"{p}.self_attn")
            x = x + self._ffn(self._ln(x, f"{p}.final_layer_norm"), p)
        return self._ln(x, "encoder.layer_norm")

    def decode_last(self, prev, enc):
        """Log-probabilities of the next token for each prefix in `prev` (batch, length)."""
        x = self.scale * self.sd["decoder.embed_tokens.weight"][prev] + self._positions(prev.shape[1])
        x = self._ln(x, "decoder.layernorm_embedding")
        for i in range(self.layers):
            p = f"decoder.layers.{i}"
            y = self._ln(x, f"{p}.self_attn_layer_norm")
            x = x + self._attn(y, y, f"{p}.self_attn", causal=True)
            x = x + self._attn(self._ln(x, f"{p}.encoder_attn_layer_norm"), enc, f"{p}.encoder_attn")
            x = x + self._ffn(self._ln(x, f"{p}.final_layer_norm"), p)
        x = self._ln(x[:, -1], "decoder.layer_norm")
        return self.t.log_softmax(self._lin(x, "decoder.output_projection"), dim=-1)


class IndicXlit:
    name = "ai4bharat/IndicXlit (en-indic v1.0)"

    def __init__(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        def get(path: str) -> str:
            return hf_hub_download(REPO, f"{PREFIX}/{path}")

        ck = torch.load(get("transformer/indicxlit.pt"), map_location="cpu", weights_only=False)
        model_cfg = ck["cfg"]["model"]
        self.src_symbols = _load_dict(get("corpus-bin/dict.en.txt"))
        self.tgt_symbols = _load_dict(get("corpus-bin/dict.mr.txt"))  # one joint Indic dictionary
        self.src_index = {s: i for i, s in enumerate(self.src_symbols)}
        assert len(self.src_symbols) == ck["model"]["encoder.embed_tokens.weight"].shape[0]
        assert len(self.tgt_symbols) == ck["model"]["decoder.embed_tokens.weight"].shape[0]
        self.net = _Transformer(ck["model"], heads=model_cfg.encoder_attention_heads)
        n_lang = len(LANG_LIST)
        self._blocked = [0, 1, 3] + list(range(len(self.tgt_symbols) - n_lang, len(self.tgt_symbols)))
        self._torch = torch
        self._lock = threading.Lock()

    def _word(self, word: str, lang: str) -> str:
        t = self._torch
        ids = [self.src_index[f"__{lang}__"]] + [self.src_index.get(ch, 3) for ch in word.lower()] + [2]
        enc = self.net.encode(t.tensor([ids]))
        beams = [([2], 0.0)]  # (tokens starting with </s>, summed log-prob)
        finished: list[tuple[list[int], float]] = []
        for _ in range(MAX_LEN):
            prev = t.tensor([b[0] for b in beams])
            logp = self.net.decode_last(prev, enc.expand(len(beams), -1, -1))
            logp[:, self._blocked] = float("-inf")  # <s>, <pad>, <unk> and language tokens; </s> stays
            cand = []
            for bi, (toks, score) in enumerate(beams):
                top = t.topk(logp[bi], BEAM)
                for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((toks + [tok], score + lp))
            cand.sort(key=lambda c: -c[1])
            beams = []
            for toks, score in cand:
                if toks[-1] == 2:
                    finished.append((toks, score / (len(toks) - 1)))  # length penalty 1
                elif len(beams) < BEAM:
                    beams.append((toks, score))
                if len(beams) == BEAM:
                    break
            if len(finished) >= BEAM or not beams:
                break
        if not finished:
            finished = [(b[0] + [2], b[1] / len(b[0])) for b in beams]
        best = max(finished, key=lambda f: f[1])[0]
        return "".join(self.tgt_symbols[i] for i in best[1:-1] if i > 3 and not self.tgt_symbols[i].startswith("__"))

    def transliterate(self, word: str, lang: str = "mr") -> str:
        """One romanized word to Devanagari (lang "mr" or "hi")."""
        if not _WORD.fullmatch(word):
            return word
        with self._lock, self._torch.inference_mode():
            return self._word(word, lang)


@load_once
def get_xlit() -> IndicXlit:
    return IndicXlit()


@lru_cache(maxsize=4096)
def transliterate_word(word: str, lang: str) -> str:
    return get_xlit().transliterate(word, lang)
