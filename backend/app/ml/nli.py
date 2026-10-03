"""Claim verification with a pretrained multilingual NLI model (plan section 6.7 step 3), optional.

MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT), used as-is: the cited
passage is the premise, the answer sentence the hypothesis. A sentence is kept when its
entailment probability against at least one cited passage reaches SAKSHYA_NLI_THRESHOLD.
Off by default (SAKSHYA_NLI=on to enable): it adds a 558 MB model and about 2 s per answer
on the team's laptop. Spot check (2026-10-03, passage "the annual income ... must not exceed
three lakh rupees"): English "the limit is three lakh" 0.41 vs "five lakh" 0.005, so it
catches English contradictions; Marathi "तीन लाख" 0.43 vs "पाच लाख" 0.42, so it cannot tell
them apart (XNLI has Hindi but not Marathi, as plan section 11 warns). Correct claims scored
0.35-0.43, below 0.5, hence the default threshold of 0.3.
"""

import logging
import os
import threading

from .loading import load_once

log = logging.getLogger("sakshya.nli")

NLI_MODEL = os.environ.get("SAKSHYA_NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
NLI_ON = os.environ.get("SAKSHYA_NLI", "off") == "on"
THRESHOLD = float(os.environ.get("SAKSHYA_NLI_THRESHOLD", "0.3"))
HF_FILES = ["*.json", "model.safetensors", "spm.model"]


class NliModel:
    def __init__(self, name: str = NLI_MODEL) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = snapshot_download(name, allow_patterns=HF_FILES)
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]
        self._torch = torch
        self._lock = threading.Lock()

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(entailment) for each (premise, hypothesis) pair."""
        torch = self._torch
        out: list[float] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(pairs), 8):
                batch = pairs[i:i + 8]
                enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                                     max_length=512, padding=True, return_tensors="pt")
                probs = self.model(**enc).logits.softmax(-1)[:, self.entail]
                out.extend(probs.tolist())
        return out


@load_once
def get_nli() -> NliModel:
    log.info("Loading %s", NLI_MODEL)
    return NliModel()


def status() -> dict:
    return {"enabled": NLI_ON, "model": NLI_MODEL, "threshold": THRESHOLD,
            "loaded": get_nli.cache_info().currsize > 0}
