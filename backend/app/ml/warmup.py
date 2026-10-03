"""Load the models in the background at startup, so the first question does not wait for them.

IndicLID's two FastText models alone take about 35 s to load on the team's laptop. The
server answers health checks and uploads while this runs; a question asked before it
finishes simply loads what it needs itself.
"""

import logging
import os
import threading
import time

log = logging.getLogger("sakshya.warmup")

_state: dict = {"started": False, "done": False, "loaded": [], "failed": {}}


def _load_all() -> None:
    from ..retrieval.query import MODELS_ON

    steps = [("embedder", "app.ml.embedder", "get_embedder"), ("reranker", "app.ml.reranker", "get_reranker")]
    if MODELS_ON:
        steps += [("langid", "app.ml.langid", "get_langid"), ("xlit", "app.ml.xlit", "get_xlit"),
                  ("translator", "app.ml.translate", "get_translator")]
    from .nli import NLI_ON

    if NLI_ON:
        steps.append(("nli", "app.ml.nli", "get_nli"))
    import importlib

    for name, module, fn in steps:
        t = time.perf_counter()
        try:
            getattr(importlib.import_module(module), fn)()
            _state["loaded"].append(name)
            log.info("Loaded %s in %.1fs", name, time.perf_counter() - t)
        except Exception as exc:  # a model that cannot load is reported in /api/health
            _state["failed"][name] = f"{type(exc).__name__}: {str(exc)[:200]}"
            log.warning("Could not load %s: %s", name, exc)
    _state["done"] = True


def start() -> None:
    if _state["started"] or os.environ.get("SAKSHYA_TEST_MODE") == "1":
        return
    _state["started"] = True
    threading.Thread(target=_load_all, name="model-warmup", daemon=True).start()


def status() -> dict:
    return {k: v for k, v in _state.items() if k != "started"}
