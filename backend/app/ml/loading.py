"""One lock for loading models.

Models are loaded lazily from several threads (the startup warm-up, background ingestion,
requests). Loading two at once from different threads raced inside `transformers`' lazy
imports ("cannot import name 'AutoModel'") and could build the same model twice, so every
loader runs under this lock and caches its result.
"""

import functools
import threading

_LOCK = threading.RLock()


def load_once(fn):
    """Like functools.lru_cache(maxsize=1) for a no-argument loader, but thread-safe."""
    cached = functools.lru_cache(maxsize=1)(fn)

    @functools.wraps(fn)
    def wrapper():
        if cached.cache_info().currsize:
            return cached()
        with _LOCK:
            return cached()

    wrapper.cache_info = cached.cache_info
    wrapper.cache_clear = cached.cache_clear
    return wrapper
