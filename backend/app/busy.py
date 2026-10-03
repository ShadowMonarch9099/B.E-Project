"""Is a question being served right now? Background work (the English shadow index) waits.

Translating the shadow index uses every core, and NLLB is also needed to translate a Hindi or
Marathi question; with both running, answering took 25-70 s instead of a few seconds on the
team's laptop. Retrieval marks itself busy, and the shadow builder pauses between batches.
"""

import threading
import time
from contextlib import contextmanager

_active = 0
_lock = threading.Lock()


@contextmanager
def serving():
    global _active
    with _lock:
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1


def active() -> int:
    return _active


def wait_until_idle(poll_s: float = 0.2, max_wait_s: float = 300.0) -> None:
    """Block while a question is being served (at most max_wait_s, so background work never starves)."""
    end = time.monotonic() + max_wait_s
    while _active and time.monotonic() < end:
        time.sleep(poll_s)
