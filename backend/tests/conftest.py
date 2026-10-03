"""Test settings: a throwaway database and data folder, set before the app is imported.

By default the tests use SQLite, so they need no database server. To run them against
PostgreSQL + pgvector instead, point SAKSHYA_TEST_DATABASE_URL at a test database whose
name ends in "_test"; its tables are dropped and recreated at the start of the run:
    postgresql+psycopg://sakshya:sakshya@localhost:5433/sakshya_test
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="sakshya-test-")
_pg = os.environ.get("SAKSHYA_TEST_DATABASE_URL")
os.environ["SAKSHYA_DATA_DIR"] = _tmp
# Models stay where the app downloads them (<repo>/data/models), not in the throwaway folder.
os.environ.setdefault("SAKSHYA_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "models"))
os.environ["SAKSHYA_DATABASE_URL"] = _pg or f"sqlite:///{_tmp}/sakshya-test.db".replace("\\", "/")
# Unit tests never call the Claude API; the live smoke test reads the real key itself.
os.environ.setdefault("SAKSHYA_TEST_MODE", "1")
# Unit tests use small stand-ins instead of downloading the 1 GB+ models; tests of the real
# models run when the model is already in the Hugging Face cache (see test_indexes.py).
if os.environ.get("SAKSHYA_TEST_REAL_MODELS") != "1":
    os.environ["SAKSHYA_EMBEDDER"] = "hash"
    os.environ["SAKSHYA_RERANKER"] = "overlap"
    os.environ["SAKSHYA_QUERY_MODELS"] = "off"  # IndicLID, IndicXlit, translation
    os.environ.setdefault("SAKSHYA_TRANSLATOR", "off")  # no 2.5 GB NLLB in unit tests
# The answer model is always the scripted fake in unit tests (no key, no cost); the live
# smoke test in test_answer.py builds the real provider itself.
os.environ["SAKSHYA_LLM_PROVIDER"] = "fake"

if _pg:
    from sqlalchemy import create_engine, make_url, text

    if not (make_url(_pg).database or "").endswith("_test"):
        raise RuntimeError("SAKSHYA_TEST_DATABASE_URL must name a database ending in _test")
    with create_engine(_pg).begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
