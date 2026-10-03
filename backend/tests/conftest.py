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
os.environ["SAKSHYA_DATABASE_URL"] = _pg or f"sqlite:///{_tmp}/sakshya-test.db".replace("\\", "/")
# Unit tests never call the Claude API; the live smoke test reads the real key itself.
os.environ.setdefault("SAKSHYA_TEST_MODE", "1")

if _pg:
    from sqlalchemy import create_engine, make_url, text

    if not (make_url(_pg).database or "").endswith("_test"):
        raise RuntimeError("SAKSHYA_TEST_DATABASE_URL must name a database ending in _test")
    with create_engine(_pg).begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
