"""Engine, sessions and migrations.

The schema is created and upgraded with Alembic (backend/migrations) when the app starts.
If the database cannot be reached, the app still starts: detection and OCR keep working,
and the document endpoints answer 503 with a setup hint.
"""

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .. import config

log = logging.getLogger("sakshya.db")

_state = {"ready": False, "error": None}


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = config.DATABASE_URL
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # ON DELETE CASCADE needs foreign keys on
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10,
                         connect_args={"connect_timeout": 5})


@lru_cache(maxsize=1)
def _factory() -> sessionmaker:
    return sessionmaker(get_engine(), expire_on_commit=False)


def is_postgres() -> bool:
    return get_engine().dialect.name == "postgresql"


def migrate() -> None:
    """Upgrade the schema to the latest migration."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(config.BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(config.BACKEND_DIR / "migrations"))
    command.upgrade(cfg, "head")


def init_db() -> bool:
    """Connect and migrate. Returns False (and remembers why) if the database is unreachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        migrate()
    except Exception as exc:  # unreachable server, wrong credentials, ...
        _state.update(ready=False, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        log.warning("Database not available: %s", _state["error"])
        return False
    _state.update(ready=True, error=None)
    return True


def status() -> dict:
    url = get_engine().url
    return {"available": _state["ready"], "dialect": url.get_backend_name(), "error": _state["error"]}


def is_ready() -> bool:
    return _state["ready"]


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session
