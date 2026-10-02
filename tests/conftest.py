import pytest

from app import tracing
from app.config import Settings


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's environment.

    Some libraries (litellm) load .env into os.environ when imported, which would otherwise leak
    real passwords and ports into unrelated tests.
    """
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


@pytest.fixture(autouse=True)
def memory_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory extraction calls a model after every chat turn; tests opt in explicitly."""
    monkeypatch.setenv("MEMORY_ENABLED", "false")


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never send traces to Langfuse, even if the developer's .env has real keys."""
    monkeypatch.setattr(tracing, "get_client", lambda: None)


@pytest.fixture
def db_schema():  # type: ignore[no-untyped-def]
    """A throwaway schema in the dev database, with all migrations applied. Skips without a DB.

    Tests never touch real tables: the schema is created for the test and dropped after.
    """
    import uuid

    import psycopg

    from app.db.migrate import apply_migrations

    try:
        url = Settings().database_url("owner")
        conn = psycopg.connect(url, connect_timeout=3)
    except Exception as exc:  # no .env, or database not running (e.g. CI)
        pytest.skip(f"database not available: {type(exc).__name__}")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    try:
        apply_migrations(conn, schema=schema)
        yield conn
    finally:
        conn.autocommit = True
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.close()
