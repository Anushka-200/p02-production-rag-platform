import pytest

from app.config import Settings


def make(**env: str) -> Settings:
    # _env_file=None: tests must not depend on a developer's real .env
    return Settings(_env_file=None, **env)  # type: ignore[call-arg]


def test_defaults_use_p02_ports_not_p01() -> None:
    s = make()
    assert (s.postgres_port, s.redis_port, s.api_port, s.ui_port) == (5434, 6380, 8001, 8502)


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "6000")
    assert Settings(_env_file=None).postgres_port == 6000  # type: ignore[call-arg]


def test_secrets_are_masked_in_repr() -> None:
    s = make(rag_app_password="hunter2", anthropic_api_key="sk-secret")
    assert "hunter2" not in repr(s)
    assert "sk-secret" not in repr(s)


def test_database_url_per_role_and_escapes_password() -> None:
    s = make(rag_app_password="p@ss/w:rd")
    url = s.database_url("app")
    assert url.startswith("postgresql://rag_app:")
    assert "p%40ss%2Fw%3Ard" in url
    assert url.endswith("@localhost:5434/rag")


def test_missing_password_gives_clear_error() -> None:
    with pytest.raises(ValueError, match="RAG_WORKER_PASSWORD"):
        make().database_url("worker")
