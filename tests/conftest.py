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
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never send traces to Langfuse, even if the developer's .env has real keys."""
    monkeypatch.setattr(tracing, "get_client", lambda: None)
