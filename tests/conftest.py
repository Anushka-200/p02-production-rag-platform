import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's environment.

    Some libraries (litellm) load .env into os.environ when imported, which would otherwise leak
    real passwords and ports into unrelated tests.
    """
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
