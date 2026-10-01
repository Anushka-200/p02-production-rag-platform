"""All configuration comes from environment variables (or a local .env file), nowhere else.

One typed class means a typo or a missing value fails loudly at startup, not silently at 2 a.m.
"""

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DbRole = Literal["owner", "app", "worker"]


class Settings(BaseSettings):
    # Field name <-> env var name, case-insensitive (postgres_port <-> POSTGRES_PORT).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "dev"
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5434
    postgres_db: str = "rag"
    # SecretStr prints as '**********', so a stray log line or repr cannot leak the value.
    rag_owner_password: SecretStr | None = None
    rag_app_password: SecretStr | None = None
    rag_worker_password: SecretStr | None = None

    redis_host: str = "localhost"
    redis_port: int = 6380

    api_port: int = 8001
    ui_port: int = 8502

    anthropic_api_key: SecretStr | None = None
    # Cheap tier by default (shared standard); the stronger model is only used as the eval judge.
    llm_default_model: str = "anthropic/claude-haiku-4-5-20251001"
    llm_judge_model: str = "anthropic/claude-sonnet-5-5"
    # Chunking (task 1.6/1.7): which strategy and how big. Tuned by the Phase 6 experiments.
    chunk_strategy: Literal["recursive", "structured"] = "structured"
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50
    parent_tokens: int = 1200
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    def database_url(self, role: DbRole) -> str:
        """Connection string for one of the three DB roles (never the superuser).

        The API uses "app", the ingestion worker "worker", migrations "owner". Keeping them
        separate is what lets row-level security apply to the API.
        """
        password = getattr(self, f"rag_{role}_password")
        if password is None:
            raise ValueError(f"RAG_{role.upper()}_PASSWORD is not set; copy .env.example to .env")
        return (
            f"postgresql://rag_{role}:{quote_plus(password.get_secret_value())}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
