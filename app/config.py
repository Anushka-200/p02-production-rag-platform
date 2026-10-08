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

    # Login tokens (JWT, HS256). No default: a guessable secret would let anyone forge a tenant.
    jwt_secret: SecretStr | None = None
    jwt_ttl_minutes: int = 60
    jwt_issuer: str = "p02-rag"

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
    # Embeddings (task 1.8). "local" runs on this machine (free); "litellm" calls a hosted model.
    # The database column is vector(384), fixed by the default model: changing the model to one
    # with a different size needs a new migration and a full re-embed.
    embedding_provider: Literal["local", "litellm"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    # 16 not 256: big batches padded to 512 tokens made an 8 GB machine swap (8x slower).
    embedding_batch_size: int = 16
    # Ingestion worker (task 1.11)
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 25
    ingest_queue: str = "ingest"
    job_max_attempts: int = 3  # first try + 2 retries, then the job is marked failed
    blurbs_enabled: bool = False  # contextual blurbs cost money: opt in explicitly
    blurb_max_usd_per_document: float = 1.0  # hard stop per document (unattended: no prompt)
    # Retrieval (Phase 2). Every stage can be switched for the experiments.
    retrieval_mode: Literal["dense", "keyword", "hybrid"] = "hybrid"
    retrieval_candidates: int = 50  # how many each method returns before fusion
    rrf_k: int = 60
    rrf_dense_weight: float = 1.0
    # Provisional (informal preview B-008): a noisy keyword list must count for less than dense.
    rrf_keyword_weight: float = 0.3
    keyword_max_df: float = 0.02  # drop question words found in more than 2% of a tenant's chunks
    # Reranker: slow on CPU (~70 ms per pair measured), so only a few candidates are reranked.
    reranker_provider: Literal["none", "local", "cohere"] = "local"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cohere_rerank_model: str = "cohere/rerank-english-v3.0"
    reranker_threads: int | None = (
        None  # CPU threads for the local reranker (None = library default)
    )
    reranker_sort_by_length: bool = True  # group similar lengths in a batch: less padding
    rerank_candidates: int = 10  # taken from the top of the fused list
    rerank_top: int = 8  # kept after reranking
    rerank_max_words: int = 220  # longer passages are cut: faster, and the start carries the topic
    context_budget_tokens: int = 3000  # how much source text may go into the answer prompt
    embedding_max_tokens: int = 512  # the model silently cuts anything longer
    # Query transforms (Phase 5; measured in benchmark B-019, decision in ADR 007).
    query_rewrite: bool = True  # follow-ups -> standalone question; costs nothing without history
    multi_query: int = (
        0  # extra rephrasings to search with: OFF, no measured benefit for its latency
    )
    hyde: bool = False  # also search with a made-up answer: OFF, same reason
    transform_model: str | None = None  # None = llm_default_model
    # Hierarchical retrieval (Phase 5 block 3); OFF until measured (B-020, ADR 008)
    parent_expansion: bool = False  # swap each hit for its parent section
    parent_max_tokens: int = 700  # a parent larger than this is not used
    summary_tree: bool = False  # also search the RAPTOR-style summary tree and follow it to chunks
    summary_top_k: int = 5  # summaries to take from the tree
    summary_members_per: int = 4  # chunks to take from under each summary
    graph_search: bool = False  # follow facts (GraphRAG slice) and add the chunks that state them
    graph_hops: int = 2
    graph_max_chunks: int = 10
    # Agentic retrieval (Phase 5 block 5); OFF until measured (B-022, ADR 008)
    agent_enabled: bool = False
    agent_only_when_weak: bool = (
        True  # plain search first; the agent only when its best match is weak
    )
    agent_model: str | None = None  # None = llm_default_model
    agent_max_rounds: int = 2  # planning rounds (the second only when the first was not enough)
    agent_max_tool_calls: int = 4
    agent_max_cost_usd: float = 0.01  # the agent's own model calls, per question
    # Semantic cache (Phase 6): off until its threshold is chosen from real data (benchmark B-028)
    semantic_cache: bool = False
    semantic_cache_threshold: float = 0.93  # cosine similarity needed to reuse an answer
    semantic_cache_ttl_hours: float = 24.0
    semantic_cache_max_per_org: int = 2000
    # Chat history and memory (Phase 3)
    history_max_turns: int = 6  # most recent question+answer pairs that may go into a prompt
    history_budget_tokens: int = 1500  # ...and never more than this many tokens of them
    memory_enabled: bool = True
    memory_max_per_user: int = 200  # oldest-updated memory is dropped beyond this
    # Both thresholds were chosen from real embeddings: benchmark B-016.
    memory_dedupe_similarity: float = 0.90  # a new fact this close to an old one replaces it
    memory_min_relevance: float = 0.6  # facts less similar to the question are not used
    memory_top_k: int = 3
    # WhatsApp through Twilio (Phase 4)
    twilio_auth_token: SecretStr | None = None
    twilio_verify_signature: bool = True  # may be switched off ONLY when ENVIRONMENT=dev
    public_base_url: str | None = None  # e.g. https://abc.trycloudflare.com: the URL Twilio calls
    webhook_only: bool = False  # True when tunnelled: serve ONLY /health and the WhatsApp webhook
    whatsapp_session_hours: int = 24  # a quiet chat starts a fresh session after this long
    trust_proxy: bool = False  # True only behind a proxy that sets X-Forwarded-For (rate limits)
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
