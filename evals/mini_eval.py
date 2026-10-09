"""A small, self-contained eval that runs in CI with no API key, no model download and no cost.

The real eval (evals/run.py, 267 questions over the GitLab / PostHog handbooks) needs the corpus, a local
embedding model and paid LLM calls. This one exists to catch REGRESSIONS in the plumbing on every push:
four short documents are ingested through the real pipeline (parse, chunk, store in Postgres/pgvector) and
15 questions are searched with the real hybrid retrieval (dense + keyword + RRF). Only the embedder is a
stand-in: a hashed bag of words (deterministic, 384 dimensions). The 3 unanswerable questions are listed
for completeness but skipped, because judging a refusal needs the LLM.

Run: make eval-ci   (needs the Postgres from `make up`; the test suite runs the same code)
"""

import hashlib
import math
import re
from pathlib import Path

import psycopg

from app.config import Settings
from app.connectors.base import SourceItem
from app.connectors.sync import make_pipeline_ingest
from app.retrieval.retrieve import retrieve
from app.retrieval.search import Scope
from evals.metrics import Report, evaluate
from evals.schema import EvalItem

HERE = Path(__file__).parent / "mini"
DIM = 384


class HashEmbedder:
    """Words hashed into 384 buckets, then normalised. Same words, same vector, every run."""

    model, dimension = "hashed-bag-of-words", DIM

    def embed_query(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if (h >> 20) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def load_items() -> list[EvalItem]:
    lines = (HERE / "qa.jsonl").read_text().splitlines()
    return [EvalItem.model_validate_json(line) for line in lines if line.strip()]


def run_mini_eval(conn: psycopg.Connection, org_id: object, user_id: object) -> Report:
    """Ingest the mini corpus for one user, search every answerable question, return the report."""
    settings = Settings(_env_file=None, environment="dev", reranker_provider="none")  # type: ignore[call-arg]
    embedder = HashEmbedder()
    ingest = make_pipeline_ingest(
        conn, org_id, user_id, settings=settings, embedder=embedder,  # type: ignore[arg-type]
        count=lambda text: len(text.split()),
    )  # fmt: skip
    for path in sorted((HERE / "corpus").glob("*.md")):
        raw = path.read_bytes()
        ingest(SourceItem(path.name, path.name, lambda raw=raw: raw, lambda: None, "org"), raw)  # type: ignore[misc]
    scope = Scope(org_id, user_id)  # type: ignore[arg-type]

    def search(item: EvalItem) -> list[object]:
        got = retrieve(conn, embedder, item.question, scope, settings=settings, mode="hybrid")
        return [fused.candidate for fused in got.items]

    return evaluate(load_items(), search)  # type: ignore[arg-type]


if __name__ == "__main__":
    import uuid

    from app.config import get_settings

    # Standalone run: a throwaway org that is deleted again at the end.
    with psycopg.connect(get_settings().database_url("owner"), autocommit=True) as c:
        org = c.execute(
            "INSERT INTO orgs (name) VALUES (%s) RETURNING id", (f"mini-{uuid.uuid4().hex[:6]}",)
        ).fetchone()[0]  # type: ignore[index]
        user = c.execute(
            "INSERT INTO users (org_id, email) VALUES (%s, 'mini@x.com') RETURNING id", (org,)
        ).fetchone()[0]  # type: ignore[index]
        report = run_mini_eval(c, org, user)
        print({k: round(v, 3) for k, v in report.overall.items()}, "n =", report.n)
        from evals.redteam import purge_org

        name = c.execute("SELECT name FROM orgs WHERE id = %s", (org,)).fetchone()[0]  # type: ignore[index]
        purge_org(c, name)
