"""LIVE check of the semantic cache against the real database and model (costs about a cent).

Asks a question twice (miss, then hit), a close paraphrase, and the same question as ANOTHER tenant.
Run: uv run python -m scripts.cache_live [--yes]
"""

import argparse

import psycopg

from app.config import get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.db.tenant import tenant_connection
from app.embeddings import get_embedder
from app.log import setup_logging
from app.rag import ask_stream
from app.retrieval.rerank import get_reranker
from app.retrieval.search import Scope
from app.semantic_cache import make_writer

QUESTION = "How many weeks of parental leave does GitLab offer?"
PARAPHRASE = "how many weeks of parental leave does GitLab give?"


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    args = parser.parse_args()
    setup_logging("ERROR")
    s = get_settings().model_copy(update={"semantic_cache": True})
    guard_from_args(
        args, "cache live", estimate_llm_cost(s.llm_default_model, 4500, 250, calls=4)
    ).confirm()
    owner = psycopg.connect(s.database_url("owner"), autocommit=True)
    scopes = {}
    for key, org in (("A", "Tenant A (GitLab handbook)"), ("B", "Tenant B (PostHog handbook)")):
        o, u = owner.execute(
            "SELECT o.id, u.id FROM orgs o JOIN users u ON u.org_id = o.id WHERE o.name = %s "
            "ORDER BY u.created_at LIMIT 1",
            (org,),
        ).fetchone()  # type: ignore[misc]
        scopes[key] = Scope(o, u)
    owner.execute("DELETE FROM semantic_cache")  # start clean
    embedder, reranker = get_embedder(s), get_reranker(s)

    def connect() -> psycopg.Connection:
        return psycopg.connect(s.database_url("app"), autocommit=True)

    writer = make_writer(connect, s)
    for label, who, question in (
        ("1 first ask (miss)", "A", QUESTION),
        ("2 same again (hit)", "A", QUESTION),
        ("3 paraphrase", "A", PARAPHRASE),
        ("4 other tenant, same words", "B", QUESTION),
    ):
        with tenant_connection(connect, scopes[who]) as conn:
            done = None
            for event in ask_stream(
                conn, embedder, reranker, question, scopes[who], settings=s, cache_writer=writer
            ):
                if event["type"] == "done":
                    done = event
        assert done is not None
        a, t = done["answer"], done["result"].timings_ms
        print(
            f"{label:30} model={a.model:22} cost=${a.cost_usd:.4f} total={t['total_ms']:.0f} ms "
            f"refused={a.refused} cites={[c.document.split('/')[-1] for c in a.citations][:2]}"
        )
    n = owner.execute(
        "SELECT count(*), count(*) FILTER (WHERE shareable) FROM semantic_cache"
    ).fetchone()
    print("cache rows:", n)


if __name__ == "__main__":
    main()
