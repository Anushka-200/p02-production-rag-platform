"""The two ways to find candidate chunks: dense (meaning) and keyword (words).

Both return child chunks only (level 0; parents are never searched) that the asker may see:
same organisation, and either org-visible or the asker's own private documents. Phase 3 adds
row-level security as a second lock; this explicit filter is the first (never trust one layer).
"""

import re
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg

from app.embeddings import vector_literal


@dataclass(frozen=True)
class Scope:
    """Who is asking. Comes from the verified login, never from the question or the model."""

    org_id: uuid.UUID
    user_id: uuid.UUID


@dataclass
class Candidate:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_uri: str
    section_path: str
    page_start: int | None
    page_end: int | None
    anchors: list[str]
    text: str
    context_blurb: str
    parent_id: uuid.UUID | None
    rank: int  # 1 = best, within the method that found it
    score: (
        float  # dense: cosine distance (smaller is closer); keyword: ts_rank_cd (bigger is better)
    )


_COLUMNS = (
    "c.id, c.document_id, d.source_uri, c.section_path, c.page_start, c.page_end, c.anchors, "
    "c.text, c.context_blurb, c.parent_id"
)
_ALLOWED = (
    "c.org_id = %(org)s AND c.level = 0 AND (c.visibility = 'org' OR c.owner_id = %(user)s) "
    "AND d.status = 'ready' AND c.version = d.active_version"
)


def _candidates(rows: list[Any]) -> list[Candidate]:
    """Rows in _COLUMNS order plus the score as last column, already sorted best first."""
    return [
        Candidate(
            chunk_id=r[0],
            document_id=r[1],
            source_uri=r[2],
            section_path=r[3],
            page_start=r[4],
            page_end=r[5],
            anchors=r[6],
            text=r[7],
            context_blurb=r[8],
            parent_id=r[9],
            rank=i,
            score=float(r[10]),
        )  # fmt: skip
        for i, r in enumerate(rows, start=1)
    ]


def dense_search(
    conn: psycopg.Connection, query_vector: list[float], scope: Scope, k: int = 50
) -> list[Candidate]:
    """The k chunks whose meaning is closest to the question's vector."""
    with conn.transaction():
        # Without this, the tenant filter is applied AFTER the index finds ~40 neighbours across
        # all tenants, so a small tenant can get far fewer than k rows (benchmark B-007).
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(
            f"SELECT {_COLUMNS}, c.embedding <=> %(vec)s::vector AS score "  # noqa: S608
            f"FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE {_ALLOWED} AND c.embedding IS NOT NULL ORDER BY score LIMIT %(k)s",
            {
                "vec": vector_literal(query_vector),
                "org": scope.org_id,
                "user": scope.user_id,
                "k": k,
            },
        ).fetchall()
    return _candidates(rows)


def keyword_query(question: str) -> str:
    """The question's words joined with OR, safe to hand to Postgres.

    Only letters, digits and underscores survive, so quotes, minus signs or the word OR typed by a
    user cannot change the query's meaning. Postgres drops stop words ("what", "does") itself.
    """
    words = dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", question))
    return " OR ".join(words)


def keyword_search(
    conn: psycopg.Connection, question: str, scope: Scope, k: int = 50, normalization: int = 32
) -> list[Candidate]:
    """The k chunks containing the most (and the closest-together) question words.

    OR, not AND: AND returns nothing as soon as one word is missing. Ranking sorts out precision.
    `normalization` 32 squeezes the rank into 0..1 (see ADR 003 for ts_rank vs BM25).
    """
    query = keyword_query(question)
    if not query:
        return []
    rows = conn.execute(
        f"SELECT {_COLUMNS}, ts_rank_cd(c.tsv, q.query, %(norm)s) AS score "  # noqa: S608
        f"FROM chunks c JOIN documents d ON d.id = c.document_id, "
        f"websearch_to_tsquery('english', %(q)s) AS q(query) "
        f"WHERE {_ALLOWED} AND c.tsv @@ q.query ORDER BY score DESC, c.id LIMIT %(k)s",
        {"q": query, "norm": normalization, "org": scope.org_id, "user": scope.user_id, "k": k},
    ).fetchall()
    return _candidates(rows)
