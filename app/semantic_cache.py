"""Semantic cache: reuse a finished answer when a near-identical question comes again.

    "How many vacation days do I get?"  ->  answered, stored with the question's meaning vector
    "how much vacation do I have?"      ->  similar enough: same answer, no search, no model call

Why it is dangerous, and what stops each danger (ADR 012):
  * Another user's private knowledge leaking: an entry is visible to its owner only, unless every
      document the model saw was non-private (`shareable`).
  * Another tenant's answer: rows are filtered by org, and by row-level security.
  * A stale answer after a document changed: each entry remembers (document, version) of everything
      the model saw; on lookup all must still be the active, visible versions, else it is ignored.
  * A different question with a similar vector: a high threshold, chosen from data (B-028).
  * Personalised or follow-up questions: the caller does not use the cache for them at all.
"""

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg

from app.config import Settings
from app.db.tenant import tenant_connection
from app.embeddings import vector_literal
from app.retrieval.context import Source
from app.retrieval.search import Scope

logger = logging.getLogger(__name__)


@dataclass
class CachedAnswer:
    id: uuid.UUID
    question: str  # the question that was originally answered
    similarity: float
    answer: dict[str, Any]  # {text, citations, sources, model}


def usable(scope: Scope) -> bool:
    """Only signed-in users: anonymous visitors have no user the entry could belong to."""
    return scope.user_id is not None


def still_valid(conn: psycopg.Connection, scope: Scope, refs: list[dict[str, Any]]) -> bool:
    """Every document the answer was built from is still ready, current and visible to the asker."""
    if not refs:
        return False
    row = conn.execute(
        "SELECT count(*) FROM unnest(%(ids)s::uuid[], %(vers)s::int[]) AS r(id, ver) "
        "JOIN documents d ON d.id = r.id AND d.active_version = r.ver AND d.status = 'ready' "
        "WHERE d.org_id = %(org)s AND (d.visibility IN ('org', 'public') OR d.owner_id = %(user)s)",
        {
            "ids": [r["document_id"] for r in refs], "vers": [r["version"] for r in refs],
            "org": scope.org_id, "user": scope.user_id,
        },
    ).fetchone()  # fmt: skip
    return row is not None and row[0] == len({r["document_id"] for r in refs})


def lookup(
    conn: psycopg.Connection,
    scope: Scope,
    vector: list[float],
    *,
    threshold: float,
    ttl_hours: float,
) -> CachedAnswer | None:
    if not usable(scope):
        return None
    rows = conn.execute(
        "SELECT id, question, answer, doc_refs, 1 - (embedding <=> %(vec)s::vector) AS sim "
        "FROM semantic_cache WHERE org_id = %(org)s AND (user_id = %(user)s OR shareable) "
        "AND created_at > now() - make_interval(secs => %(ttl)s) "
        "ORDER BY embedding <=> %(vec)s::vector LIMIT 3",
        {
            "vec": vector_literal(vector), "org": scope.org_id, "user": scope.user_id,
            "ttl": ttl_hours * 3600,
        },
    ).fetchall()  # fmt: skip
    for entry_id, question, answer, refs, sim in rows:
        if sim < threshold:
            break  # sorted by closeness: nothing further down can pass
        if still_valid(conn, scope, refs):
            return CachedAnswer(entry_id, question, float(sim), answer)
    return None


def store(
    conn: psycopg.Connection,
    scope: Scope,
    question: str,
    vector: list[float],
    answer: dict[str, Any],
    sources: list[Source],
    *,
    ttl_hours: float,
    max_per_org: int,
) -> bool:
    """Save an answer (`conn` = a WRITE tenant connection). False if not worth saving."""
    if not usable(scope) or not sources:
        return False
    ids = list({s.document_id for s in sources})
    docs = conn.execute(
        "SELECT id, active_version, visibility FROM documents WHERE id = ANY(%s)", (ids,)
    ).fetchall()
    if len(docs) != len(ids):
        return False  # a document vanished or is hidden from us: do not cache
    shareable = all(vis != "private" for _, _, vis in docs)
    refs = [{"document_id": str(i), "version": v} for i, v, _ in docs]
    conn.execute(
        "DELETE FROM semantic_cache WHERE org_id = %s "
        "AND (created_at < now() - make_interval(secs => %s) "
        "OR id IN (SELECT id FROM semantic_cache WHERE org_id = %s "
        "ORDER BY created_at DESC OFFSET %s))",
        (scope.org_id, ttl_hours * 3600, scope.org_id, max_per_org - 1),
    )
    conn.execute(
        "INSERT INTO semantic_cache "
        "(org_id, user_id, shareable, question, embedding, answer, doc_refs) "
        "VALUES (%s, %s, %s, %s, %s::vector, %s, %s)",
        (
            scope.org_id,
            scope.user_id,
            shareable,
            question,
            vector_literal(vector),
            json.dumps(answer, default=str),
            json.dumps(refs),
        ),
    )
    return True


# What ask_stream calls after a fresh answer: (scope, question, vector, answer fields, sources).
CacheWriter = Callable[[Scope, str, list[float], dict[str, Any], list[Source]], None]


def make_writer(open_connection: Callable[[], psycopg.Connection], s: Settings) -> CacheWriter:
    """A writer that saves into the cache on its OWN short write connection, as the asker."""

    def write(
        scope: Scope,
        question: str,
        vector: list[float],
        answer: dict[str, Any],
        sources: list[Source],
    ) -> None:
        with tenant_connection(open_connection, scope, write=True) as conn:
            store(
                conn, scope, question, vector, answer, sources,
                ttl_hours=s.semantic_cache_ttl_hours, max_per_org=s.semantic_cache_max_per_org,
            )  # fmt: skip

    return write
