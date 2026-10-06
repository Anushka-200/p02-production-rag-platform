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
    """Who is asking. Comes from the verified login, never from the question or the model.

    `user_id` is None for an ANONYMOUS visitor (widget, WhatsApp): they belong to an org's public
    site but are nobody in particular, and may see only documents marked 'public'.
    """

    org_id: uuid.UUID
    user_id: uuid.UUID | None

    @property
    def anonymous(self) -> bool:
        return self.user_id is None


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
    "c.org_id = %(org)s AND c.level = 0 AND (c.visibility = 'public' "
    "OR (%(user)s::uuid IS NOT NULL AND (c.visibility = 'org' OR c.owner_id = %(user)s::uuid))) "
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


_DF_CACHE: dict[tuple[uuid.UUID, uuid.UUID | None], tuple[int, dict[str, int]]] = {}


def document_frequencies(
    conn: psycopg.Connection, org_id: uuid.UUID, user_id: uuid.UUID | None = None
) -> tuple[int, dict[str, int]]:
    """(number of searchable chunks, {word: in how many of them}) for one org AS SEEN BY one user.

    Row-level security hides chunks the user may not read, so the statistics are computed over
    what THIS user can see and cached per (org, user): an anonymous visitor's numbers (public
    documents only) never become a member's numbers. Refreshed when the visible count changes.
    """
    n = conn.execute(
        "SELECT count(*) FROM chunks WHERE org_id = %s AND level = 0", (org_id,)
    ).fetchone()[0]  # type: ignore[index]
    key = (org_id, user_id)
    cached = _DF_CACHE.get(key)
    if cached and cached[0] == n:
        return cached
    safe_id = uuid.UUID(str(org_id))  # re-parsed: only a well-formed UUID can reach the SQL text
    rows = conn.execute(
        "SELECT word, ndoc FROM ts_stat(%s)",
        (f"SELECT tsv FROM chunks WHERE org_id = '{safe_id}' AND level = 0",),
    ).fetchall()
    _DF_CACHE[key] = (n, {word: ndoc for word, ndoc in rows})
    return _DF_CACHE[key]


def rare_keyword_query(
    conn: psycopg.Connection,
    question: str,
    scope: Scope,
    max_df: float = 0.02,
    keep_at_least: int = 3,
) -> str:
    """The OR query of the question's INFORMATIVE words.

    Postgres ranking has no inverse document frequency, so words like "how" or "company" would
    flood the results. We drop words found in more than max_df of the tenant's chunks (and words
    found nowhere, which cannot match). If every word is common, keep the rarest few.
    """
    words = list(dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", question)))
    if not words:
        return ""
    lexemes: dict[str, str | None] = dict(
        conn.execute(
            "SELECT w, (tsvector_to_array(to_tsvector('english', w)))[1] "
            "FROM unnest(%s::text[]) AS w",
            (words,),
        ).fetchall()
    )  # word -> its stem; stop words have none
    n, df = document_frequencies(conn, scope.org_id, scope.user_id)
    known = [
        (df[lex] / n, w) for w in words if n and (lex := lexemes.get(w)) is not None and lex in df
    ]
    rare = [w for ratio, w in known if ratio <= max_df]
    if not rare:
        rare = [w for _, w in sorted(known)[:keep_at_least]]
    return " OR ".join(rare)


def keyword_search(
    conn: psycopg.Connection,
    question: str,
    scope: Scope,
    k: int = 50,
    normalization: int = 32,
    max_df: float | None = None,
) -> list[Candidate]:
    """The k chunks containing the most (and the closest-together) question words.

    OR, not AND: AND returns nothing as soon as one word is missing. Ranking sorts out precision.
    `normalization` 32 squeezes the rank into 0..1 (see ADR 003 for ts_rank vs BM25).
    """
    query = (
        rare_keyword_query(conn, question, scope, max_df)
        if max_df is not None
        else keyword_query(question)
    )
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
