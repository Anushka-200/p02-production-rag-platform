"""Searching the summary tree: find a broad summary, then walk DOWN to the real chunks under it.

    question --embed--> nearest summaries (level 2 / 3)    "Summary of 8 passages about leave..."
                              |  follow summary_members down to level-0 chunks
                              v
                  the chunks under the best summaries, closest to the question first

A summary is never shown to the answer model or cited. It is only a SIGNPOST that says "the answer
is probably in this group of chunks". The chunks are real document passages, so citations still
name a real document and page, and exact numbers and names are not lost to summarising.
Visibility is checked on the summaries AND on the chunks under them.
"""

import psycopg

from app.embeddings import vector_literal
from app.retrieval.search import _COLUMNS, VISIBLE_SQL, Candidate, Scope, _candidates


def summary_search(
    conn: psycopg.Connection, query_vector: list[float], scope: Scope, k: int
) -> list[Candidate]:
    """The k summaries (levels 2 and 3) closest to the question that the asker may read."""
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(
            f"SELECT {_COLUMNS}, c.embedding <=> %(vec)s::vector AS score "  # noqa: S608
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE c.org_id = %(org)s AND c.level IN (2, 3) AND {VISIBLE_SQL} "
            "AND d.status = 'ready' AND c.version = d.active_version AND c.embedding IS NOT NULL "
            "ORDER BY score LIMIT %(k)s",
            {
                "vec": vector_literal(query_vector),
                "org": scope.org_id,
                "user": scope.user_id,
                "k": k,
            },
        ).fetchall()
    return _candidates(rows)


def chunks_under(
    conn: psycopg.Connection, summaries: list[Candidate], query_vector: list[float], scope: Scope,
    per_summary: int,
) -> list[Candidate]:  # fmt: skip
    """Level-0 chunks reachable below the given summaries, best summary first, closest chunk first.

    At most `per_summary` from each summary; a chunk under several summaries is listed once, at
    its best position. The returned `rank` and `score` (distance to the question) are filled in."""
    if not summaries:
        return []
    rows = conn.execute(
        "WITH RECURSIVE walk(summary_id, chunk_id) AS ("
        "  SELECT m.summary_id, m.member_id FROM summary_members m "
        "  WHERE m.summary_id = ANY(%(ids)s)"
        "  UNION"
        "  SELECT w.summary_id, m.member_id FROM walk w "
        "  JOIN summary_members m ON m.summary_id = w.chunk_id"
        f") SELECT w.summary_id, {_COLUMNS}, c.embedding <=> %(vec)s::vector AS score "  # noqa: S608
        "FROM walk w JOIN chunks c ON c.id = w.chunk_id "
        "JOIN documents d ON d.id = c.document_id "
        f"WHERE c.org_id = %(org)s AND c.level = 0 AND {VISIBLE_SQL} "
        "AND d.status = 'ready' AND c.version = d.active_version AND c.embedding IS NOT NULL "
        "ORDER BY score",
        {
            "ids": [s.chunk_id for s in summaries], "vec": vector_literal(query_vector),
            "org": scope.org_id, "user": scope.user_id,
        },
    ).fetchall()  # fmt: skip
    by_summary: dict[object, list[tuple[object, ...]]] = {s.chunk_id: [] for s in summaries}
    for row in rows:
        by_summary[row[0]].append(row[1:])
    ordered: list[tuple[object, ...]] = []
    seen: set[object] = set()
    for summary in summaries:  # best summary first
        taken = 0
        for row in by_summary[summary.chunk_id]:  # already closest-first
            if row[0] in seen:
                continue
            seen.add(row[0])
            ordered.append(row)
            taken += 1
            if taken == per_summary:
                break
    return _candidates(list(ordered))


def tree_candidates(
    conn: psycopg.Connection,
    query_vector: list[float],
    scope: Scope,
    *,
    top_k: int,
    per_summary: int,
) -> list[Candidate]:
    return chunks_under(
        conn, summary_search(conn, query_vector, scope, top_k), query_vector, scope, per_summary
    )
