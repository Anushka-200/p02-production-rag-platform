"""Parent expansion ("small to big"): find with a small chunk, read the larger section around it.

Small chunks are good for FINDING (a short passage matches a question tightly) but poor for
ANSWERING (the sentence before or after may hold the missing half). Chunker B therefore stores each
searchable child chunk next to a larger PARENT chunk (a whole section) that is never searched.
After reranking, each hit is swapped for its parent:

    hit  (child, 120 words):  "...Parental leave is 16 weeks."
    read (parent, 450 words): "Leave policies ... Parental leave is 16 weeks. It can be taken in
                               blocks within 12 months. Sick leave ..."

Rules: two hits with the same parent become ONE source (no duplicates); a parent larger than the
cap is not used (the child stays: a huge parent would crowd out the other sources); the swapped
source keeps the child's rank, and its pages and anchors cover the whole parent so citations point
at the full section. Everything is read under the asker's own visibility (public / org / own).
"""

import uuid
from dataclasses import replace

import psycopg

from app.ingest.chunking import TokenCounter
from app.retrieval.context import approx_tokens
from app.retrieval.rerank import Reranked
from app.retrieval.search import VISIBLE_SQL, Scope


def expand_to_parents(
    conn: psycopg.Connection,
    items: list[Reranked],
    scope: Scope,
    *,
    max_parent_tokens: int = 700,
    count: TokenCounter = approx_tokens,
) -> list[Reranked]:
    """The same list with each hit that has a usable parent replaced by that parent's text."""
    parent_ids = list({i.fused.candidate.parent_id for i in items if i.fused.candidate.parent_id})
    if not parent_ids:
        return items
    rows = conn.execute(
        "SELECT c.id, c.text, c.section_path, c.page_start, c.page_end, c.anchors "
        "FROM chunks c WHERE c.id = ANY(%(ids)s) AND c.org_id = %(org)s AND c.level = 1 "
        f"AND {VISIBLE_SQL}",
        {"ids": parent_ids, "org": scope.org_id, "user": scope.user_id},
    ).fetchall()
    parents = {r[0]: r for r in rows}
    out: list[Reranked] = []
    used: set[uuid.UUID] = set()
    for item in items:
        c = item.fused.candidate
        parent = parents.get(c.parent_id) if c.parent_id else None
        if parent is None or count(parent[1]) > max_parent_tokens:
            out.append(item)  # no parent, a hidden one, or too big: keep the small chunk
            continue
        if parent[0] in used:
            continue  # a better-ranked hit already brought this parent
        used.add(parent[0])
        bigger = replace(
            c,
            text=parent[1],
            section_path=parent[2] or c.section_path,
            page_start=parent[3] if parent[3] is not None else c.page_start,
            page_end=parent[4] if parent[4] is not None else c.page_end,
            anchors=sorted({*c.anchors, *(parent[5] or [])}),
            context_blurb="",
        )
        out.append(replace(item, fused=replace(item.fused, candidate=bigger)))
    return out
