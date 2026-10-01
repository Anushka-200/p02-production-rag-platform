"""Database tests for migration 0002: vectors, the HNSW index and the anchors column."""

import psycopg
import pytest

from app.embeddings import vector_literal
from tests.test_migrations import add_chunk, make_org_user_doc


def unit(i: int) -> list[float]:
    """A 384-number vector pointing along axis i (axes are all equally far from each other)."""
    v = [0.0] * 384
    v[i] = 1.0
    return v


def test_nearest_neighbour_by_cosine_distance(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    ids = {}
    for i, name in enumerate(["vacation", "sick", "travel"]):
        cid = add_chunk(db_schema, org, user, doc, text=name, idx=i)
        db_schema.execute(
            "UPDATE chunks SET embedding = %s::vector, embedding_model = 'test' WHERE id = %s",
            (vector_literal(unit(i)), cid),
        )
        ids[name] = cid
    query = unit(1)  # points exactly at "sick"
    nearest = db_schema.execute(
        "SELECT text FROM chunks ORDER BY embedding <=> %s::vector LIMIT 1",
        (vector_literal(query),),
    ).fetchone()[0]
    assert nearest == "sick"


def test_hnsw_index_is_used_for_cosine_queries(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    cid = add_chunk(db_schema, org, user, doc)
    db_schema.execute(
        "UPDATE chunks SET embedding = %s::vector, embedding_model = 'test' WHERE id = %s",
        (vector_literal(unit(0)), cid),
    )
    db_schema.execute("SET enable_seqscan = off")  # tiny table: force the planner to show its index
    plan = "\n".join(
        r[0]
        for r in db_schema.execute(
            "EXPLAIN SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 1",
            (vector_literal(unit(0)),),
        )
    )
    assert "chunks_embedding_idx" in plan


def test_wrong_size_vector_is_rejected(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    cid = add_chunk(db_schema, org, user, doc)
    with pytest.raises(psycopg.errors.DataException):
        db_schema.execute(
            "UPDATE chunks SET embedding = '[1,2,3]'::vector, embedding_model = 'test' "
            "WHERE id = %s",
            (cid,),
        )


def test_embedding_must_name_its_model(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    cid = add_chunk(db_schema, org, user, doc)
    with pytest.raises(psycopg.errors.CheckViolation):
        db_schema.execute(
            "UPDATE chunks SET embedding = %s::vector WHERE id = %s", (vector_literal(unit(0)), cid)
        )


def test_parent_chunks_may_have_no_embedding_and_anchors_roundtrip(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    cid = add_chunk(db_schema, org, user, doc)
    db_schema.execute(
        "UPDATE chunks SET anchors = %s WHERE id = %s", (["time-off", "expenses"], cid)
    )
    row = db_schema.execute(
        "SELECT embedding, anchors FROM chunks WHERE id = %s", (cid,)
    ).fetchone()
    assert row[0] is None and row[1] == ["time-off", "expenses"]
