"""Tests for saving documents: versions, dedupe, atomic swaps. Need the dev database."""

import uuid

import pytest

from app.embeddings import vector_literal
from app.ingest.chunking import Chunk
from app.ingest.store import IngestError, file_hash, plan_ingest, write_version


def unit(i: int) -> list[float]:
    v = [0.0] * 384
    v[i % 384] = 1.0
    return v


def make_chunks(prefix: str = "v1", n_children: int = 3, with_parent: bool = True) -> list[Chunk]:
    chunks: list[Chunk] = []
    if with_parent:
        chunks.append(
            Chunk(
                f"{prefix} parent text",
                ("Doc", "Topic"),
                "topic",
                5,
                6,
                30,
                0,
                level=1,
                anchors=("topic",),
            )
        )
    for i in range(n_children):
        chunks.append(
            Chunk(f"{prefix} child {i}", ("Doc", "Topic"), f"sec-{i}", 5 + i // 2, 6, 10,
                  len(chunks), level=0, anchors=(f"sec-{i}",), context_blurb=f"{prefix} blurb {i}",
                  parent_index=0 if with_parent else None)
        )  # fmt: skip
    return chunks


def vectors(chunks: list[Chunk]) -> list[list[float] | None]:
    return [None if c.level == 1 else unit(c.index) for c in chunks]


def tenant(conn):  # type: ignore[no-untyped-def]
    org = conn.execute(
        "INSERT INTO orgs (name) VALUES (%s) RETURNING id", (uuid.uuid4().hex,)
    ).fetchone()[0]
    user = conn.execute(
        "INSERT INTO users (org_id, email) VALUES (%s, 'a@x.com') RETURNING id", (org,)
    ).fetchone()[0]
    return org, user


def ingest(conn, org, user, raw: bytes, chunks, uri="handbook.md", vis="org", embeddings=None):  # type: ignore[no-untyped-def]
    h = file_hash(raw)
    plan = plan_ingest(conn, org_id=org, owner_id=user, source_uri=uri, content_hash=h)
    return plan, write_version(
        conn, plan, org_id=org, owner_id=user, visibility=vis, source_uri=uri, source_type="md",
        content_hash=h, title="Doc", chunks=chunks,
        embeddings=embeddings if embeddings is not None else vectors(chunks),
        embedding_model="test-model",
    )  # fmt: skip


def test_first_ingest_stores_document_chunks_parent_links_and_embeddings(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    plan, result = ingest(db_schema, org, user, b"v1", make_chunks())
    assert plan.action == "ingest" and result.created and result.version == 1
    doc = db_schema.execute("SELECT status, active_version, content_hash FROM documents").fetchone()
    assert doc == ("ready", 1, file_hash(b"v1"))
    rows = db_schema.execute(
        "SELECT level, parent_id IS NOT NULL, embedding IS NOT NULL, embedding_model, "
        "section_path, page_start, page_end, anchors, context_blurb "
        "FROM chunks ORDER BY chunk_index"
    ).fetchall()
    assert rows[0][:3] == (1, False, False)  # the parent: no embedding, no parent
    assert [r[:3] for r in rows[1:]] == [(0, True, True)] * 3  # children: linked and embedded
    assert rows[1][3:] == ("test-model", "Doc > Topic", 5, 6, ["sec-0"], "v1 blurb 0")
    parent_id = db_schema.execute("SELECT id FROM chunks WHERE level = 1").fetchone()[0]
    linked = db_schema.execute(
        "SELECT count(*) FROM chunks WHERE parent_id = %s", (parent_id,)
    ).fetchone()[0]
    assert linked == 3


def test_same_file_again_is_skipped_before_any_work(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    ingest(db_schema, org, user, b"v1", make_chunks())
    plan = plan_ingest(db_schema, org_id=org, owner_id=user, source_uri="handbook.md",
                       content_hash=file_hash(b"v1"))  # fmt: skip
    assert plan.action == "skip_unchanged"
    assert db_schema.execute("SELECT count(*) FROM chunks").fetchone()[0] == 4  # untouched


def test_edited_file_replaces_the_old_version_completely(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    ingest(db_schema, org, user, b"v1", make_chunks("v1", 3))
    old_ids = {r[0] for r in db_schema.execute("SELECT id FROM chunks")}
    plan, result = ingest(db_schema, org, user, b"v2 edited", make_chunks("v2", 2))
    assert plan.action == "ingest" and not result.created
    assert result.version == 2 and result.chunks_written == 3 and result.chunks_deleted == 4
    texts = sorted(r[0] for r in db_schema.execute("SELECT text FROM chunks"))
    assert texts == ["v2 child 0", "v2 child 1", "v2 parent text"]  # no v1 text left anywhere
    assert not old_ids & {r[0] for r in db_schema.execute("SELECT id FROM chunks")}
    assert db_schema.execute("SELECT active_version, content_hash FROM documents").fetchone() == (
        2, file_hash(b"v2 edited"),
    )  # fmt: skip
    assert db_schema.execute("SELECT count(*) FROM documents").fetchone()[0] == 1  # same document


def test_a_failed_write_leaves_the_old_version_untouched_and_records_why(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    ingest(db_schema, org, user, b"v1", make_chunks("v1"))
    bad = make_chunks("v2")
    vecs = vectors(bad)
    vecs[2] = [1.0, 2.0, 3.0]  # wrong size: the database rejects it halfway through the write
    with pytest.raises(IngestError):
        ingest(db_schema, org, user, b"v2", bad, embeddings=vecs)
    texts = sorted(r[0] for r in db_schema.execute("SELECT text FROM chunks"))
    assert all(t.startswith("v1") for t in texts) and len(texts) == 4  # old version fully intact
    status, version, err = db_schema.execute(
        "SELECT status, active_version, error FROM documents"
    ).fetchone()
    assert (status, version) == ("failed", 1) and err  # failure recorded with a message
    # and the next attempt, with good data, succeeds and clears the error
    ingest(db_schema, org, user, b"v2", make_chunks("v2"))
    assert db_schema.execute("SELECT status, active_version, error FROM documents").fetchone() == (
        "ready",
        2,
        None,
    )


def test_first_ever_write_that_fails_leaves_nothing_behind(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    chunks = make_chunks()
    vecs = vectors(chunks)
    vecs[1] = [1.0]
    with pytest.raises(IngestError):
        ingest(db_schema, org, user, b"v1", chunks, embeddings=vecs)
    assert db_schema.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert db_schema.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_same_bytes_under_another_name_is_a_duplicate_for_the_same_owner_only(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    ingest(db_schema, org, user, b"same bytes", make_chunks(), uri="a.md")
    twin = plan_ingest(db_schema, org_id=org, owner_id=user, source_uri="copy-of-a.md",
                       content_hash=file_hash(b"same bytes"))  # fmt: skip
    assert twin.action == "skip_duplicate" and twin.duplicate_of is not None
    # Another user in the same org uploading the same bytes must NOT be told it exists.
    other = db_schema.execute(
        "INSERT INTO users (org_id, email) VALUES (%s, 'b@x.com') RETURNING id", (org,)
    ).fetchone()[0]
    plan = plan_ingest(db_schema, org_id=org, owner_id=other, source_uri="a.md",
                       content_hash=file_hash(b"same bytes"))  # fmt: skip
    assert plan.action == "ingest" and plan.duplicate_of is None


def test_visibility_is_copied_to_chunks_and_follows_a_change(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    ingest(db_schema, org, user, b"v1", make_chunks(), vis="private")
    assert {r[0] for r in db_schema.execute("SELECT visibility FROM chunks")} == {"private"}
    ingest(db_schema, org, user, b"v2", make_chunks("v2"), vis="org")  # re-uploaded as shared
    assert {r[0] for r in db_schema.execute("SELECT visibility FROM chunks")} == {"org"}


def test_blurb_words_are_found_by_keyword_search_after_saving(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    chunk = Chunk("The fee is 5%.", ("Doc",), "d", None, None, 5, 0, anchors=("d",),
                  context_blurb="From the Acme late payment terms.")  # fmt: skip
    ingest(db_schema, org, user, b"x", [chunk], embeddings=[unit(0)])
    hit = db_schema.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ websearch_to_tsquery('english', 'acme payment')"
    ).fetchone()[0]
    assert hit == 1  # the blurb words, not in the chunk text, are searchable


def test_dense_search_only_sees_children(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    ingest(db_schema, org, user, b"v1", make_chunks())
    nearest = db_schema.execute(
        "SELECT level FROM chunks WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::vector LIMIT 10",
        (vector_literal(unit(1)),),
    ).fetchall()
    assert nearest and all(level == 0 for (level,) in nearest)  # parents have no embedding


def test_embeddings_must_line_up_with_chunks(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user = tenant(db_schema)
    with pytest.raises(ValueError, match="line up"):
        ingest(db_schema, org, user, b"x", make_chunks(), embeddings=[None])
