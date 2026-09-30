"""Integration tests for the schema. Need the dev database (`make up`); skipped otherwise."""

import psycopg
import pytest


def make_org_user_doc(conn: psycopg.Connection, org: str = "acme", vis: str = "org"):  # type: ignore[no-untyped-def]
    org_id = conn.execute("INSERT INTO orgs (name) VALUES (%s) RETURNING id", (org,)).fetchone()[0]
    user_id = conn.execute(
        "INSERT INTO users (org_id, email) VALUES (%s, 'a@x.com') RETURNING id", (org_id,)
    ).fetchone()[0]
    doc_id = conn.execute(
        "INSERT INTO documents (org_id, owner_id, visibility, title, source_type, source_uri, "
        "content_hash) VALUES (%s, %s, %s, 'T', 'md', 'a.md', 'h1') RETURNING id",
        (org_id, user_id, vis),
    ).fetchone()[0]
    return org_id, user_id, doc_id


def add_chunk(conn, org_id, user_id, doc_id, text="hello", vis="org", idx=0, **kw):  # type: ignore[no-untyped-def]
    return conn.execute(
        "INSERT INTO chunks (org_id, owner_id, visibility, document_id, version, chunk_index, "
        "text, content_hash, section_path, context_blurb) "
        "VALUES (%s,%s,%s,%s,1,%s,%s,'c',%s,%s) RETURNING id",
        (org_id, user_id, vis, doc_id, idx, text, kw.get("section", ""), kw.get("blurb", "")),
    ).fetchone()[0]


def test_migrations_are_recorded_and_rerun_is_a_noop(db_schema) -> None:  # type: ignore[no-untyped-def]
    from app.db.migrate import apply_migrations

    assert apply_migrations(db_schema, schema=None) == []  # already applied in the fixture
    versions = [r[0] for r in db_schema.execute("SELECT version FROM schema_migrations")]
    assert "0001_init" in versions


def test_tsv_is_generated_and_keyword_search_finds_stemmed_words(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    add_chunk(db_schema, org, user, doc, text="Employees get 20 vacation days per year")
    hit = db_schema.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ websearch_to_tsquery('english', 'vacations')"
    ).fetchone()[0]
    assert hit == 1  # 'vacations' matches 'vacation' because both are stemmed


def test_context_blurb_is_searchable(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    add_chunk(db_schema, org, user, doc, text="Fee is 5%", blurb="From the Acme late payment terms")
    hit = db_schema.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ to_tsquery('english', 'acme & payment')"
    ).fetchone()[0]
    assert hit == 1  # blurb words are in the keyword index, not only the text


def test_chunk_must_match_its_documents_org_and_owner(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    other_org = db_schema.execute(
        "INSERT INTO orgs (name) VALUES ('evil') RETURNING id"
    ).fetchone()[0]
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        add_chunk(db_schema, other_org, user, doc)  # forged org_id


def test_visibility_change_cascades_to_chunks(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema, vis="org")
    add_chunk(db_schema, org, user, doc, vis="org")
    db_schema.execute("UPDATE documents SET visibility = 'private' WHERE id = %s", (doc,))
    vis = db_schema.execute(
        "SELECT visibility FROM chunks WHERE document_id = %s", (doc,)
    ).fetchone()[0]
    assert vis == "private"


def test_deleting_a_document_deletes_its_chunks(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    add_chunk(db_schema, org, user, doc)
    db_schema.execute("DELETE FROM documents WHERE id = %s", (doc,))
    assert db_schema.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_same_upload_twice_is_rejected_by_uniqueness(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    with pytest.raises(psycopg.errors.UniqueViolation):
        db_schema.execute(
            "INSERT INTO documents (org_id, owner_id, title, source_type, source_uri, "
            "content_hash) VALUES (%s, %s, 'T', 'md', 'a.md', 'h2')",
            (org, user),
        )


def test_bad_page_range_is_rejected(db_schema) -> None:  # type: ignore[no-untyped-def]
    org, user, doc = make_org_user_doc(db_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        db_schema.execute(
            "INSERT INTO chunks (org_id, owner_id, visibility, document_id, version, chunk_index, "
            "page_start, page_end, text, content_hash) VALUES (%s,%s,'org',%s,1,0,5,2,'x','c')",
            (org, user, doc),
        )
