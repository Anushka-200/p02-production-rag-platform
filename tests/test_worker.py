"""The worker end to end (real Postgres, fake embedder and LLM): success and every failure."""

import hashlib
import uuid
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from app.config import Settings
from app.ingest.chunking import approx_tokens
from app.ingest.upload import enqueue_upload
from app.llm import LLMResult
from app.worker import run_job

MARKDOWN = (
    b"---\ntitle: Acme Handbook\n---\nWelcome to Acme.\n\n"
    b"## Benefits\n\nAcme offers many benefits to every employee.\n\n"
    b"### Time off\n\nEmployees receive twenty vacation days each year.\n\n"
    b"## Equipment\n\nEveryone receives a laptop on their first day.\n"
)


class FakeEmbedder:
    model, dimension = "fake-embed", 384

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times, self.calls, self.stage_seen = fail_times, 0, None
        self.conn = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.conn is not None:  # what stage does the database show while embedding?
            self.stage_seen = self.conn.execute("SELECT stage FROM ingestion_jobs").fetchone()[0]
        if self.calls <= self.fail_times:
            raise RuntimeError("embedding service unavailable")
        out = []
        for text in texts:
            v = [0.0] * 384
            v[int(hashlib.md5(text.encode()).hexdigest(), 16) % 384] = 1.0
            out.append(v)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeQueue:
    def enqueue(self, *a, **k):  # type: ignore[no-untyped-def]
        return None


@pytest.fixture
def world(db_schema, tmp_path: Path):  # type: ignore[no-untyped-def]
    org = db_schema.execute(
        "INSERT INTO orgs (name) VALUES (%s) RETURNING id", (uuid.uuid4().hex,)
    ).fetchone()[0]
    user = db_schema.execute(
        "INSERT INTO users (org_id, email) VALUES (%s, 'a@x.com') RETURNING id", (org,)
    ).fetchone()[0]
    settings = Settings(_env_file=None, upload_dir=str(tmp_path / "uploads"))  # type: ignore[call-arg]
    return SimpleNamespace(conn=db_schema, org=org, user=user, settings=settings)


def submit(w, filename="handbook.md", raw=MARKDOWN):  # type: ignore[no-untyped-def]
    return enqueue_upload(
        w.conn, FakeQueue(), org_id=w.org, owner_id=w.user, visibility="org",
        filename=filename, raw=raw, settings=w.settings,
    ).job_id  # fmt: skip


def run(w, job_id, embedder=None, **kw):  # type: ignore[no-untyped-def]
    return run_job(
        w.conn, job_id, settings=kw.pop("settings", w.settings),
        embedder=embedder or FakeEmbedder(), count=approx_tokens, **kw,
    )  # fmt: skip


def job_row(w, job_id):  # type: ignore[no-untyped-def]
    return w.conn.execute(
        "SELECT status, stage, attempts, error, result FROM ingestion_jobs WHERE id = %s", (job_id,)
    ).fetchone()


def test_a_markdown_upload_becomes_searchable_chunks(world) -> None:  # type: ignore[no-untyped-def]
    job = submit(world)
    assert run(world, job) == "succeeded"
    status, stage, attempts, error, result = job_row(world, job)
    assert (status, stage, attempts, error) == ("succeeded", "done", 1, None)
    assert result["action"] == "created" and result["version"] == 1 and result["children"] >= 1
    assert result["embedding_model"] == "fake-embed"
    doc = world.conn.execute("SELECT status, active_version, title FROM documents").fetchone()
    assert doc == ("ready", 1, "Acme Handbook")
    assert world.conn.execute("SELECT document_id FROM ingestion_jobs").fetchone()[0] is not None
    with_vec = world.conn.execute(
        "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    assert with_vec == result["children"]
    hit = world.conn.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ websearch_to_tsquery('english', 'vacation')"
    ).fetchone()[0]
    assert hit >= 1  # keyword search works on the saved chunks


def test_the_database_shows_which_stage_a_running_job_is_in(world) -> None:  # type: ignore[no-untyped-def]
    embedder = FakeEmbedder()
    embedder.conn = world.conn
    run(world, submit(world), embedder)
    assert embedder.stage_seen == "embed"


def test_a_succeeded_job_delivered_again_does_nothing(world) -> None:  # type: ignore[no-untyped-def]
    job = submit(world)
    run(world, job)
    embedder = FakeEmbedder()
    assert run(world, job, embedder) == "succeeded" and embedder.calls == 0
    assert job_row(world, job)[2] == 1  # attempts unchanged


def test_the_same_bytes_under_another_name_costs_nothing(world) -> None:  # type: ignore[no-untyped-def]
    run(world, submit(world, "a.md"))
    chunks_before = world.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    embedder = FakeEmbedder()
    copy = submit(world, "copy-of-a.md")
    assert run(world, copy, embedder) == "succeeded"
    assert job_row(world, copy)[4]["action"] == "skip_duplicate" and embedder.calls == 0
    assert world.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == chunks_before


def test_an_edited_file_replaces_the_old_chunks(world) -> None:  # type: ignore[no-untyped-def]
    run(world, submit(world))
    edited = MARKDOWN.replace(b"twenty vacation days", b"thirty unlimited vacation days")
    job = submit(world, raw=edited)
    run(world, job)
    assert job_row(world, job)[4]["action"] == "updated" and job_row(world, job)[4]["version"] == 2
    texts = " ".join(r[0] for r in world.conn.execute("SELECT text FROM chunks"))
    assert "thirty unlimited" in texts and "twenty vacation" not in texts
    assert world.conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1


def test_a_corrupt_pdf_fails_at_once_with_a_readable_message(world) -> None:  # type: ignore[no-untyped-def]
    job = submit(world, "report.pdf", b"%PDF-1.4 this is garbage, not a pdf")
    assert run(world, job) == "failed"  # returns normally: no retry for a bad file
    status, _, attempts, error, _ = job_row(world, job)
    assert (status, attempts) == ("failed", 1) and "cannot open PDF" in error


def test_a_scanned_pdf_without_text_fails_with_a_clear_reason(world) -> None:  # type: ignore[no-untyped-def]
    blank = pymupdf.open()
    blank.new_page()
    job = submit(world, "scan.pdf", blank.tobytes())
    assert run(world, job) == "failed"
    assert "no extractable text" in job_row(world, job)[3]


def test_a_file_with_no_readable_text_fails(world) -> None:  # type: ignore[no-untyped-def]
    job = submit(world, "empty.md", b"---\ntitle: Nothing\n---\n")
    assert run(world, job) == "failed" and "no readable text" in job_row(world, job)[3]


def test_a_transient_error_is_retried_and_then_succeeds(world) -> None:  # type: ignore[no-untyped-def]
    job = submit(world)
    flaky = FakeEmbedder(fail_times=1)
    with pytest.raises(RuntimeError, match="unavailable"):
        run(world, job, flaky)  # re-raised, so the queue will retry it
    status, stage, attempts, error, _ = job_row(world, job)
    assert (status, stage, attempts) == ("queued", "queued", 1) and "attempt 1" in error
    assert run(world, job, flaky) == "succeeded"
    assert job_row(world, job)[:4] == ("succeeded", "done", 2, None)  # error cleared on success


def test_it_gives_up_after_the_maximum_attempts(world) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(_env_file=None, upload_dir=world.settings.upload_dir, job_max_attempts=2)  # type: ignore[call-arg]
    job = submit(world)
    broken = FakeEmbedder(fail_times=99)
    with pytest.raises(RuntimeError):
        run(world, job, broken, settings=settings)
    assert run(world, job, broken, settings=settings) == "failed"  # second attempt: no more retries
    status, _, attempts, error, _ = job_row(world, job)
    assert (status, attempts) == ("failed", 2) and "gave up after 2 attempts" in error


def test_a_tampered_stored_file_is_refused(world) -> None:  # type: ignore[no-untyped-def]
    job = submit(world)
    path = Path(world.conn.execute("SELECT raw_path FROM ingestion_jobs").fetchone()[0])
    path.write_bytes(b"# Something else entirely\n\ntext")
    assert run(world, job) == "failed" and "checksum" in job_row(world, job)[3]


def test_blurbs_are_generated_saved_and_costed_when_enabled(world) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(_env_file=None, upload_dir=world.settings.upload_dir, blurbs_enabled=True)  # type: ignore[call-arg]
    fake = lambda m, **kw: LLMResult("From the Acme handbook.", "m", 100, 20, 0.002, 0.1, 1)  # noqa: E731
    job = submit(world)
    assert run(world, job, settings=settings, complete_fn=fake) == "succeeded"
    blurbs = {r[0] for r in world.conn.execute("SELECT context_blurb FROM chunks WHERE level = 0")}
    assert blurbs == {"From the Acme handbook."}
    assert job_row(world, job)[4]["blurb_cost_usd"] > 0
    found = world.conn.execute(  # the blurb words are searchable even though the text lacks them
        "SELECT count(*) FROM chunks WHERE tsv @@ websearch_to_tsquery('english', 'handbook')"
    ).fetchone()[0]
    assert found >= 1


def test_a_blown_blurb_budget_fails_the_job_without_retrying(world) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        upload_dir=world.settings.upload_dir,
        blurbs_enabled=True,
        blurb_max_usd_per_document=0.001,
    )
    pricey = lambda m, **kw: LLMResult("x", "m", 100, 20, 0.5, 0.1, 1)  # noqa: E731
    job = submit(world)
    assert run(world, job, settings=settings, complete_fn=pricey) == "failed"  # returns: no retry
    status, _, attempts, error, _ = job_row(world, job)
    assert (status, attempts) == ("failed", 1) and "limit" in error
    assert (
        world.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    )  # nothing half-saved


def test_a_fresh_worker_connection_can_be_prepared_for_a_tenant() -> None:
    """Regression: set the tenant on a NEW (non-autocommit) connection, as process_job does."""
    import psycopg

    from app.config import Settings
    from app.retrieval.search import Scope
    from app.worker import prepare_worker_connection

    try:
        conn = psycopg.connect(Settings().database_url("owner"), connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"database not available: {type(exc).__name__}")
    org, user = uuid.uuid4(), uuid.uuid4()
    with conn:
        prepare_worker_connection(conn, Scope(org, user))
        assert conn.autocommit is True
        got = conn.execute("SELECT current_setting('app.org_id'), current_setting('app.user_id')")
        assert got.fetchone() == (str(org), str(user))
