import io
import uuid
from pathlib import Path
from types import SimpleNamespace

import docx
import pymupdf
import pytest

from app.config import Settings
from app.ingest.store import file_hash
from app.ingest.upload import UploadError, enqueue_upload, validate_upload

MB = 1_000_000


def real_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Hello policy")
    return doc.tobytes()  # type: ignore[no-any-return]


def real_docx() -> bytes:
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph("Hello policy")
    d.save(buf)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("filename", "raw", "expected"),
    [
        ("a.md", b"# Title\n\ntext", "md"),
        ("A.MARKDOWN", b"# Title", "md"),
        ("page.mdx", b"import X from 'x'\n\n# Title\n\n<X />", "md"),
        ("notes.txt", b"plain text", "txt"),
        ("page.html", b"<p>hi</p>", "html"),
        ("policy.pdf", real_pdf(), "pdf"),
        ("policy.docx", real_docx(), "docx"),
    ],
)
def test_valid_uploads_are_accepted(filename: str, raw: bytes, expected: str) -> None:
    assert validate_upload(filename, raw, 25 * MB) == expected


@pytest.mark.parametrize(
    ("filename", "raw", "message"),
    [
        ("virus.exe", b"MZ...", "unsupported"),
        ("noext", b"text", "unsupported"),
        ("empty.md", b"", "empty"),
        ("fake.pdf", b"just some text", "not a real PDF"),
        ("fake.docx", b"not a zip at all", "not a real DOCX"),
        ("fake.txt", b"abc\x00\x01\x02binary", "binary"),
    ],
)
def test_bad_uploads_are_refused_with_a_readable_reason(
    filename: str, raw: bytes, message: str
) -> None:
    with pytest.raises(UploadError, match=message):
        validate_upload(filename, raw, 25 * MB)


def test_a_zip_that_is_not_a_docx_is_refused() -> None:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(UploadError, match="not a real DOCX"):
        validate_upload("x.docx", buf.getvalue(), 25 * MB)


def test_size_limit() -> None:
    with pytest.raises(UploadError, match="too large"):
        validate_upload("big.txt", b"a" * 2000, 1000)


# ---- enqueueing ---------------------------------------------------------------------------------
class FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []  # type: ignore[type-arg]

    def enqueue(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((args, kwargs))


@pytest.fixture
def world(db_schema, tmp_path: Path):  # type: ignore[no-untyped-def]
    org = db_schema.execute(
        "INSERT INTO orgs (name) VALUES (%s) RETURNING id", (uuid.uuid4().hex,)
    ).fetchone()[0]
    user = db_schema.execute(
        "INSERT INTO users (org_id, email) VALUES (%s, 'a@x.com') RETURNING id", (org,)
    ).fetchone()[0]
    settings = Settings(_env_file=None, upload_dir=str(tmp_path / "uploads"))  # type: ignore[call-arg]
    return SimpleNamespace(conn=db_schema, org=org, user=user, settings=settings, queue=FakeQueue())


def submit(w, filename: str = "handbook.md", raw: bytes = b"# Title\n\nSome text."):  # type: ignore[no-untyped-def]
    return enqueue_upload(
        w.conn, w.queue, org_id=w.org, owner_id=w.user, visibility="org",
        filename=filename, raw=raw, settings=w.settings,
    )  # fmt: skip


def test_enqueue_creates_a_job_stores_the_file_and_queues_it_with_retries(world) -> None:  # type: ignore[no-untyped-def]
    result = submit(world)
    assert result.created and result.status == "queued"
    row = world.conn.execute(
        "SELECT status, stage, source_uri, source_type, content_hash, raw_path FROM ingestion_jobs"
    ).fetchone()
    assert row[:4] == ("queued", "queued", "handbook.md", "md") and row[4] == file_hash(
        b"# Title\n\nSome text."
    )
    assert Path(row[5]).read_bytes() == b"# Title\n\nSome text."  # the original is stored
    ((args, kwargs),) = world.queue.calls
    assert args[:2] == ("app.worker.process_job", str(result.job_id))
    assert kwargs["retry"].max == 2 and kwargs["job_id"] == str(result.job_id)


def test_the_same_upload_twice_is_one_job_queued_once(world) -> None:  # type: ignore[no-untyped-def]
    first, second = submit(world), submit(world)
    assert second.job_id == first.job_id and not second.created
    assert len(world.queue.calls) == 1
    assert world.conn.execute("SELECT count(*) FROM ingestion_jobs").fetchone()[0] == 1


def test_uploading_a_failed_file_again_retries_the_job(world) -> None:  # type: ignore[no-untyped-def]
    first = submit(world)
    world.conn.execute("UPDATE ingestion_jobs SET status='failed', error='boom', attempts=3")
    again = submit(world)
    assert again.job_id == first.job_id and len(world.queue.calls) == 2
    assert world.conn.execute("SELECT status, error, attempts FROM ingestion_jobs").fetchone() == (
        "queued",
        None,
        0,
    )


def test_a_path_in_the_filename_cannot_escape_the_upload_folder(world) -> None:  # type: ignore[no-untyped-def]
    submit(world, filename="../../etc/passwd.md")
    uri, path = world.conn.execute("SELECT source_uri, raw_path FROM ingestion_jobs").fetchone()
    assert uri == "passwd.md"
    assert Path(path).resolve().is_relative_to(Path(world.settings.upload_dir).resolve())


def test_a_bad_upload_creates_no_job(world) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(UploadError):
        submit(world, filename="x.exe", raw=b"MZ")
    assert (
        world.conn.execute("SELECT count(*) FROM ingestion_jobs").fetchone()[0] == 0
        and not world.queue.calls
    )


def test_enqueue_really_reaches_redis(world) -> None:  # type: ignore[no-untyped-def]
    from redis import Redis
    from rq import Queue

    connection = Redis.from_url(world.settings.redis_url)
    try:
        connection.ping()
    except Exception as e:
        pytest.skip(f"redis not available: {type(e).__name__}")
    queue = Queue(f"test-{uuid.uuid4().hex[:8]}", connection=connection)
    try:
        result = enqueue_upload(
            world.conn, queue, org_id=world.org, owner_id=world.user, visibility="org",
            filename="a.md", raw=b"# T\n\ntext", settings=world.settings,
        )  # fmt: skip
        assert queue.count == 1
        job = queue.jobs[0]
        assert (
            job.args == (str(result.job_id), str(world.org), str(world.user))
            and job.retries_left == 2
            and job.retry_intervals == [10, 60, 300]
        )
    finally:
        queue.empty()
        queue.delete()


def test_a_bulk_source_can_pass_a_path_as_the_document_identity(world) -> None:  # type: ignore[no-untyped-def]
    def add(text: bytes, uri: str):  # type: ignore[no-untyped-def]
        return enqueue_upload(
            world.conn, world.queue, org_id=world.org, owner_id=world.user, visibility="org",
            filename="_index.md", raw=text, settings=world.settings, source_uri=uri,
        )  # fmt: skip

    a, b = (
        add(b"# A\n\ntext a", "benefits/_index.md"),
        add(b"# B\n\ntext b", "compensation/_index.md"),
    )
    assert (
        a.job_id != b.job_id
    )  # two different pages that share a file name stay separate documents
    uris = sorted(r[0] for r in world.conn.execute("SELECT source_uri FROM ingestion_jobs"))
    assert uris == ["benefits/_index.md", "compensation/_index.md"]


@pytest.mark.parametrize(
    ("raw_uri", "clean"),
    [("a/b/c.md", "a/b/c.md"), ("../../etc/x.md", "etc/x.md"), ("/abs/path.md", "abs/path.md"),
     ("a\\b\\c.md", "a/b/c.md"), ("./a//b.md", "a/b.md")],
)  # fmt: skip
def test_source_uri_is_normalised(raw_uri: str, clean: str) -> None:
    from app.ingest.upload import normalize_source_uri

    assert normalize_source_uri(raw_uri) == clean


@pytest.mark.parametrize("bad", ["", "..", "a\x00b.md", "x" * 501])
def test_unusable_source_uris_are_refused(bad: str) -> None:
    from app.ingest.upload import normalize_source_uri

    with pytest.raises(UploadError, match="invalid file name"):
        normalize_source_uri(bad)
