"""POST /documents and GET /jobs/{id}: who owns an upload, what is refused."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from tests.test_api import FakeEmbedder, SharedConnection
from tests.test_search import World

ANN = {"X-Dev-User": "ann@x.com"}
AMY = {"X-Dev-User": "amy@x.com"}


class FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple] = []  # type: ignore[type-arg]

    def enqueue(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(args)


@pytest.fixture
def rig(db_schema, tmp_path):  # type: ignore[no-untyped-def]
    w = World(db_schema)
    w.ann, w.amy = w.person("acme", "ann"), w.person("acme", "amy")  # type: ignore[attr-defined]
    queue = FakeQueue()
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, environment="dev", reranker_provider="none", upload_dir=str(tmp_path),
        max_upload_mb=1,
    )  # fmt: skip
    app = create_app(
        settings, connect=lambda: SharedConnection(db_schema), embedder=FakeEmbedder(),
        reranker=None, queue=queue,
    )  # fmt: skip
    return w, queue, TestClient(app)


def upload(client, name="leave.md", content=b"# Leave\n\nTwenty days.", headers=ANN, **form):  # type: ignore[no-untyped-def]
    return client.post("/documents", files={"file": (name, content)}, data=form, headers=headers)


def test_an_upload_is_queued_for_the_logged_in_user_and_starts_private(rig) -> None:  # type: ignore[no-untyped-def]
    w, queue, client = rig
    r = upload(client)
    assert r.status_code == 202 and r.json()["duplicate"] is False
    job_id, org_id, owner_id = queue.calls[0][1:4]
    assert (org_id, owner_id) == (
        str(w.ann.org_id),
        str(w.ann.user_id),
    )  # from the login, not the form
    row = w.conn.execute("SELECT visibility, owner_id FROM ingestion_jobs").fetchone()
    assert row == ("private", w.ann.user_id)


def test_the_same_file_twice_is_one_job(rig) -> None:  # type: ignore[no-untyped-def]
    _, queue, client = rig
    first, second = upload(client), upload(client)
    assert second.json()["duplicate"] is True and second.json()["job_id"] == first.json()["job_id"]
    assert len(queue.calls) == 1


def test_a_user_can_upload_as_shared_but_not_as_something_else(rig) -> None:  # type: ignore[no-untyped-def]
    w, _, client = rig
    assert upload(client, visibility="org").status_code == 202
    assert w.conn.execute("SELECT visibility FROM ingestion_jobs").fetchone()[0] == "org"
    assert (
        upload(client, name="b.md", content=b"# B\n\ntext", visibility="secret").status_code == 422
    )


def test_bad_uploads_are_refused_with_a_clear_reason(rig) -> None:  # type: ignore[no-untyped-def]
    _, queue, client = rig
    assert "unsupported" in upload(client, name="virus.exe", content=b"MZ").json()["detail"]
    assert "empty" in upload(client, content=b"").json()["detail"]
    assert "too large" in upload(client, content=b"x" * 1_500_000).json()["detail"]
    assert "not a real PDF" in upload(client, name="a.pdf", content=b"hello").json()["detail"]
    assert queue.calls == []


def test_uploading_needs_a_login(rig) -> None:  # type: ignore[no-untyped-def]
    assert upload(rig[2], headers={}).status_code == 401


def test_a_job_is_visible_only_to_its_owner(rig) -> None:  # type: ignore[no-untyped-def]
    _, _, client = rig
    job = upload(client).json()["job_id"]
    assert client.get(f"/jobs/{job}", headers=ANN).json()["status"] == "queued"
    assert client.get(f"/jobs/{job}", headers=AMY).status_code == 404  # same org, other user


def test_only_an_admin_can_upload_or_publish_a_public_document(rig) -> None:  # type: ignore[no-untyped-def]
    w, queue, client = rig
    assert upload(client, visibility="public").status_code == 403  # ann is a plain member
    assert queue.calls == []
    w.conn.execute("UPDATE users SET role = 'admin' WHERE id = %s", (w.ann.user_id,))
    assert upload(client, visibility="public").status_code == 202
    assert w.conn.execute("SELECT visibility FROM ingestion_jobs").fetchone()[0] == "public"


def test_changing_a_document_to_public_needs_admin_and_says_403_not_404(rig) -> None:  # type: ignore[no-untyped-def]
    w, _, client = rig
    w.chunk(w.ann, "acme", "Handbook text", None, visibility="org", uri="h.md")
    doc = w.conn.execute("SELECT id FROM documents").fetchone()[0]
    url = f"/documents/{doc}/visibility"
    assert client.patch(url, json={"visibility": "public"}, headers=ANN).status_code == 403
    w.conn.execute("UPDATE users SET role = 'admin' WHERE id = %s", (w.ann.user_id,))
    assert client.patch(url, json={"visibility": "public"}, headers=ANN).status_code == 200
    assert w.conn.execute("SELECT visibility FROM chunks").fetchone()[0] == "public"
