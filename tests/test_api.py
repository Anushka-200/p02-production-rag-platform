"""The HTTP API with fakes (no model, network or key): SSE chat, dev login, tenant scoping, inspector."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.llm import LLMResult
from tests.test_search import World, vec

QUESTION = "How many vacation days do I get?"


class FakeEmbedder:
    model, dimension = "fake", 384

    def embed_query(self, text: str) -> list[float]:
        return vec(30)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec(30) for _ in texts]


class FakeStream:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces, self.result = pieces, None

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield from self.pieces
        self.result = LLMResult("".join(self.pieces), "fake-model", 100, 20, 0.002, 0.3, 1)


class SharedConnection:
    """The test's one connection, with close() disabled so each request can 'close' it safely."""

    def __init__(self, conn) -> None:  # type: ignore[no-untyped-def]
        self._conn = conn

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._conn, name)

    def close(self) -> None:
        pass


@pytest.fixture
def api(db_schema):  # type: ignore[no-untyped-def]
    w = World(db_schema)
    ann = w.person("acme", "ann")
    w.conn.execute("UPDATE users SET role = 'admin' WHERE id = %s", (ann.user_id,))
    bob = w.person("globex", "bob")
    w.chunk(
        ann, "acme", "Employees receive twenty vacation days per year", vec(30), uri="acme/leave.md"
    )
    w.chunk(
        bob, "globex", "GLOBEX SECRET vacation policy is unlimited", vec(30), uri="globex/leave.md"
    )
    w.person("acme", "member")  # a normal member of acme

    def make(pieces=("Twenty days [S1].",), env="dev", stream_calls=None):  # type: ignore[no-untyped-def]
        def stream_fn(messages, **kw):  # type: ignore[no-untyped-def]
            if stream_calls is not None:
                stream_calls.append(messages)
            return FakeStream(list(pieces))

        settings = Settings(
            _env_file=None, environment=env, reranker_provider="none", retrieval_candidates=10
        )  # type: ignore[call-arg]
        app = create_app(
            settings, connect=lambda: SharedConnection(w.conn), embedder=FakeEmbedder(), reranker=None,
            stream_fn=stream_fn, complete_fn=lambda *a, **k: pytest.fail("no JSON fallback expected"),
        )  # fmt: skip
        return TestClient(app)

    return SimpleNamespace(make=make, w=w)


def read_events(response) -> list[tuple[str, dict]]:  # type: ignore[no-untyped-def,type-arg]
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: ") and name:
            events.append((name, json.loads(line[6:])))
    return events


def chat(client, who="ann@x.com", question=QUESTION):  # type: ignore[no-untyped-def]
    with client.stream(
        "POST", "/chat", json={"question": question}, headers={"X-Dev-User": who}
    ) as r:
        return r.status_code, r.headers.get("content-type", ""), read_events(r)


def test_health_needs_no_login(api) -> None:  # type: ignore[no-untyped-def]
    assert api.make().get("/health").json() == {"status": "ok"}


def test_chat_streams_sources_first_then_tokens_then_one_done_event(api) -> None:  # type: ignore[no-untyped-def]
    status, ctype, events = chat(api.make(pieces=("Twenty ", "days ", "[S1].")))
    assert status == 200 and ctype.startswith("text/event-stream")
    kinds = [name for name, _ in events]
    assert kinds == ["sources", "token", "token", "token", "done"]
    assert events[0][1]["sources"][0]["document"].startswith(
        "acme/leave.md"
    )  # the UI can show sources early
    assert "".join(d["text"] for n, d in events if n == "token") == "Twenty days [S1]."
    done = events[-1][1]
    assert done["text"] == "Twenty days [S1]." and not done["refused"] and not done["replaced"]
    assert done["citations"][0]["id"] == "S1" and done["cost_usd"] == 0.002
    assert {"retrieve_ms", "rerank_ms", "first_token_ms", "total_ms"} <= set(done["timings_ms"])


def test_not_found_becomes_a_refusal_event_without_tokens(api) -> None:  # type: ignore[no-untyped-def]
    _, _, events = chat(api.make(pieces=("NOT_FOUND",)))
    assert [n for n, _ in events] == ["sources", "done"] and events[-1][1]["refused"]


def test_the_tenant_comes_from_the_login_never_from_the_body(api) -> None:  # type: ignore[no-untyped-def]
    client = api.make()
    _, _, mine = chat(client, "ann@x.com")
    assert [s["document"][:13] for s in mine[0][1]["sources"]] == [
        "acme/leave.md"
    ]  # not globex's chunk
    _, _, theirs = chat(client, "bob@x.com")
    assert [s["document"][:15] for s in theirs[0][1]["sources"]] == ["globex/leave.md"]
    with client.stream(  # forged organisation fields in the body are simply ignored
        "POST",
        "/chat",
        json={"question": QUESTION, "org_id": str(api.w.orgs["globex"])},
        headers={"X-Dev-User": "ann@x.com"},
    ) as r:
        forged = read_events(r)
    assert [s["document"][:13] for s in forged[0][1]["sources"]] == ["acme/leave.md"]


def test_login_failures_are_401_and_dev_login_is_off_outside_dev(api) -> None:  # type: ignore[no-untyped-def]
    client = api.make()
    assert client.post("/chat", json={"question": "q"}).status_code == 401  # no header
    assert (
        client.post(
            "/chat", json={"question": "q"}, headers={"X-Dev-User": "nobody@x.com"}
        ).status_code
        == 401
    )
    prod = api.make(env="production")
    assert (
        prod.post("/chat", json={"question": "q"}, headers={"X-Dev-User": "ann@x.com"}).status_code
        == 401
    )


def test_bad_input_is_rejected_with_422(api) -> None:  # type: ignore[no-untyped-def]
    client = api.make()
    headers = {"X-Dev-User": "ann@x.com"}
    assert client.post("/chat", json={"question": ""}, headers=headers).status_code == 422
    assert client.post("/chat", json={"question": "x" * 2001}, headers=headers).status_code == 422
    assert client.post("/chat", json={}, headers=headers).status_code == 422


def test_an_internal_error_becomes_an_error_event_that_leaks_nothing(api) -> None:  # type: ignore[no-untyped-def]
    client = api.make()

    class Boom(FakeEmbedder):
        def embed_query(self, text: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("secret database password is hunter2")

    client.app.router.lifespan_context  # noqa: B018  (the app object is reused below)
    broken = create_app(
        Settings(_env_file=None, reranker_provider="none"),  # type: ignore[call-arg]
        connect=lambda: SharedConnection(api.w.conn), embedder=Boom(), reranker=None,
    )  # fmt: skip
    _, _, events = chat(TestClient(broken))
    assert events == [("error", {"message": "something went wrong; please try again"})]
    assert "hunter2" not in json.dumps(events)


# ---- the inspector (2.8) -----------------------------------------------------------------------
def inspect(client, who="ann@x.com", mode=None):  # type: ignore[no-untyped-def]
    body = {"question": QUESTION, **({"mode": mode} if mode else {})}
    return client.post("/admin/inspect", json=body, headers={"X-Dev-User": who})


def test_the_inspector_shows_every_score_for_every_candidate_without_calling_the_model(api) -> None:  # type: ignore[no-untyped-def]
    calls: list = []  # type: ignore[type-arg]
    r = inspect(api.make(stream_calls=calls))
    assert r.status_code == 200 and calls == []  # no LLM call, no cost
    body = r.json()
    assert (
        body["mode"] == "hybrid"
        and body["keyword_query"]
        and {"embed_ms", "dense_ms"} <= set(body["timings_ms"])
    )
    c = body["candidates"][0]
    assert (
        c["document"].startswith("acme/leave.md")
        and c["dense_rank"] == 1
        and c["dense_distance"] == pytest.approx(0.0, abs=1e-3)
    )
    assert c["keyword_rank"] == 1 and c["keyword_score"] > 0 and c["rrf_score"] > 0
    assert c["context_label"] == "S1" and "twenty vacation days" in c["text_preview"]
    assert c["kept_for_answer"] is True and "rerank_rank" in c
    assert all("GLOBEX" not in x["text_preview"] for x in body["candidates"])  # still tenant-scoped


def test_the_inspector_is_admin_only_and_modes_are_validated(api) -> None:  # type: ignore[no-untyped-def]
    client = api.make()
    assert inspect(client, who="member@x.com").status_code == 403
    assert inspect(client, who="nobody@x.com").status_code == 401
    assert inspect(client, mode="nonsense").status_code == 422
    dense = inspect(client, mode="dense").json()
    assert dense["mode"] == "dense" and all(c["keyword_rank"] is None for c in dense["candidates"])


def test_documents_list_and_only_the_owner_can_change_visibility(api) -> None:  # type: ignore[no-untyped-def]
    client = api.make()
    doc = api.w.conn.execute(
        "SELECT d.id FROM documents d JOIN orgs o ON o.id = d.org_id WHERE o.name = 'acme'"
    ).fetchone()[0]
    listed = client.get("/documents", headers={"X-Dev-User": "ann@x.com"}).json()
    assert [d["source_uri"].startswith("acme/leave.md") for d in listed] == [True]

    url = f"/documents/{doc}/visibility"
    colleague = client.patch(
        url, json={"visibility": "private"}, headers={"X-Dev-User": "member@x.com"}
    )
    assert colleague.status_code == 404  # not the owner
    owner = client.patch(url, json={"visibility": "private"}, headers={"X-Dev-User": "ann@x.com"})
    assert owner.status_code == 200
    bad = client.patch(url, json={"visibility": "secret"}, headers={"X-Dev-User": "ann@x.com"})
    assert bad.status_code == 422  # only private | org | public exist
