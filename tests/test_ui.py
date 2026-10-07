"""The Streamlit pages and their API client, against the real FastAPI app (fake model, real DB)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.api.main import create_app
from app.api.security import hash_password
from app.config import Settings
from tests.test_api import FakeEmbedder, FakeStream, SharedConnection
from tests.test_memory import fake_llm
from tests.test_search import World, vec
from ui.api import ApiError, PlatformClient, parse_sse

SECRET = "u" * 40
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stack(db_schema):  # type: ignore[no-untyped-def]
    w = World(db_schema)
    ann, amy, bob = w.person("acme", "ann"), w.person("acme", "amy"), w.person("globex", "bob")
    for who, pw in ((ann, "ann-pass-1"), (amy, "amy-pass-1"), (bob, "bob-pass-1")):
        db_schema.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s", (hash_password(pw), who.user_id)
        )
    db_schema.execute("UPDATE users SET role = 'admin' WHERE id = %s", (ann.user_id,))
    w.chunk(ann, "acme", "Employees receive twenty vacation days", vec(30), uri="acme/leave.md")
    w.chunk(bob, "globex", "GLOBEX unlimited vacation", vec(30), uri="globex/leave.md")
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, environment="prod", jwt_secret=SECRET, reranker_provider="none",
        retrieval_candidates=10, memory_enabled=True,
    )  # fmt: skip
    app = create_app(
        settings, connect=lambda: SharedConnection(db_schema), embedder=FakeEmbedder(), reranker=None,
        stream_fn=lambda messages, **kw: FakeStream(["Twenty days [S1]."]),
        complete_fn=lambda *a, **k: pytest.fail("no JSON fallback expected"),
        memory_complete_fn=fake_llm({"facts": []}),
    )  # fmt: skip
    http = TestClient(app)

    def login(email: str, password: str) -> PlatformClient:
        api = PlatformClient("http://test", http=http)
        api.login(email, password)
        return api

    return login, w, http


# ---- the client ---------------------------------------------------------------------------------
def test_parse_sse_turns_event_text_into_pairs() -> None:
    lines = iter(
        ["event: token", 'data: {"text": "Twenty "}', "", "event: done", 'data: {"text": "x"}']
    )
    assert list(parse_sse(lines)) == [("token", {"text": "Twenty "}), ("done", {"text": "x"})]


def test_login_me_and_a_wrong_password(stack) -> None:  # type: ignore[no-untyped-def]
    login, _, http = stack
    api = login("ann@x.com", "ann-pass-1")
    assert api.me() == {"email": "ann@x.com", "role": "admin"}
    with pytest.raises(ApiError) as exc:
        PlatformClient("http://test", http=http).login("ann@x.com", "wrong")
    assert exc.value.status == 401 and "wrong email or password" in exc.value.detail


def test_chat_streams_sources_tokens_and_a_done_event_with_the_conversation(stack) -> None:  # type: ignore[no-untyped-def]
    login, *_ = stack
    api = login("ann@x.com", "ann-pass-1")
    events = list(api.chat("How many vacation days?"))
    names = [n for n, _ in events]
    assert names[0] == "sources" and "token" in names and names[-1] == "done"
    done = events[-1][1]
    assert done["citations"][0]["id"] == "S1" and done["conversation_id"]
    assert [c["id"] for c in api.conversations()] == [done["conversation_id"]]
    assert [m["role"] for m in api.messages(done["conversation_id"])] == ["user", "assistant"]
    api.delete_conversation(done["conversation_id"])
    assert api.conversations() == []


def test_a_request_without_a_token_is_an_api_error(stack) -> None:  # type: ignore[no-untyped-def]
    _, _, http = stack
    with pytest.raises(ApiError) as exc:
        list(PlatformClient("http://test", http=http).chat("hi"))
    assert exc.value.status == 401


# ---- the admin views ------------------------------------------------------------------------------
def test_admin_views_show_status_chunks_and_previews_for_the_admins_org_only(stack) -> None:  # type: ignore[no-untyped-def]
    login, w, _ = stack
    api = login("ann@x.com", "ann-pass-1")
    docs = api.admin_documents()
    assert [d["source_uri"].split("-")[0] for d in docs] == ["acme/leave.md"]  # not globex
    assert (
        docs[0]["status"] == "ready" and docs[0]["chunks"] == 1 and docs[0]["owner"] == "ann@x.com"
    )
    chunks = api.admin_chunks(str(docs[0]["id"]))
    assert chunks[0]["preview"].startswith("Employees receive") and chunks[0]["embedded"] is True
    assert api.admin_chunks(str(docs[0]["id"]), level=1) == []  # no parent chunks in this document
    globex_doc = w.conn.execute(
        "SELECT id FROM documents WHERE source_uri LIKE 'globex%'"
    ).fetchone()[0]
    assert api.admin_chunks(str(globex_doc)) == []  # another org's document id shows nothing


def test_the_admin_endpoints_refuse_ordinary_members(stack) -> None:  # type: ignore[no-untyped-def]
    login, *_ = stack
    member = login("amy@x.com", "amy-pass-1")
    for call in (member.admin_documents, member.admin_jobs, lambda: member.inspect("q")):
        with pytest.raises(ApiError) as exc:
            call()
        assert exc.value.status == 403


def test_the_inspector_returns_scored_candidates(stack) -> None:  # type: ignore[no-untyped-def]
    login, *_ = stack
    result = login("ann@x.com", "ann-pass-1").inspect("vacation days", "dense")
    assert result["mode"] == "dense" and result["candidates"][0]["document"].startswith("acme/")
    assert {"dense_distance", "rrf_score", "rerank_score", "kept_for_answer"} <= set(
        result["candidates"][0]
    )


# ---- the pages (Streamlit's headless test runner) ----------------------------------------------
def page(name: str, client: PlatformClient | None = None, me: dict | None = None) -> AppTest:  # type: ignore[type-arg]
    at = AppTest.from_file(str(ROOT / name), default_timeout=60)
    if client is not None:
        at.session_state["client"] = client
        at.session_state["me"] = me
    return at.run()


def test_the_chat_page_asks_for_a_login_first() -> None:
    at = page("ui/Chat.py")
    assert not at.exception and at.title[0].value == "Sign in" and not at.chat_input


def test_a_signed_in_user_can_ask_and_sees_the_answer_with_sources_and_the_ai_notice(stack) -> None:  # type: ignore[no-untyped-def]
    login, *_ = stack
    at = page(
        "ui/Chat.py", login("amy@x.com", "amy-pass-1"), {"email": "amy@x.com", "role": "member"}
    )
    assert not at.exception
    assert any("AI assistant" in i.value for i in at.info)  # the disclosure line
    at.chat_input[0].set_value("How many vacation days?").run()
    assert not at.exception
    texts = [m.markdown[0].value for m in at.chat_message]
    assert "How many vacation days?" in texts[0] and texts[1].startswith("Twenty days")
    assert at.session_state["conversation_id"]  # the next question continues this conversation
    assert any("Sources" in e.label for e in at.expander)


def test_the_admin_page_is_closed_to_members_and_open_to_admins(stack) -> None:  # type: ignore[no-untyped-def]
    login, *_ = stack
    member = page(
        "ui/pages/1_Admin.py",
        login("amy@x.com", "amy-pass-1"),
        {"email": "amy@x.com", "role": "member"},
    )
    assert not member.exception and member.warning and not member.tabs
    admin = page(
        "ui/pages/1_Admin.py",
        login("ann@x.com", "ann-pass-1"),
        {"email": "ann@x.com", "role": "admin"},
    )
    assert not admin.exception
    assert [t.label for t in admin.tabs] == [
        "Retrieval inspector",
        "Ingestion status",
        "Chunk previews",
        "Leads & handoffs",
    ]


def test_the_agents_trace_is_described_in_plain_lines() -> None:
    from ui.common import describe_steps

    steps = [
        {"n": 1, "kind": "plan", "note": "connections: who owns alerts"},
        {"n": 2, "kind": "search", "tool": "connections", "query": "who owns alerts", "hits": 1},
        {"n": 3, "kind": "check", "note": "enough"},
    ]
    lines = describe_steps(steps)
    assert lines[0].startswith("planned") and "connections" in lines[1] and "(1 hit)" in lines[1]
    assert lines[2] == "checked the results: enough" and describe_steps([]) == []
