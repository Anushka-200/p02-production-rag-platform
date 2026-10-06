"""The embeddable widget: visitors, CORS, rate limits, budget, leads, handoff, feedback."""

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.db.tenant import set_session_tenant
from app.ratelimit import MemoryRateLimiter, RedisRateLimiter
from app.retrieval.search import Scope
from app.widget import add_lead, create_session, record_usage, spent_today
from scripts.create_site import create_site
from tests.test_api import FakeEmbedder, FakeStream, SharedConnection
from tests.test_rls import app_connection, world  # noqa: F401, F811
from tests.test_search import World, vec

ORIGIN = "https://www.acme.com"
PUBLIC, INTERNAL = "Acme opens at 8:30 every weekday.", "INTERNAL salary bands are confidential"
ADMIN = {"X-Dev-User": "ann@x.com"}


def parse(response) -> list[tuple[str, dict]]:  # type: ignore[no-untyped-def,type-arg]
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: ") and name:
            events.append((name, json.loads(line[6:])))
    return events


class Rig:
    def __init__(self, db_schema, **settings):  # type: ignore[no-untyped-def]
        self.w = w = World(db_schema)
        self.ann = w.person("acme", "ann")
        w.conn.execute("UPDATE users SET role = 'admin' WHERE id = %s", (self.ann.user_id,))
        w.person("acme", "amy")
        w.chunk(self.ann, "acme", PUBLIC, vec(30), visibility="public", uri="hours.md")
        w.chunk(self.ann, "acme", INTERNAL, vec(30), visibility="org", uri="salary.md")
        self.key = create_site(db_schema, "acme", "Acme website", [ORIGIN])
        self.other_key = create_site(db_schema, "acme", "Acme shop", ["https://shop.acme.com"])
        self.prompts: list = []  # type: ignore[type-arg]
        self.limiter = MemoryRateLimiter()

        def stream_fn(messages, **kw):  # type: ignore[no-untyped-def]
            self.prompts.append(messages)
            return FakeStream(["We open at 8:30 [S1]."])

        cfg = Settings(  # type: ignore[call-arg]
            _env_file=None, environment="dev", reranker_provider="none", retrieval_candidates=10,
            **settings,
        )  # fmt: skip
        self.client = TestClient(
            create_app(
                cfg,
                connect=lambda: SharedConnection(db_schema),
                embedder=FakeEmbedder(),
                reranker=None,
                stream_fn=stream_fn,
                limiter=self.limiter,
                complete_fn=lambda *a, **k: pytest.fail("no JSON fallback expected"),
            )  # fmt: skip
        )

    def chat(self, question="When do you open?", session_id=None, key=None, origin=ORIGIN, **kw):  # type: ignore[no-untyped-def]
        body = {"question": question, **({"session_id": str(session_id)} if session_id else {})}
        headers = {"Origin": origin, **kw.pop("headers", {})} if origin else kw.pop("headers", {})
        return self.client.post(f"/widget/{key or self.key}/chat", json=body, headers=headers)

    def done(self, response) -> dict:  # type: ignore[no-untyped-def,type-arg]
        return next(d for n, d in parse(response) if n == "done")

    def post(self, action: str, body: dict, key=None, origin=ORIGIN):  # type: ignore[no-untyped-def,type-arg]
        return self.client.post(
            f"/widget/{key or self.key}/{action}", json=body, headers={"Origin": origin}
        )


@pytest.fixture
def rig(db_schema) -> Rig:  # type: ignore[no-untyped-def]
    return Rig(db_schema)


# ---- the gate: key, origin, CORS ---------------------------------------------------------------
def test_config_gives_the_site_name_and_the_ai_disclosure(rig: Rig) -> None:
    r = rig.client.get(f"/widget/{rig.key}/config", headers={"Origin": ORIGIN})
    assert r.status_code == 200 and r.json()["name"] == "Acme website"
    assert "AI assistant" in r.json()["disclosure"]


@pytest.mark.parametrize(
    ("origin", "status"),
    [("https://evil.example", 403), (None, 403), ("https://shop.acme.com", 403)],
)
def test_a_site_key_only_works_from_its_own_origin(rig: Rig, origin, status) -> None:  # type: ignore[no-untyped-def]
    assert (
        rig.chat(origin=origin).status_code == status
    )  # shop.acme.com has its OWN key, not this one


def test_unknown_and_disabled_keys_are_401(rig: Rig) -> None:
    assert rig.chat(key="pk_nope").status_code == 401
    rig.w.conn.execute("UPDATE sites SET enabled = false WHERE site_key = %s", (rig.key,))
    assert rig.chat().status_code == 401


def test_cors_allows_the_sites_own_origin_and_nobody_else(rig: Rig) -> None:
    ok = rig.client.options(
        f"/widget/{rig.key}/chat",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert ok.status_code == 204 and ok.headers["access-control-allow-origin"] == ORIGIN
    bad = rig.client.options(
        f"/widget/{rig.key}/chat",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert bad.status_code == 403 and "access-control-allow-origin" not in bad.headers
    real = rig.chat()
    assert (
        real.headers["access-control-allow-origin"] == ORIGIN and real.headers["vary"] == "Origin"
    )
    other = rig.client.get(f"/widget/{rig.key}/config", headers={"Origin": "https://evil.example"})
    assert (
        "access-control-allow-origin" not in other.headers
    )  # and the request itself is refused (403)
    assert other.status_code == 403


def test_the_script_is_served_with_a_javascript_type(rig: Rig) -> None:
    r = rig.client.get("/widget.js")
    assert (
        r.status_code == 200
        and "javascript" in r.headers["content-type"]
        and "attachShadow" in r.text
    )


# ---- chatting as a visitor ---------------------------------------------------------------------
def test_a_visitor_gets_a_cited_answer_from_public_documents_only(rig: Rig) -> None:
    done = rig.done(rig.chat())
    assert done["text"].startswith("We open at 8:30") and done["citations"][0]["id"] == "S1"
    prompt = json.dumps(rig.prompts)
    assert (
        PUBLIC in prompt and INTERNAL not in prompt
    )  # the org-internal text never reaches the model
    assert {"model", "cost_usd", "timings_ms", "reason"}.isdisjoint(
        done
    )  # internals are not exposed
    assert set(done["citations"][0]) == {"id", "document", "section", "pages"}


def test_the_session_keeps_the_conversation_and_stores_both_sides(rig: Rig) -> None:
    first = rig.done(rig.chat("When do you open?"))
    second = rig.done(rig.chat("And on Friday?", session_id=first["session_id"]))
    assert second["session_id"] == first["session_id"]
    rows = rig.w.conn.execute("SELECT role FROM widget_messages ORDER BY created_at, id").fetchall()
    assert [r[0] for r in rows] == ["user", "assistant", "user", "assistant"]


def test_a_session_id_from_another_site_is_not_adopted(rig: Rig) -> None:
    mine = rig.done(rig.chat())
    shop = rig.client.post(
        f"/widget/{rig.other_key}/chat",
        json={"question": "hi", "session_id": mine["session_id"]},
        headers={"Origin": "https://shop.acme.com"},
    )
    assert rig.done(shop)["session_id"] != mine["session_id"]  # a fresh session, not a takeover


def test_hostile_input_is_handled(rig: Rig) -> None:
    for q in ["x" * 501, ""]:
        assert rig.chat(q).status_code == 422
    assert rig.chat("vacation\x00 days").status_code == 200  # NUL stripped, not a server error
    assert rig.chat("'; DROP TABLE chunks; --").status_code == 200
    assert rig.w.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2


# ---- rate limit and budget ---------------------------------------------------------------------
def test_too_many_requests_from_one_address_get_429_with_retry_after(rig: Rig) -> None:
    rig.w.conn.execute("UPDATE sites SET rate_limit_per_minute = 3 WHERE site_key = %s", (rig.key,))
    codes = [rig.chat(headers={}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    blocked = rig.chat()
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) >= 1


def test_each_visitor_address_has_its_own_allowance_behind_a_trusted_proxy(db_schema) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(db_schema, trust_proxy=True)
    rig.w.conn.execute("UPDATE sites SET rate_limit_per_minute = 1 WHERE site_key = %s", (rig.key,))
    a = lambda: rig.chat(headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}).status_code  # noqa: E731
    b = lambda: rig.chat(headers={"X-Forwarded-For": "2.2.2.2"}).status_code  # noqa: E731
    assert (a(), a(), b()) == (200, 429, 200)


def test_without_a_trusted_proxy_a_forged_forwarded_header_is_ignored(rig: Rig) -> None:
    rig.w.conn.execute("UPDATE sites SET rate_limit_per_minute = 1 WHERE site_key = %s", (rig.key,))
    codes = [rig.chat(headers={"X-Forwarded-For": f"9.9.9.{i}"}).status_code for i in range(3)]
    assert codes == [200, 429, 429]  # rotating the header does not escape the limit


def test_the_memory_limiter_resets_in_the_next_window() -> None:
    now = [1000.0]
    limiter = MemoryRateLimiter(clock=lambda: now[0])
    assert [limiter.hit("k", 2)[0] for _ in range(3)] == [True, True, False]
    now[0] += 60
    assert limiter.hit("k", 2)[0] is True


def test_the_redis_limiter_counts_and_expires_keys() -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.data: dict[str, int] = {}
            self.ttl: dict[str, int] = {}

        def pipeline(self):  # type: ignore[no-untyped-def]
            outer = self
            ops: list = []  # type: ignore[type-arg]

            class P:
                def incr(self, k):  # type: ignore[no-untyped-def]
                    ops.append(("incr", k))

                def expire(self, k, s):  # type: ignore[no-untyped-def]
                    ops.append(("expire", k, s))

                def execute(self):  # type: ignore[no-untyped-def]
                    out = []
                    for op in ops:
                        if op[0] == "incr":
                            outer.data[op[1]] = outer.data.get(op[1], 0) + 1
                            out.append(outer.data[op[1]])
                        else:
                            outer.ttl[op[1]] = op[2]
                            out.append(True)
                    return out

            return P()

    redis = FakeRedis()
    limiter = RedisRateLimiter(redis, clock=lambda: 120.5)
    assert [limiter.hit("k", 2)[0] for _ in range(3)] == [True, True, False]
    assert list(redis.ttl.values()) == [120]  # keys clean themselves up


def test_the_daily_budget_stops_the_assistant_and_records_spend(db_schema) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(db_schema)
    rig.w.conn.execute("UPDATE sites SET daily_budget_usd = 0.01 WHERE site_key = %s", (rig.key,))
    site = rig.w.conn.execute(
        "SELECT id, org_id FROM sites WHERE site_key = %s", (rig.key,)
    ).fetchone()
    scope = Scope(site[1], None, site[0])
    assert rig.chat().status_code == 200
    record_usage(rig.w.conn, scope, 0.02)  # a busy day
    assert spent_today(rig.w.conn, scope) >= 0.02
    stopped = rig.chat()
    assert stopped.status_code == 429 and "unavailable" in stopped.json()["detail"]


# ---- leads, handoff, feedback ------------------------------------------------------------------
LEAD = {
    "name": "Jo Visitor",
    "email": "Jo@Example.com",
    "phone": "+1 415 555 0132",
    "message": "Pricing?",
    "consent": True,
}


def test_a_lead_needs_consent_and_a_valid_email(rig: Rig) -> None:
    session = rig.done(rig.chat())["session_id"]
    ok = rig.post("lead", {**LEAD, "session_id": session})
    assert ok.status_code == 200
    assert rig.w.conn.execute("SELECT email, consent FROM leads").fetchone() == (
        "jo@example.com",
        True,
    )
    for bad, why in [
        ({"consent": False}, "consent"),
        ({"email": "not-an-email"}, "email"),
        ({"name": "  "}, "name"),
    ]:
        r = rig.post("lead", {**LEAD, "session_id": session, **bad})
        assert r.status_code == 422 and why in r.json()["detail"], why
    assert (
        rig.post("lead", {**LEAD, "session_id": str(uuid.uuid4())}).status_code == 422
    )  # unknown session
    assert rig.w.conn.execute("SELECT count(*) FROM leads").fetchone()[0] == 1


def test_a_visitor_cannot_attach_a_lead_to_another_sites_session(rig: Rig) -> None:
    session = rig.done(rig.chat())["session_id"]
    r = rig.post(
        "lead", {**LEAD, "session_id": session}, key=rig.other_key, origin="https://shop.acme.com"
    )
    assert r.status_code == 422


def test_a_handoff_flags_the_session_and_staff_can_follow_it_up(rig: Rig) -> None:
    session = rig.done(rig.chat("I need a person"))["session_id"]
    assert (
        rig.post("handoff", {"session_id": session, "reason": "billing problem"}).status_code == 200
    )
    rig.post("lead", {**LEAD, "session_id": session})
    listed = rig.client.get("/admin/handoffs", headers=ADMIN).json()
    assert [(h["handoff_reason"], h["email"], h["messages"]) for h in listed] == [
        ("billing problem", "jo@example.com", 2)
    ]
    transcript = rig.client.get(f"/admin/handoffs/{session}", headers=ADMIN).json()
    assert [m["role"] for m in transcript] == ["user", "assistant"]
    assert rig.client.get("/admin/leads", headers=ADMIN).json()[0]["name"] == "Jo Visitor"
    assert rig.client.post(f"/admin/handoffs/{session}/resolve", headers=ADMIN).status_code == 200
    assert (
        rig.client.get("/admin/handoffs", headers=ADMIN).json()[0]["handoff_resolved_at"]
        is not None
    )


def test_staff_endpoints_are_for_admins_only(rig: Rig) -> None:
    member = {"X-Dev-User": "amy@x.com"}
    for path in ("/admin/leads", "/admin/handoffs"):
        assert rig.client.get(path, headers=member).status_code == 403
        assert rig.client.get(path).status_code == 401  # and a visitor has no way in at all
    assert (
        rig.client.post(f"/admin/handoffs/{uuid.uuid4()}/resolve", headers=ADMIN).status_code == 404
    )


def test_handoff_for_an_unknown_session_is_refused(rig: Rig) -> None:
    assert rig.post("handoff", {"session_id": str(uuid.uuid4())}).status_code == 422


def test_thumbs_are_saved_only_for_the_visitors_own_answers(rig: Rig) -> None:
    done = rig.done(rig.chat())
    body = {"session_id": done["session_id"], "message_id": done["message_id"]}
    assert rig.post("feedback", {**body, "rating": 1, "comment": "helpful"}).status_code == 200
    assert rig.w.conn.execute("SELECT rating, comment FROM feedback").fetchone() == (1, "helpful")
    assert rig.post("feedback", {**body, "rating": 5}).status_code == 422
    other = rig.done(rig.chat("another visitor"))  # a different session
    assert (
        rig.post(
            "feedback",
            {"session_id": other["session_id"], "message_id": done["message_id"], "rating": 1},
        ).status_code
        == 422
    )
    user_msg = rig.w.conn.execute(
        "SELECT id FROM widget_messages WHERE role = 'user' LIMIT 1"
    ).fetchone()[0]
    assert (
        rig.post("feedback", {**body, "message_id": str(user_msg), "rating": 1}).status_code == 422
    )


def test_script_tags_in_a_lead_are_stored_as_plain_text_and_never_echoed_as_html(rig: Rig) -> None:
    session = rig.done(rig.chat())["session_id"]
    r = rig.post("lead", {**LEAD, "session_id": session, "name": "<script>alert(1)</script>"})
    assert r.status_code == 200 and "<script>" not in r.text
    assert "application/json" in r.headers["content-type"]


# ---- row-level security for the widget tables (restricted role) --------------------------------
def test_sites_cannot_see_each_others_widget_rows_but_members_can_read_them(world) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
    c = world.conn
    org = world.ann.org_id
    s1 = c.execute(
        "INSERT INTO sites (org_id, name, site_key) VALUES (%s, 'one', 'pk_1') RETURNING id", (org,)
    ).fetchone()[0]
    s2 = c.execute(
        "INSERT INTO sites (org_id, name, site_key) VALUES (%s, 'two', 'pk_2') RETURNING id", (org,)
    ).fetchone()[0]
    sess1 = create_session(c, Scope(org, None, s1))
    add_lead(c, Scope(org, None, s1), sess1, name="Jo", email="jo@example.com", consent=True)
    conn = app_connection(world)
    count = lambda t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # noqa: E731
    set_session_tenant(conn, Scope(org, None, s1))
    assert (count("widget_sessions"), count("leads")) == (1, 1)
    set_session_tenant(conn, Scope(org, None, s2))  # same org, other site
    assert (count("widget_sessions"), count("leads")) == (0, 0)
    set_session_tenant(conn, Scope(world.bob.org_id, None, s1))  # right site id, wrong org
    assert (count("widget_sessions"), count("leads")) == (0, 0)
    set_session_tenant(conn, Scope(org, None, None))  # no site: nothing
    assert (count("widget_sessions"), count("leads")) == (0, 0)
    set_session_tenant(conn, world.amy)  # a member of the org
    assert (count("widget_sessions"), count("leads")) == (1, 1)
    with pytest.raises(
        psycopg.errors.InsufficientPrivilege
    ):  # members read; only visitors/sites write
        conn.execute("INSERT INTO widget_sessions (org_id, site_id) VALUES (%s, %s)", (org, s1))


# ---- the JavaScript's pure logic (Node) ----------------------------------------------------------
@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_the_widget_stream_parser_passes_its_node_tests() -> None:
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        ["node", "--test", "widget/test/parse.test.js"], cwd=root, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout[-800:]
