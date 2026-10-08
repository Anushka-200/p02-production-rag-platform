"""Per-organisation daily budget: alert once, stop at the cap, never touch another tenant, no runaway loop."""

import logging

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.budget import EXHAUSTED, OK, WARNING, record, status, usage_report
from app.config import Settings
from tests.test_api import FakeEmbedder, FakeStream, SharedConnection, chat
from tests.test_search import World, vec
from tests.test_widget import Rig as WidgetRig


def settings(**kw) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(_env_file=None, org_daily_budget_usd=1.0, budget_alert_fraction=0.8, **kw)  # type: ignore[call-arg]


@pytest.fixture
def w(db_schema):  # type: ignore[no-untyped-def]
    return World(db_schema)


# ---- the ledger ---------------------------------------------------------------------------------
def test_spend_accumulates_and_the_levels_follow_the_fractions(w: World) -> None:
    ann, s = w.person("acme", "ann"), settings()
    assert status(w.conn, ann, s).level == OK
    assert record(w.conn, ann, s, cost_usd=0.5, input_tokens=100, output_tokens=10).level == OK
    assert record(w.conn, ann, s, cost_usd=0.31).level == WARNING  # 0.81 >= 0.8 of 1.0
    assert record(w.conn, ann, s, cost_usd=0.2).level == EXHAUSTED  # 1.01 >= 1.0
    got = status(w.conn, ann, s)
    assert got.exhausted and got.spent_usd == pytest.approx(1.01)
    row = w.conn.execute("SELECT requests, input_tokens, output_tokens FROM org_usage").fetchone()
    assert row == (3, 100, 10)


def test_each_alert_is_logged_once_per_day(w: World, caplog: pytest.LogCaptureFixture) -> None:
    ann, s = w.person("acme", "ann"), settings()
    with caplog.at_level(logging.WARNING, logger="app.budget"):
        for _ in range(8):  # 0.85 then keeps climbing past the cap
            record(w.conn, ann, s, cost_usd=0.17)
    alerts = [r.getMessage() for r in caplog.records]
    assert alerts == ["org budget alert", "org budget exhausted"]  # not once per request


def test_an_org_can_have_its_own_limit_and_zero_means_unlimited(w: World) -> None:
    ann, bob = w.person("acme", "ann"), w.person("globex", "bob")
    s = settings()
    w.conn.execute("UPDATE orgs SET daily_budget_usd = 10 WHERE id = %s", (ann.org_id,))
    record(w.conn, ann, s, cost_usd=5)
    record(w.conn, bob, s, cost_usd=5)
    assert status(w.conn, ann, s).level == OK  # acme's own cap is 10
    assert status(w.conn, bob, s).exhausted  # globex uses the default of 1
    unlimited = Settings(_env_file=None, org_daily_budget_usd=0)  # type: ignore[call-arg]
    assert (
        status(w.conn, bob, unlimited).level == OK
        and status(w.conn, bob, unlimited).limit_usd is None
    )


def test_one_tenant_running_out_does_not_affect_another(w: World) -> None:
    ann, bob = w.person("acme", "ann"), w.person("globex", "bob")
    s = settings()
    record(w.conn, ann, s, cost_usd=50)
    assert status(w.conn, ann, s).exhausted and status(w.conn, bob, s).spent_usd == 0


def test_a_new_day_starts_from_zero(w: World) -> None:
    ann, s = w.person("acme", "ann"), settings()
    record(w.conn, ann, s, cost_usd=5)
    w.conn.execute("UPDATE org_usage SET day = current_date - 1")
    assert status(w.conn, ann, s).spent_usd == 0
    record(w.conn, ann, s, cost_usd=0.1)
    assert w.conn.execute("SELECT count(*) FROM org_usage").fetchone()[0] == 2


def test_the_usage_report_lists_recent_days(w: World) -> None:
    ann, s = w.person("acme", "ann"), settings()
    record(w.conn, ann, s, cost_usd=0.9)
    report = usage_report(w.conn, ann, s)
    assert report["today"]["level"] == WARNING and report["days"][0]["requests"] == 1


# ---- through the HTTP API: the runaway-cost test ----------------------------------------------
@pytest.fixture
def api(db_schema):  # type: ignore[no-untyped-def]
    w = World(db_schema)
    ann, bob = w.person("acme", "ann"), w.person("globex", "bob")
    w.conn.execute("UPDATE users SET role = 'admin' WHERE id = %s", (ann.user_id,))
    w.chunk(ann, "acme", "Employees receive twenty vacation days", vec(30), uri="a.md")
    w.chunk(bob, "globex", "Globex gives ten vacation days", vec(30), uri="g.md")
    model_calls: list[object] = []

    def stream_fn(messages, **kw):  # type: ignore[no-untyped-def]
        model_calls.append(messages)
        return FakeStream(["Twenty days [S1]."])  # every fresh answer costs $0.002

    cfg = Settings(  # type: ignore[call-arg]
        _env_file=None, environment="dev", reranker_provider="none", retrieval_candidates=10,
        org_daily_budget_usd=0.005,
    )  # fmt: skip
    app = create_app(
        cfg, connect=lambda: SharedConnection(w.conn), embedder=FakeEmbedder(), reranker=None,
        stream_fn=stream_fn, complete_fn=lambda *a, **k: pytest.fail("no JSON fallback expected"),
    )  # fmt: skip
    return type("Api", (), {"client": TestClient(app), "calls": model_calls, "w": w})()


def test_a_runaway_loop_is_stopped_at_the_cap_and_the_model_stops_being_called(api, caplog) -> None:  # type: ignore[no-untyped-def]
    statuses = [chat(api.client)[0] for _ in range(50)]  # a script hammering /chat
    assert statuses.count(200) == 3 and statuses[3:] == [429] * 47
    assert len(api.calls) == 3  # the 47 blocked requests never reached the model
    spent = float(api.w.conn.execute("SELECT cost_usd FROM org_usage").fetchone()[0])
    assert spent == pytest.approx(0.006)  # cap 0.005 plus at most the one answer in flight


def test_another_org_keeps_working_while_the_first_is_blocked(api) -> None:  # type: ignore[no-untyped-def]
    for _ in range(5):
        chat(api.client)
    assert chat(api.client)[0] == 429
    assert chat(api.client, who="bob@x.com")[0] == 200


def test_the_blocked_request_explains_itself_and_creates_no_conversation(api) -> None:  # type: ignore[no-untyped-def]
    for _ in range(3):
        chat(api.client)
    before = api.w.conn.execute("SELECT count(*) FROM conversations").fetchone()[0]
    r = api.client.post("/chat", json={"question": "again?"}, headers={"X-Dev-User": "ann@x.com"})
    assert r.status_code == 429 and "budget" in r.json()["detail"]
    assert api.w.conn.execute("SELECT count(*) FROM conversations").fetchone()[0] == before


def test_an_admin_sees_todays_spend_and_the_alert_level(api) -> None:  # type: ignore[no-untyped-def]
    for _ in range(3):
        chat(api.client)
    r = api.client.get("/admin/usage", headers={"X-Dev-User": "ann@x.com"})
    today = r.json()["today"]
    assert r.status_code == 200 and today["level"] == EXHAUSTED and today["limit_usd"] == 0.005


# ---- the other channels share the same cap ------------------------------------------------------
def test_the_widget_is_stopped_by_the_org_budget_too(db_schema) -> None:  # type: ignore[no-untyped-def]
    rig = WidgetRig(db_schema, org_daily_budget_usd=0.005)
    codes = [rig.chat().status_code for _ in range(6)]
    assert codes.count(200) == 3 and codes[3:] == [429, 429, 429]
