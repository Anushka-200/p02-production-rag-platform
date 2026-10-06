"""Channel tests (task 4.9): the website widget and WhatsApp side by side, two companies, one database.

What must hold across channels: a visitor or a phone of one company never sees another company's
text, staff see only their own company's leads and handoffs (from BOTH channels), budgets and rate
limits are per site, and a phone that switches company leaves nothing behind.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.channels import whatsapp
from app.channels.whatsapp import Deps, handle_message
from app.config import Settings
from app.llm import LLMResult
from app.ratelimit import MemoryRateLimiter
from scripts.create_site import create_site
from scripts.create_whatsapp_channel import create_channel
from tests.test_api import FakeEmbedder, FakeStream, SharedConnection
from tests.test_search import World, vec

ACME_PUBLIC, ACME_ORG = "ACME-PUBLIC we open at 8:30", "ACME-INTERNAL salaries"
GLOBEX_PUBLIC, GLOBEX_ORG = "GLOBEX-PUBLIC we open at 9:00", "GLOBEX-INTERNAL salaries"
SANDBOX = "whatsapp:+14155238886"
JSON_ANSWER = json.dumps({"supported": True, "answer": "We open soon [S1].", "citations": ["S1"]})
ORIGIN = {"acme": "https://www.acme.com", "globex": "https://www.globex.com"}


class Companies:
    def __init__(self, db_schema) -> None:  # type: ignore[no-untyped-def]
        self.conn = db_schema
        w = World(db_schema)
        self.staff = {}
        self.keys: dict[str, str] = {}
        for name, public, internal in (
            ("acme", ACME_PUBLIC, ACME_ORG),
            ("globex", GLOBEX_PUBLIC, GLOBEX_ORG),
        ):
            admin = w.person(name, "boss")
            db_schema.execute("UPDATE users SET role = 'admin' WHERE id = %s", (admin.user_id,))
            w.chunk(admin, name, public, vec(30), visibility="public", uri=f"{name}-pub.md")
            w.chunk(admin, name, internal, vec(30), visibility="org", uri=f"{name}-int.md")
            self.keys[name] = create_site(db_schema, name, f"{name} website", [ORIGIN[name]])
            create_channel(db_schema, self.keys[name], name, None)
            self.staff[name] = {
                "X-Dev-User": "boss@x.com"
            }  # the same email exists in both: see below
        self.prompts: list = []  # type: ignore[type-arg]
        self.limiter = MemoryRateLimiter()
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None, environment="dev", jwt_secret="c" * 40, reranker_provider="none",
            retrieval_candidates=10,
        )  # fmt: skip

        def stream_fn(messages, **kw):  # type: ignore[no-untyped-def]
            self.prompts.append(messages)
            return FakeStream(["We open soon [S1]."])

        def complete_fn(messages, **kw):  # type: ignore[no-untyped-def]
            self.prompts.append(messages)
            return LLMResult(JSON_ANSWER, "fake", 100, 20, 0.002, 0.3, 1)

        self.client = TestClient(
            create_app(
                settings,
                connect=lambda: SharedConnection(db_schema),
                embedder=FakeEmbedder(),
                reranker=None,
                stream_fn=stream_fn,
                complete_fn=complete_fn,
                limiter=self.limiter,
            )  # fmt: skip
        )
        self.deps = Deps(
            settings=settings, open_connection=lambda: SharedConnection(db_schema),
            state={"embedder": FakeEmbedder(), "reranker": None}, limiter=self.limiter,
            complete_fn=complete_fn,
        )  # fmt: skip
        self.n = 0

    def web(self, company: str, question: str = "When do you open?", session_id=None):  # type: ignore[no-untyped-def]
        body = {"question": question, **({"session_id": session_id} if session_id else {})}
        r = self.client.post(
            f"/widget/{self.keys[company]}/chat", json=body, headers={"Origin": ORIGIN[company]}
        )
        assert r.status_code == 200, r.text
        done = next(
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data: ") and '"session_id"' in line
        )
        return done

    def phone(self, number: str, text: str) -> str:
        self.n += 1
        return handle_message(self.deps, number, SANDBOX, text, f"SM{self.n}")

    def context_sent(self) -> str:
        parts = [m["content"].split("\n\nQuestion:")[0] for call in self.prompts for m in call]
        return "\n".join(parts)


@pytest.fixture
def co(db_schema) -> Companies:  # type: ignore[no-untyped-def]
    return Companies(db_schema)


def test_no_channel_ever_shows_another_companys_text_or_any_internal_text(co: Companies) -> None:
    co.web("acme")
    co.phone("whatsapp:+4915100000001", "START acme")
    co.phone("whatsapp:+4915100000001", "When do you open?")
    sent = co.context_sent()
    assert ACME_PUBLIC in sent
    assert not any(t in sent for t in (GLOBEX_PUBLIC, GLOBEX_ORG, ACME_ORG))
    co.prompts.clear()
    co.web("globex")
    co.phone("whatsapp:+4915100000002", "START globex")
    co.phone("whatsapp:+4915100000002", "When do you open?")
    sent = co.context_sent()
    assert GLOBEX_PUBLIC in sent and not any(t in sent for t in (ACME_PUBLIC, ACME_ORG, GLOBEX_ORG))


def test_one_companys_staff_see_only_their_own_handoffs_and_leads_from_both_channels(
    co: Companies,
) -> None:
    web = co.web("acme")["session_id"]
    co.client.post(
        f"/widget/{co.keys['acme']}/handoff",
        json={"session_id": web},
        headers={"Origin": ORIGIN["acme"]},
    )
    co.phone("whatsapp:+4915100000001", "START acme")
    co.phone("whatsapp:+4915100000001", "When do you open?")
    co.phone("whatsapp:+4915100000001", "human")
    co.phone("whatsapp:+4915100000001", "yes Jo")
    other = co.web("globex")["session_id"]
    co.client.post(
        f"/widget/{co.keys['globex']}/handoff",
        json={"session_id": other},
        headers={"Origin": ORIGIN["globex"]},
    )
    # both companies have a "boss@x.com"; use each org's own admin through a scope-correct dev header
    rows = co.conn.execute(
        "SELECT u.email, o.name FROM users u JOIN orgs o ON o.id = u.org_id"
    ).fetchall()
    assert len(rows) == 2
    co.conn.execute(
        "UPDATE users SET email = 'boss@acme.test' WHERE org_id = (SELECT id FROM orgs WHERE name = 'acme')"
    )
    co.conn.execute(
        "UPDATE users SET email = 'boss@globex.test' WHERE org_id = (SELECT id FROM orgs WHERE name = 'globex')"
    )
    acme = co.client.get("/admin/handoffs", headers={"X-Dev-User": "boss@acme.test"}).json()
    globex = co.client.get("/admin/handoffs", headers={"X-Dev-User": "boss@globex.test"}).json()
    assert sorted(h["channel"] for h in acme) == ["web", "whatsapp"] and [
        h["channel"] for h in globex
    ] == ["web"]
    assert [
        lead["phone"]
        for lead in co.client.get("/admin/leads", headers={"X-Dev-User": "boss@acme.test"}).json()
    ] == ["+4915100000001"]
    assert co.client.get("/admin/leads", headers={"X-Dev-User": "boss@globex.test"}).json() == []


def test_a_budget_or_rate_limit_hit_by_one_site_does_not_affect_the_other(co: Companies) -> None:
    co.web("acme")  # spends a little
    co.conn.execute(
        "UPDATE sites SET daily_budget_usd = 0.01 WHERE name = 'acme website'"
    )  # one cent
    co.conn.execute("UPDATE site_usage SET cost_usd = 0.01")  # ...and it has been spent
    r = co.client.post(
        f"/widget/{co.keys['acme']}/chat",
        json={"question": "again"},
        headers={"Origin": ORIGIN["acme"]},
    )
    assert r.status_code == 429
    co.phone("whatsapp:+4915100000001", "START acme")
    assert co.phone("whatsapp:+4915100000001", "hello?") == whatsapp.BUSY
    assert co.web("globex")["text"]  # globex carries on


def test_web_and_whatsapp_spend_adds_up_in_the_same_sites_total(co: Companies) -> None:
    co.web("acme")
    co.phone("whatsapp:+4915100000001", "START acme")
    co.phone("whatsapp:+4915100000001", "When do you open?")
    requests, cost = co.conn.execute("SELECT requests, cost_usd FROM site_usage").fetchone()
    assert requests == 2 and float(cost) > 0


def test_a_phone_that_switches_company_leaves_nothing_behind_in_the_old_one(co: Companies) -> None:
    phone = "whatsapp:+4915100000001"
    co.phone(phone, "START acme")
    co.phone(phone, "When do you open?")
    co.phone(phone, "human")
    co.phone(phone, "yes Jo")
    assert co.conn.execute("SELECT count(*) FROM leads").fetchone()[0] == 1
    co.phone(phone, "START globex")
    assert (
        co.conn.execute(
            "SELECT count(*) FROM widget_sessions WHERE channel = 'whatsapp'"
        ).fetchone()[0]
        == 0
    )
    assert (
        co.conn.execute("SELECT count(*) FROM leads").fetchone()[0] == 0
    )  # acme's data went with it
    assert co.conn.execute("SELECT count(*) FROM widget_messages").fetchone()[0] == 0
    co.phone(phone, "When do you open?")
    assert co.conn.execute(
        "SELECT o.name FROM widget_sessions s JOIN orgs o ON o.id = s.org_id"
    ).fetchall() == [("globex",)]


def test_a_session_id_from_one_company_does_nothing_on_another_companys_widget(
    co: Companies,
) -> None:
    acme_session = co.web("acme")["session_id"]
    on_globex = co.web("globex", session_id=acme_session)["session_id"]
    assert on_globex != acme_session
    lead = co.client.post(
        f"/widget/{co.keys['globex']}/lead", headers={"Origin": ORIGIN["globex"]},
        json={"session_id": acme_session, "name": "Eve", "email": "eve@example.com", "consent": True},
    )  # fmt: skip
    assert lead.status_code == 422
