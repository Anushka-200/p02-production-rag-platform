"""Retention: old visitor data is deleted, young data, leads inside their period and open handoffs are kept."""

from app.retention import purge_expired
from scripts.create_site import create_site
from tests.test_search import World


def session(conn, org, site, days_old, *, handoff=False):  # type: ignore[no-untyped-def]
    sid = conn.execute(
        "INSERT INTO widget_sessions (org_id, site_id, last_seen_at, handoff_requested) "
        "VALUES (%s, %s, now() - make_interval(days => %s), %s) RETURNING id",
        (org, site, days_old, handoff),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO widget_messages (session_id, org_id, site_id, role, content) "
        "VALUES (%s, %s, %s, 'user', 'hello')",
        (sid, org, site),
    )
    return sid


def test_old_visitor_data_goes_but_leads_and_open_handoffs_stay(db_schema) -> None:  # type: ignore[no-untyped-def]
    w = World(db_schema)
    w.person("acme", "ann")
    create_site(db_schema, "acme", "Site", ["https://a.example"])
    org, site = db_schema.execute("SELECT org_id, id FROM sites").fetchone()
    old, young = session(db_schema, org, site, 100), session(db_schema, org, site, 5)
    old_handoff = session(db_schema, org, site, 100, handoff=True)
    old_with_lead = session(db_schema, org, site, 100)
    db_schema.execute(
        "INSERT INTO leads (org_id, site_id, session_id, name, email, consent) "
        "VALUES (%s, %s, %s, 'Pat', 'pat@x.example', true)",
        (org, site, old_with_lead),
    )

    counts = purge_expired(db_schema, visitor_days=90, lead_days=365)

    left = {r[0] for r in db_schema.execute("SELECT id FROM widget_sessions").fetchall()}
    assert left == {young, old_handoff, old_with_lead}
    assert counts["visitor_chats"] == 1
    assert (
        db_schema.execute(
            "SELECT count(*) FROM widget_messages WHERE session_id = %s", (old,)
        ).fetchone()[0]
        == 0
    )
    # once the lead itself expires, the lead goes first and its old chat goes in the same pass
    db_schema.execute("UPDATE leads SET created_at = now() - interval '400 days'")
    again = purge_expired(db_schema, visitor_days=90, lead_days=365)
    assert (again["leads"], again["visitor_chats"]) == (1, 1)
    assert purge_expired(db_schema, visitor_days=90, lead_days=365) == dict.fromkeys(
        ["leads", "visitor_chats", "feedback", "whatsapp_contacts", "whatsapp_seen"], 0
    )


def test_every_channel_tells_the_user_it_is_an_ai_assistant() -> None:
    """EU AI Act Article 50: people must be told they are talking to an AI (web, widget, WhatsApp)."""
    from app.api.widget import DISCLOSURE as widget
    from app.channels.whatsapp import DISCLOSURE as whatsapp
    from ui.common import AI_NOTICE as web

    for text in (web, widget, whatsapp):
        assert "AI assistant" in text
