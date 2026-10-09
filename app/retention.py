"""Data retention (task 7.5): delete old personal data from the anonymous channels.

Why: a visitor's chat and a lead's contact details are personal data the company now
holds, and keeping them forever is a liability. This runs on a schedule (cron, or
`python -m scripts.purge_expired`) as the database OWNER; the restricted API role cannot
see across tenants, by design.

Order matters: leads go first (kept longer than chats, because the visitor agreed to be
contacted), then the chats that no longer have a lead; a deleted chat takes its messages
with it. Staff-owned data (documents, a member's conversations and memories) is NOT touched
here: members delete those themselves, and an org deletes its documents.
"""

import psycopg


def purge_expired(conn: psycopg.Connection, *, visitor_days: int, lead_days: int) -> dict[str, int]:
    """Delete expired rows; returns how many were removed per kind. Safe to run repeatedly."""
    counts: dict[str, int] = {}

    def run(name: str, sql: str, *params: int) -> None:
        counts[name] = conn.execute(sql, params).rowcount

    run(
        "leads", "DELETE FROM leads WHERE created_at < now() - make_interval(days => %s)", lead_days
    )
    # A chat that still has a lead is kept until the lead expires; an open handoff is never purged.
    run(
        "visitor_chats",
        "DELETE FROM widget_sessions s WHERE s.last_seen_at < now() - make_interval(days => %s) "
        "AND NOT s.handoff_requested "
        "AND NOT EXISTS (SELECT 1 FROM leads l WHERE l.session_id = s.id)",
        visitor_days,
    )
    run(
        "feedback",
        "DELETE FROM feedback WHERE created_at < now() - make_interval(days => %s)",
        visitor_days,
    )
    run(
        "whatsapp_contacts",
        "DELETE FROM whatsapp_contacts WHERE last_seen_at < now() - make_interval(days => %s)",
        visitor_days,
    )
    run("whatsapp_seen", "DELETE FROM whatsapp_seen WHERE created_at < now() - interval '7 days'")
    return counts
