"""Connect a website's assistant (a "site") to WhatsApp. Runs as the DB OWNER.

Twilio's sandbox has ONE shared number, so each company gets a join KEYWORD: a user sends
"START acme" once and from then on that phone talks to Acme's assistant. A company with its own
WhatsApp number can also pass --to-number so no keyword is needed.

Run: uv run python -m scripts.create_whatsapp_channel --site-key pk_... --keyword acme
"""

import argparse

import psycopg

from app.channels.twilio import normalise_number
from app.config import get_settings


def create_channel(
    conn: psycopg.Connection, site_key: str, keyword: str, to_number: str | None = None
) -> str:
    site = conn.execute("SELECT id, org_id FROM sites WHERE site_key = %s", (site_key,)).fetchone()
    if site is None:
        raise SystemExit("no site with that key; create one with scripts/create_site.py")
    row = conn.execute(
        "INSERT INTO whatsapp_channels (org_id, site_id, keyword, to_number) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (keyword) DO UPDATE SET site_id = EXCLUDED.site_id, org_id = EXCLUDED.org_id, "
        "to_number = EXCLUDED.to_number, enabled = true RETURNING id",
        (
            site[1],
            site[0],
            keyword.strip().lower(),
            normalise_number(to_number) if to_number else None,
        ),
    ).fetchone()
    return str(row[0])  # type: ignore[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-key", required=True)
    parser.add_argument("--keyword", required=True)
    parser.add_argument(
        "--to-number", help="a dedicated WhatsApp number, e.g. whatsapp:+14155550100"
    )
    args = parser.parse_args()
    with psycopg.connect(get_settings().database_url("owner"), autocommit=True) as conn:
        print("channel id:", create_channel(conn, args.site_key, args.keyword, args.to_number))


if __name__ == "__main__":
    main()
