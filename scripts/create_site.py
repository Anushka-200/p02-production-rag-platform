"""Register a website that may embed the chat widget. Runs as the DB OWNER.

The site key printed here goes into the page's HTML; it is public by design. What protects the org
is the origin allow-list (only these websites may use the key), the enabled switch, and the fact
that anonymous visitors can only ever read documents marked 'public'.

Run: uv run python -m scripts.create_site --org Initech --name "Initech website" \
        --origin https://www.initech.example [--origin http://localhost:8080]
"""

import argparse
import secrets

import psycopg

from app.config import get_settings


def create_site(conn: psycopg.Connection, org: str, name: str, origins: list[str]) -> str:
    org_row = conn.execute("SELECT id FROM orgs WHERE name = %s", (org,)).fetchone()
    if org_row is None:
        raise SystemExit(f"no organisation named {org!r}; create a user in it first")
    key = "pk_" + secrets.token_urlsafe(16)
    conn.execute(
        "INSERT INTO sites (org_id, name, site_key, allowed_origins) VALUES (%s, %s, %s, %s)",
        (org_row[0], name, key, [o.strip().lower().rstrip("/") for o in origins]),
    )
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--origin", action="append", required=True, help="repeatable")
    args = parser.parse_args()
    with psycopg.connect(get_settings().database_url("owner"), autocommit=True) as conn:
        print("site key:", create_site(conn, args.org, args.name, args.origin))


if __name__ == "__main__":
    main()
