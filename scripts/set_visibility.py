"""Change a document's visibility from the command line. Runs as the DB OWNER (a demo helper).

Run: uv run python -m scripts.set_visibility --org Initech --source initech_handbook --visibility public
Publishing makes the WHOLE document readable by anonymous visitors (website chat, WhatsApp).
"""

import argparse

import psycopg

from app.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument("--source", required=True, help="start of the document's source_uri")
    parser.add_argument("--visibility", required=True, choices=["private", "org", "public"])
    args = parser.parse_args()
    with psycopg.connect(get_settings().database_url("owner"), autocommit=True) as conn:
        rows = conn.execute(
            "UPDATE documents SET visibility = %s, updated_at = now() WHERE source_uri LIKE %s "
            "AND org_id = (SELECT id FROM orgs WHERE name = %s) RETURNING source_uri",
            (args.visibility, args.source + "%", args.org),
        ).fetchall()
    print(
        f"{len(rows)} document(s) now {args.visibility}:", [r[0] for r in rows]
    )  # chunks follow by cascade


if __name__ == "__main__":
    main()
