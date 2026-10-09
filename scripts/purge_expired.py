"""Delete expired visitor data (see app/retention.py). Runs as the DB OWNER.

Run: uv run python -m scripts.purge_expired [--visitor-days 90] [--lead-days 365]
"""

import argparse

import psycopg

from app.config import get_settings
from app.retention import purge_expired


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--visitor-days", type=int, default=s.retention_visitor_days)
    parser.add_argument("--lead-days", type=int, default=s.retention_lead_days)
    args = parser.parse_args()
    with psycopg.connect(s.database_url("owner"), autocommit=True) as conn:
        for kind, n in purge_expired(
            conn, visitor_days=args.visitor_days, lead_days=args.lead_days
        ).items():
            print(f"{kind:20} {n} deleted")


if __name__ == "__main__":
    main()
