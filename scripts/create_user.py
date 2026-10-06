"""Create (or update) an organisation and a user with a password. Runs as the DB OWNER.

The API's database role is deliberately not allowed to write `users`, so accounts are created here
(later: an admin invite flow). The password is hashed with scrypt; it is never stored or logged.

Run: uv run python -m scripts.create_user --org "Initech" --email ann@initech.example [--role admin]
     (the password is asked for on the keyboard, or read from env USER_PASSWORD)
"""

import argparse
import getpass
import os

import psycopg

from app.api.security import hash_password
from app.config import get_settings

MIN_PASSWORD_CHARS = 10


def create_user(conn: psycopg.Connection, org: str, email: str, password: str, role: str) -> str:
    if len(password) < MIN_PASSWORD_CHARS:
        raise SystemExit(f"password must have at least {MIN_PASSWORD_CHARS} characters")
    org_id = conn.execute(
        "INSERT INTO orgs (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
        "RETURNING id",
        (org,),
    ).fetchone()[0]  # type: ignore[index]
    user_id = conn.execute(
        "INSERT INTO users (org_id, email, role, password_hash) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (org_id, email) DO UPDATE SET role = EXCLUDED.role, "
        "password_hash = EXCLUDED.password_hash RETURNING id",
        (org_id, email.strip().lower(), role, hash_password(password)),
    ).fetchone()[0]  # type: ignore[index]
    return str(user_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", choices=["admin", "member"], default="member")
    args = parser.parse_args()
    password = os.environ.get("USER_PASSWORD") or getpass.getpass("password: ")
    with psycopg.connect(get_settings().database_url("owner"), autocommit=True) as conn:
        print("user id:", create_user(conn, args.org, args.email, password, args.role))


if __name__ == "__main__":
    main()
