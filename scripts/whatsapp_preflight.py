"""Check everything that can be checked BEFORE you connect Twilio, and print what to paste where.

Run: uv run python -m scripts.whatsapp_preflight
Exit code 0 when nothing FAILED (warnings are advice), 1 otherwise.
"""

import sys

import httpx
import psycopg

from app.config import get_settings

results: list[tuple[str, str, str]] = []


def report(level: str, what: str, detail: str = "") -> None:
    results.append((level, what, detail))
    print(f"  {level:<4}  {what}" + (f"   [{detail}]" if detail else ""))


def main() -> None:
    s = get_settings()
    print("configuration (.env)")
    token = s.twilio_auth_token.get_secret_value() if s.twilio_auth_token else ""
    report(
        "PASS" if len(token) >= 20 else "FAIL",
        "TWILIO_AUTH_TOKEN is set (Twilio console > Account info)",
    )
    report(
        "PASS" if s.jwt_secret else "FAIL", "JWT_SECRET is set (also keys the phone-number hash)"
    )
    base = (s.public_base_url or "").rstrip("/")
    report(
        "PASS" if base.startswith("https://") else "FAIL",
        "PUBLIC_BASE_URL is the https tunnel address",
        base or "not set",
    )
    report("PASS" if s.twilio_verify_signature else "FAIL", "signature checking is ON")
    report(
        "PASS" if s.webhook_only else "WARN",
        "WEBHOOK_ONLY=true while tunnelled (only the webhook is reachable)",
    )

    print("database")
    try:
        with psycopg.connect(s.database_url("owner"), autocommit=True) as conn:
            rows = conn.execute(
                "SELECT c.keyword, si.name, o.name, (c.enabled AND si.enabled), si.daily_budget_usd, "
                "(SELECT count(*) FROM documents d WHERE d.org_id = c.org_id AND d.visibility = 'public' "
                " AND d.status = 'ready') FROM whatsapp_channels c JOIN sites si ON si.id = c.site_id "
                "JOIN orgs o ON o.id = c.org_id ORDER BY c.keyword"
            ).fetchall()
    except Exception as exc:
        report("FAIL", "database reachable", type(exc).__name__)
        rows = []
    if not rows:
        report("FAIL", "a WhatsApp channel exists", "run scripts/create_whatsapp_channel.py")
    for keyword, site, org, enabled, budget, public_docs in rows:
        report(
            "PASS" if enabled else "FAIL",
            f"channel '{keyword}' ({org} / {site}) is enabled",
            f"daily budget ${budget}",
        )
        report(
            "PASS" if public_docs else "WARN",
            f"'{org}' has public documents to answer from",
            f"{public_docs} ready",
        )

    print("public address")
    if base:
        try:
            health = httpx.get(f"{base}/health", timeout=10)
            report(
                "PASS" if health.status_code == 200 else "FAIL",
                "GET /health through the tunnel",
                str(health.status_code),
            )
            probe = httpx.post(
                f"{base}/auth/login", json={"email": "a@b.co", "password": "x"}, timeout=10
            )
            report(
                "PASS" if probe.status_code == 404 else "WARN",
                "the rest of the API is NOT reachable through the tunnel",
                str(probe.status_code),
            )
        except Exception as exc:
            report("FAIL", "the tunnel answers", type(exc).__name__)
    else:
        report("WARN", "tunnel not checked", "set PUBLIC_BASE_URL first")

    print("\nTwilio console > Messaging > Try it out > Send a WhatsApp message > Sandbox settings")
    print(
        f"  WHEN A MESSAGE COMES IN : {base or 'https://<tunnel>'}/channels/whatsapp/webhook   (method POST)"
    )
    print(
        "  then, from YOUR phone  : send 'join <your-two-word-sandbox-code>' to the sandbox number,"
    )
    for keyword, _, org, *_ in rows:
        print(f"  then send              : START {keyword}     ({org})")
    failed = [r for r in results if r[0] == "FAIL"]
    print(
        f"\n{len(results)} checks, {len(failed)} failed, {sum(r[0] == 'WARN' for r in results)} warnings"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
