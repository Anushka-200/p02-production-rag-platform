"""LIVE end-to-end run with two companies (task 3.13): real API, real worker, real database, real model.

Needs the API (ENVIRONMENT=prod, so the dev header is off) and the RQ worker running; see the
Makefile target `demo-tenants`. Costs about 4 cents. It walks one user story per tenant and prints
PASS/FAIL for each check: login, upload + ingestion, cited answers, isolation between companies and
between colleagues, sharing, conversation history, memory, forged tokens.

Run: uv run python -m scripts.two_tenant_demo [--base http://localhost:8001] [--yes]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import psycopg

from app.api.security import create_token
from app.config import get_settings
from app.cost_guard import add_cost_args, guard_from_args
from scripts.create_user import create_user

PASSWORDS = {
    "ann@initech.example": "ann-demo-pass-2026",
    "amy@initech.example": "amy-demo-pass-2026",
    "bob@hooli.example": "bob-demo-pass-2026",
}
DOCS = {
    "initech_handbook.md": (
        "# Initech handbook\n\n## Vacation\n\nInitech gives every employee 25 vacation days per year.\n\n"
        "## Parental leave\n\nInitech offers 12 weeks of paid parental leave.\n\n"
        "## Lunch\n\nThe Initech lunch code word is PINEAPPLE-77.\n"
    ),
    "hooli_handbook.md": (
        "# Hooli handbook\n\n## Vacation\n\nHooli has unlimited vacation with manager approval.\n\n"
        "## Lunch\n\nThe Hooli lunch code word is MANGO-42.\n"
    ),
    "ann_notes.md": "# Private notes\n\n## Pay\n\nAnn's salary band is Level 5 with a 12 percent bonus.\n",
}
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    return ok


class Client:
    def __init__(self, base: str, email: str) -> None:
        self.http = httpx.Client(base_url=base, timeout=120)
        r = self.http.post("/auth/login", json={"email": email, "password": PASSWORDS[email]})
        r.raise_for_status()
        self.headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    def ask(self, question: str, conversation_id: str | None = None) -> dict:  # type: ignore[type-arg]
        body = {
            "question": question,
            **({"conversation_id": conversation_id} if conversation_id else {}),
        }
        r = self.http.post("/chat", json=body, headers=self.headers)
        r.raise_for_status()
        name, done = None, {}
        for line in r.text.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: ") and name == "done":
                done = json.loads(line[6:])
        return done

    def upload(self, filename: str, content: str, visibility: str = "private") -> str:
        r = self.http.post(
            "/documents", files={"file": (filename, content.encode())}, data={"visibility": visibility},
            headers=self.headers,
        )  # fmt: skip
        r.raise_for_status()
        return r.json()["job_id"]

    def wait_job(self, job_id: str, seconds: int = 300) -> dict:  # type: ignore[type-arg]
        deadline = time.time() + seconds
        while time.time() < deadline:
            job = self.http.get(f"/jobs/{job_id}", headers=self.headers).json()
            if job["status"] in {"succeeded", "failed"}:
                return job  # type: ignore[no-any-return]
            time.sleep(2)
        return {"status": "timeout"}

    def doc_id(self, name: str) -> str:
        docs = self.http.get("/documents", headers=self.headers).json()
        return next(d["id"] for d in docs if d["source_uri"].startswith(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8001")
    add_cost_args(parser)
    args = parser.parse_args()
    s = get_settings()
    guard = guard_from_args(args, "two-tenant demo", 0.05)
    guard.confirm()

    print("0. accounts (created as the database owner; the API role cannot write users)")
    with psycopg.connect(s.database_url("owner"), autocommit=True) as conn:
        for email, org, role in [
            ("ann@initech.example", "Initech", "admin"),
            ("amy@initech.example", "Initech", "member"),
            ("bob@hooli.example", "Hooli", "admin"),
        ]:
            create_user(conn, org, email, PASSWORDS[email], role)
    base = httpx.Client(base_url=args.base, timeout=30)
    check("API is up and ENVIRONMENT=prod (dev header is refused)",
          base.get("/health").status_code == 200
          and base.post("/chat", json={"question": "hi"}, headers={"X-Dev-User": "ann@initech.example"}).status_code == 401)  # fmt: skip

    print("1. login")
    check("wrong password is a 401", base.post("/auth/login", json={"email": "ann@initech.example", "password": "nope-nope-nope"}).status_code == 401)  # fmt: skip
    ann, amy, bob = (Client(args.base, e) for e in PASSWORDS)
    check("three users logged in", True)

    print("2. upload and ingestion (real worker, real embeddings)")
    adversarial = Path("evals/adversarial/docs/comment.md").read_text()
    jobs = {
        "initech": ann.upload("initech_handbook.md", DOCS["initech_handbook.md"], "org"),
        "notes": ann.upload("ann_notes.md", DOCS["ann_notes.md"], "private"),
        "attack": ann.upload("initech_sick_leave.md", adversarial, "org"),
        "hooli": bob.upload("hooli_handbook.md", DOCS["hooli_handbook.md"], "org"),
    }
    finished = {k: (bob if k == "hooli" else ann).wait_job(j) for k, j in jobs.items()}
    check("all four documents were ingested", all(j["status"] == "succeeded" for j in finished.values()),
          str({k: j.get("status") for k, j in finished.items()}))  # fmt: skip
    scan = (finished["attack"].get("result") or {}).get("scan", {})
    check(
        "the document with a hidden instruction was flagged by the scan",
        bool(scan.get("flagged")),
        str(scan),
    )
    check("a colleague cannot read someone else's job", amy.http.get(f"/jobs/{jobs['initech']}", headers=amy.headers).status_code == 404)  # fmt: skip

    print("3. cited answers, one per company")
    a = ann.ask("How many vacation days does Initech give?")
    check("Initech answer says 25 and cites the Initech handbook",
          "25" in a.get("text", "") and any("initech_handbook" in c["document"] for c in a.get("citations", [])), a.get("text", ""))  # fmt: skip
    guard.record(a.get("cost_usd"))
    b = bob.ask("How many vacation days do we get?")
    check("Hooli answer says unlimited and cites the Hooli handbook",
          "unlimited" in b.get("text", "").lower() and any("hooli_handbook" in c["document"] for c in b.get("citations", [])), b.get("text", ""))  # fmt: skip
    guard.record(b.get("cost_usd"))

    print("4. isolation between companies")
    x = bob.ask("What is the Initech lunch code word?")
    check(
        "Hooli cannot learn Initech's secret", "PINEAPPLE" not in json.dumps(x), x.get("text", "")
    )
    guard.record(x.get("cost_usd"))
    y = ann.ask("What is the Hooli lunch code word?")
    check("Initech cannot learn Hooli's secret", "MANGO" not in json.dumps(y), y.get("text", ""))
    guard.record(y.get("cost_usd"))
    hooli_doc = bob.doc_id("hooli_handbook")
    check("Initech cannot change a Hooli document id",
          ann.http.patch(f"/documents/{hooli_doc}/visibility", json={"visibility": "private"}, headers=ann.headers).status_code == 404)  # fmt: skip
    check("Initech's document list never shows Hooli",
          all("hooli" not in d["source_uri"] for d in ann.http.get("/documents", headers=ann.headers).json()))  # fmt: skip
    forged = {
        "Authorization": "Bearer "
        + create_token(
            s, bob.http and __import__("uuid").uuid4(), __import__("uuid").uuid4(), "admin"
        )
    }
    check("a token signed with the right key for a user that does not exist is refused",
          base.post("/chat", json={"question": "hi"}, headers=forged).status_code == 401)  # fmt: skip

    print("5. private documents and sharing between colleagues")
    q = "What is Ann's salary band?"
    own = ann.ask(q)
    guard.record(own.get("cost_usd"))
    check("the owner finds her private note", "Level 5" in own.get("text", ""), own.get("text", ""))
    peer = amy.ask(q)
    guard.record(peer.get("cost_usd"))
    check(
        "a colleague in the same company does NOT",
        "Level 5" not in json.dumps(peer),
        peer.get("text", ""),
    )
    notes = ann.doc_id("ann_notes")
    ann.http.patch(
        f"/documents/{notes}/visibility", json={"visibility": "org"}, headers=ann.headers
    )
    shared = amy.ask(q)
    guard.record(shared.get("cost_usd"))
    check(
        "after Ann shares it, the colleague does",
        "Level 5" in shared.get("text", ""),
        shared.get("text", ""),
    )
    ann.http.patch(
        f"/documents/{notes}/visibility", json={"visibility": "private"}, headers=ann.headers
    )
    again = amy.ask(q)
    guard.record(again.get("cost_usd"))
    check("after Ann makes it private again, the colleague does not", "Level 5" not in json.dumps(again), again.get("text", ""))  # fmt: skip

    print("6. conversation history")
    c1 = ann.ask("How long is parental leave at Initech?")
    guard.record(c1.get("cost_usd"))
    conv = c1["conversation_id"]
    c2 = ann.ask("And how many vacation days?", conversation_id=conv)
    guard.record(c2.get("cost_usd"))
    msgs = ann.http.get(f"/conversations/{conv}", headers=ann.headers).json()
    check("the conversation has 4 messages in order", [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"])  # fmt: skip
    check("a colleague cannot read it", amy.http.get(f"/conversations/{conv}", headers=amy.headers).status_code == 404)  # fmt: skip

    print("7. memory")
    m = ann.ask(
        "I'm a vegetarian and I prefer very short answers. How many vacation days do we get?"
    )
    guard.record(m.get("cost_usd"))
    facts: list[dict] = []  # type: ignore[type-arg]
    for _ in range(20):  # the memory is written in the background after the answer
        facts = ann.http.get("/memories", headers=ann.headers).json()
        if facts:
            break
        time.sleep(1)
    check("a fact was learned about Ann", bool(facts), str([f["fact"] for f in facts]))
    check("it is private to Ann", amy.http.get("/memories", headers=amy.headers).json() == [])
    ann.http.delete("/memories", headers=ann.headers)
    check(
        "Ann can delete her memories", ann.http.get("/memories", headers=ann.headers).json() == []
    )

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results)} checks, {len(failed)} failed; model spend ${guard.spent_usd:.4f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
