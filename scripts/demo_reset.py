"""Put the demo accounts back to a clean state so the upload step can be recorded again.

Through the real API (as the demo users): deletes the sample policy document, all conversations and all memories
of demo-a and demo-b. It touches nothing else (the handbooks stay).

Run: make demo-reset   (the API must be running on :8001)
"""

import httpx

PASSWORD = "demo-password-123"
SAMPLE = "acme-benefits-policy.docx"


def main() -> None:
    with httpx.Client(base_url="http://localhost:8001", timeout=60) as c:
        for email in ("demo-a@demo.example", "demo-b@demo.example"):
            r = c.post("/auth/login", json={"email": email, "password": PASSWORD})
            r.raise_for_status()
            h = {"Authorization": "Bearer " + r.json()["access_token"]}
            n = {"documents": 0, "conversations": 0, "memories": 0}
            for d in c.get("/documents", headers=h).json():
                if d.get("source_uri") == SAMPLE or d.get("filename") == SAMPLE:
                    c.delete(f"/documents/{d['id']}", headers=h).raise_for_status()
                    n["documents"] += 1
            for conv in c.get("/conversations", headers=h).json():
                c.delete(f"/conversations/{conv['id']}", headers=h)
                n["conversations"] += 1
            for m in c.get("/memories", headers=h).json():
                c.delete(f"/memories/{m['id']}", headers=h)
                n["memories"] += 1
            print(email, n)


if __name__ == "__main__":
    main()
