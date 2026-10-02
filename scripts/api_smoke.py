"""Start the REAL API (real models, real database, real LLM), stream one chat, call the inspector.

Costs about half a cent. Run: uv run python -m scripts.api_smoke
"""

import json
import subprocess
import sys
import time

import httpx

PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"
HEADERS = {"X-Dev-User": "admin@tenant-a.example"}
QUESTION = "What mental health support does GitLab offer?"


def main() -> None:
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(PORT), "--log-level", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        started = time.time()
        while True:  # the models need about 40 s to load at startup
            try:
                if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if server.poll() is not None or time.time() - started > 180:
                raise SystemExit("the server did not start")
            time.sleep(2)
        print(f"server ready after {time.time() - started:.0f} s (models loaded)\n")

        print("--- login failures")
        print("no header     ->", httpx.post(f"{BASE}/chat", json={"question": "q"}).status_code)
        print(
            "unknown user  ->",
            httpx.post(
                f"{BASE}/chat", json={"question": "q"}, headers={"X-Dev-User": "x@y.z"}
            ).status_code,
        )

        print("--- warm-up chat (not measured; the server also warms its models at startup)")
        httpx.post(
            f"{BASE}/chat",
            json={"question": "How are team members paid?"},
            headers=HEADERS,
            timeout=60,
        )
        print(f"\n--- POST /chat: {QUESTION!r}")
        t0, seen = time.time(), set()
        with httpx.stream(
            "POST", f"{BASE}/chat", json={"question": QUESTION}, headers=HEADERS, timeout=60
        ) as r:
            event = None
            for line in r.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    if event not in seen or event == "done":
                        seen.add(event)
                        label = {
                            "sources": f"{len(data.get('sources', []))} sources",
                            "token": repr(data.get("text", "")[:30]),
                            "done": f"refused={data.get('refused')} cost=${data.get('cost_usd')} timings={data.get('timings_ms')}",
                        }.get(event, "")
                        print(f"  +{time.time() - t0:5.2f}s  {event:8} {label}")
                    if event == "done":
                        print("\nANSWER:", data["text"][:400])
                        print(
                            "CITES :",
                            [
                                (c["id"], c["document"].split("/")[-1], c["section"][-35:])
                                for c in data["citations"]
                            ],
                        )

        print("\n--- POST /admin/inspect (no LLM, no cost)")
        r = httpx.post(
            f"{BASE}/admin/inspect", json={"question": QUESTION}, headers=HEADERS, timeout=60
        )
        body = r.json()
        print(
            f"status {r.status_code} | keyword query: {body['keyword_query']!r} | context: {body['context_sources']} sources, {body['context_tokens']} tokens"
        )
        print(f"{'#':>2} {'dense':>5} {'kw':>4} {'rrf':>7} {'rerank':>7} ctx  section")
        for c in body["candidates"][:6]:
            print(
                f"{c['fused_rank']:>2} {c['dense_rank']!s:>5} {c['keyword_rank']!s:>4} {c['rrf_score']:7.4f} {c['rerank_score']!s:>7} {c['context_label']!s:>3}  {c['document'].split('/')[-1]} > {c['section'][-40:]}"
            )
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
