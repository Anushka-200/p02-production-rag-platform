"""LIVE end-to-end run on real questions: retrieve > rerank > assemble > grounded answer (costs cents).

Run: uv run python -m scripts.ask_demo [--yes] [--max-usd 0.3]
"""

import argparse
import json

import psycopg

from app.config import get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.embeddings import get_embedder
from app.log import setup_logging
from app.rag import ask
from app.retrieval.rerank import get_reranker
from app.retrieval.search import Scope


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    args = parser.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    items = [json.loads(line) for line in open("evals/qa.jsonl")]
    pick = (
        [i for i in items if i["kind"] == "single" and i["tenant"] == "A"][3:8]
        + [i for i in items if i["kind"] == "single" and i["tenant"] == "B"][:2]
        + [i for i in items if i["kind"] == "multi_hop"][:1]
        + [i for i in items if i["kind"] == "unanswerable"][:4]
    )
    guard = guard_from_args(
        args, "ask demo", estimate_llm_cost(s.llm_default_model, 4500, 250, calls=len(pick) * 2)
    )
    guard.confirm()

    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    scopes = {}
    for t, name in {"A": "Tenant A (GitLab handbook)", "B": "Tenant B (PostHog handbook)"}.items():
        org, user = conn.execute(
            "SELECT o.id, u.id FROM orgs o JOIN users u ON u.org_id = o.id WHERE o.name = %s",
            (name,),
        ).fetchone()
        scopes[t] = Scope(org, user)

    refused_unanswerable = answered_ok = 0
    for item in pick:
        r = ask(conn, embedder, reranker, item["question"], scopes[item["tenant"]], settings=s)
        guard.record(r.answer.cost_usd)
        a = r.answer
        gold_in_context = any(
            src.document == g["doc"]
            and (g["locator"] in src.anchors or g["locator"].startswith("page:"))
            for src in r.context.sources
            for g in item["gold"]
        )
        print(f"\n[{item['id']}] {item['question']}")
        print(f"   {'REFUSED' if a.refused else 'ANSWER '}: {a.text[:230]}")
        if a.citations:
            print(
                "   cites  :",
                "; ".join(
                    f"{c.id} {c.document.split('/')[-1]} > {c.section[-40:]}" for c in a.citations
                ),
            )
        if a.refused:
            print("   reason :", a.reason)
        if item["answerable"]:
            print(
                f"   gold section was in the context: {gold_in_context} | top rerank scores: {[round(x.rerank_score, 1) for x in r.reranked[:3]]}"
            )
            answered_ok += (not a.refused) and gold_in_context
        else:
            refused_unanswerable += a.refused
        print("   timing :", r.timings_ms, f"| llm ${a.cost_usd:.4f}")
    print(
        f"\nanswerable answered with the gold section in context: {answered_ok}/{sum(i['answerable'] for i in pick)}"
        f" | unanswerable correctly refused: {refused_unanswerable}/{sum(not i['answerable'] for i in pick)}"
    )
    print(guard.summary())


if __name__ == "__main__":
    main()
