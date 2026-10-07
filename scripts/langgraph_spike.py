"""ADR 010 spike: the same retrieve > rerank > assemble > answer flow as a LangGraph graph.

Reuses our retrieval and answer functions as graph nodes (re-writing them in LangChain's own
retrievers would drop our scores, tenant filter and citation checks). What it measures: lines of
code, latency overhead of the framework, and whether the answers match plain Python. A few cents.

Run: uv run --with langgraph python -m scripts.langgraph_spike [--yes] [--max-usd 0.3]
"""

import argparse
import json
import statistics
import time
from typing import Any, TypedDict

import psycopg
from langgraph.graph import END, START, StateGraph

from app.answer import Answer, answer_question
from app.config import get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.embeddings import get_embedder, get_token_counter
from app.log import setup_logging
from app.rag import ask
from app.retrieval.context import Context, build_context
from app.retrieval.expand import retrieve_expanded
from app.retrieval.rerank import Reranked, get_reranker, rerank
from app.retrieval.search import Scope


class State(TypedDict, total=False):
    question: str
    scope: Scope  # set by the caller from the verified login, never by a node
    reranked: list[Reranked]
    context: Context
    answer: Answer


def build_graph(conn: psycopg.Connection, embedder: Any, reranker: Any, s: Any) -> Any:
    count = get_token_counter(s)

    def retrieve(state: State) -> State:
        r = retrieve_expanded(conn, embedder, state["question"], state["scope"], settings=s)
        top = rerank(r.items, r.question or state["question"], reranker, candidates=s.rerank_candidates,
                     top=s.rerank_top, max_words=s.rerank_max_words)  # fmt: skip
        return {"reranked": top}

    def assemble(state: State) -> State:
        return {"context": build_context(state["reranked"], s.context_budget_tokens, count)}

    def answer(state: State) -> State:
        return {"answer": answer_question(state["question"], state["context"])}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("assemble", assemble)
    g.add_node("answer", answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "assemble")
    g.add_edge("assemble", "answer")
    g.add_edge("answer", END)
    return g.compile()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    args = parser.parse_args()
    setup_logging("ERROR")
    s = get_settings()
    rows = [json.loads(line) for line in open("evals/qa.jsonl")]
    pick = [r for r in rows if r["tenant"] == "A" and r["kind"] == "single"][:8]
    guard = guard_from_args(
        args,
        "langgraph spike",
        estimate_llm_cost(s.llm_default_model, 4500, 250, calls=len(pick) * 2),
    )  # noqa: E501
    guard.confirm()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    org, user = conn.execute(
        "SELECT o.id, u.id FROM orgs o JOIN users u ON u.org_id = o.id WHERE o.name = %s",
        ("Tenant A (GitLab handbook)",),
    ).fetchone()  # type: ignore[misc]
    scope = Scope(org, user)
    graph = build_graph(conn, embedder, reranker, s)
    ask(conn, embedder, reranker, "warm up", scope, settings=s)  # load models before timing
    plain_t, graph_t, same_cites = [], [], 0
    for q in pick:
        t = time.perf_counter()
        a = ask(conn, embedder, reranker, q["question"], scope, settings=s)
        plain_t.append(time.perf_counter() - t)
        t = time.perf_counter()
        out = graph.invoke({"question": q["question"], "scope": scope})
        graph_t.append(time.perf_counter() - t)
        guard.record(a.answer.cost_usd + out["answer"].cost_usd)
        same_cites += {c.document for c in a.answer.citations} == {
            c.document for c in out["answer"].citations
        }
    print(f"questions {len(pick)} | same cited documents: {same_cites}/{len(pick)}")
    print(
        f"plain python  p50 {statistics.median(plain_t):.2f}s  mean {statistics.mean(plain_t):.2f}s"
    )
    print(
        f"langgraph     p50 {statistics.median(graph_t):.2f}s  mean {statistics.mean(graph_t):.2f}s"
    )
    print(guard.summary())


if __name__ == "__main__":
    main()
