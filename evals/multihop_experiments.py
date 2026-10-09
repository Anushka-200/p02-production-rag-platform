"""Experiment: do general retrieval-to-context tricks help questions that need SEVERAL places? (dev questions only, free)

For every answerable dev question of the multi-place kinds (multi_hop, broad, relationship) plus single and paraphrase (to check we do not
hurt them) the pipeline runs up to the context the answer model would see, and we check whether ANY / ALL gold places (or accepted
alternatives) are inside it. No model calls.

  baseline  rerank the top 10 fused candidates, keep 8 (production)
  wide      rerank 20 candidates, keep 8 (is the pool the limit?)
  diverse   rerank 20, then pick the 8 greedily with a penalty for each already-chosen chunk of the SAME section (spread, not repeat)
  prf       a second search for "question + the opening of the best chunk" (pseudo-relevance feedback: a cheap stand-in for a second hop),
            merged with the first by RRF, reranked 20, keep 8
  prf+diverse

Run: uv run python -m evals.multihop_experiments [--split dev]
"""

import argparse
import json
import statistics
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.embeddings import get_embedder, get_token_counter
from app.retrieval.context import build_context
from app.retrieval.expand import retrieve_expanded
from app.retrieval.fusion import merge_fused
from app.retrieval.rerank import Reranked, get_reranker, rerank
from app.retrieval.retrieve import retrieve
from evals.context_experiments import covered
from evals.run import scopes_from_db
from evals.schema import EvalItem
from evals.splits import split_of

KINDS = ("multi_hop", "broad", "relationship", "single", "paraphrase")
PENALTY = 1.5  # rerank-score points taken off per already-chosen chunk from the same section


def diversify(ranked: list[Reranked], keep: int, penalty: float = PENALTY) -> list[Reranked]:
    chosen: list[Reranked] = []
    pool = list(ranked)
    while pool and len(chosen) < keep:

        def adjusted(r: Reranked) -> float:
            same = sum(
                1
                for c in chosen
                if (c.fused.candidate.document_id, c.fused.candidate.section_path)
                == (r.fused.candidate.document_id, r.fused.candidate.section_path)
            )  # noqa: E501
            return (r.rerank_score or 0.0) - penalty * same

        best = max(pool, key=adjusted)
        chosen.append(best)
        pool.remove(best)
    return [Reranked(c.fused, c.rerank_score, i) for i, c in enumerate(chosen, 1)]


def build(
    name: str,
    item: EvalItem,
    conn: psycopg.Connection,
    embedder: Any,
    reranker: Any,
    scope: Any,
    s: Settings,
) -> list[Reranked]:
    first = retrieve_expanded(
        conn, embedder, item.question, scope, history=item.history or None, settings=s
    )
    pool = first.items
    question = first.question or item.question
    if "prf" in name:
        seed = rerank(pool, question, reranker, candidates=10, top=1, max_words=s.rerank_max_words)
        if seed:
            opening = " ".join(seed[0].fused.candidate.text.split()[:40])
            second = retrieve(
                conn, embedder, f"{question} {opening}", scope, settings=s, mode="hybrid"
            ).items
            pool = merge_fused([pool, second], k=s.rrf_k)
    wide = name != "baseline"
    ranked = rerank(
        pool,
        question,
        reranker,
        candidates=20 if wide else s.rerank_candidates,
        top=20 if wide else s.rerank_top,
        max_words=s.rerank_max_words,
    )
    if "diverse" in name:
        return diversify(ranked, s.rerank_top)
    return ranked[: s.rerank_top]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--configs", default="baseline,wide,diverse,prf,prf+diverse")
    args = parser.parse_args()
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa.jsonl")]
    items = [i for i in items if i.answerable and i.kind in KINDS and split_of(i.id) == args.split]
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker, scopes, count = (
        get_embedder(s),
        get_reranker(s),
        scopes_from_db(conn),
        get_token_counter(s),
    )
    print(f"{len(items)} {args.split} questions; any / all gold places in the context")
    out = {}
    for name in args.configs.split(","):
        row = {}
        for kind in KINDS:
            members = [i for i in items if i.kind == kind]
            anys, alls, toks = [], [], []
            for item in members:
                ranked = build(name, item, conn, embedder, reranker, scopes[item.tenant], s)
                ctx = build_context(ranked, s.context_budget_tokens, count)
                hits = [covered(g, ctx.sources) for g in item.gold]
                anys.append(any(hits))
                alls.append(all(hits))
                toks.append(ctx.tokens)
            row[kind] = (
                len(members),
                statistics.mean(anys),
                statistics.mean(alls),
                statistics.mean(toks),
            )
        out[name] = row
        print(
            f"{name:14} "
            + "  ".join(
                f"{k[:5]} n={v[0]:2} any {v[1]:.2f} all {v[2]:.2f}" for k, v in row.items()
            ),
            flush=True,
        )
    json.dump(out, open(f"evals/results/multihop-experiments-{args.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
