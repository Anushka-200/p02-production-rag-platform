"""Experiment 9 (task 6.12): does compressing a long chat history keep follow-up questions answerable?

A follow-up like "what other systems is it connected to?" is rewritten into a standalone question using the
chat so far. Production sends only the last 3 turns, each cut to 500 characters (app/retrieval/transform.py).
That is cheap, but a reference to something said EARLIER is lost. We build long conversations from the 10
eval follow-ups by placing 3-7 unrelated turns around the real one, then compare ways of handing history to
the rewriter, scoring Recall@5 of hybrid+rerank on the rewritten question (the real eval metric):

  none        no history (shows how much the rewrite matters)
  window-3    production: last 3 turns, 500 characters each
  wide        last 10 turns, 500 characters each (more context, more tokens)
  summary     older turns summarised by the model in <= 60 words, plus the last 2 turns verbatim

Two placements: the referenced turn is the LAST one, or it was 4 turns ago. Cost: a few cents.
Run: uv run python -m evals.history_experiments [--yes]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import psycopg

import app.retrieval.transform as transform
from app.config import get_settings
from app.cost_guard import add_cost_args, guard_from_args
from app.embeddings import get_embedder
from app.llm import complete
from app.retrieval.rerank import get_reranker
from app.retrieval.transform import TransformCache, format_history, rewrite_query
from evals.metrics import evaluate
from evals.run import _ranker, scopes_from_db
from evals.schema import EvalItem

SUMMARY_SYSTEM = (
    "Summarise this conversation between a user and a company-documents assistant in at most 60 words. "
    "Keep the specific topics, teams, tools and names that were discussed. No preamble."
)
ANSWERS = Path("data/cache/answers-baseline-2.jsonl")


def distractors(
    items: list[EvalItem], n: int, rng: random.Random, answers: dict[str, str]
) -> list[dict[str, str]]:
    pool = [i for i in items if i.kind == "single" and answers.get(i.id)]
    rng.shuffle(pool)
    out: list[dict[str, str]] = []
    for it in pool[:n]:
        out += [
            {"role": "user", "content": it.question},
            {"role": "assistant", "content": answers[it.id]},
        ]
    return out


def build(
    item: EvalItem, rng: random.Random, pool: list[EvalItem], answers: dict[str, str], back: int
) -> list[dict[str, str]]:
    """Real turn buried so that `back`-1 unrelated turns follow it (back=1: it is the last turn)."""
    before = distractors(pool, 8 - back, rng, answers)
    after = distractors(pool, back - 1, rng, answers) if back > 1 else []
    return before + item.history + after


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    args = parser.parse_args()
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa.jsonl")]
    follow = [i for i in items if i.kind == "follow_up" and i.gold]
    answers = {}
    for line in ANSWERS.read_text().splitlines():
        r = json.loads(line)
        if not r["refused"]:
            answers[r["item_id"]] = r["text"][:600]
    guard = guard_from_args(args, "history compression", 0.08)
    guard.confirm()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    scopes = scopes_from_db(conn)
    cache = TransformCache("data/cache/history.jsonl")
    rank = _ranker("hybrid+rerank", conn, embedder, reranker, scopes, s)

    def summarise(old: list[dict[str, str]]) -> tuple[str, float]:
        text = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in old)
        key = TransformCache.key("summary", None, text)
        if (hit := cache.get(key)) is not None:
            return str(hit["text"]), 0.0
        r = complete(
            [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": text}],
            max_tokens=120,
            name="history-summary",
        )
        cache.put(key, {"text": r.text})
        return str(r.text), float(r.cost_usd or 0.0)

    print(
        "| referenced turn | strategy | R@5 | R@1 | rewrite prompt chars (mean) | model cost per question |"
    )
    print("|---|---|---|---|---|---|")
    for back in (1, 4):
        histories = {
            i.id: build(
                i, random.Random(i.id), [x for x in items if x.tenant == i.tenant], answers, back
            )
            for i in follow
        }
        for name in ("none", "window-3", "wide", "summary"):
            rewritten, chars, cost = [], [], 0.0
            for item in follow:
                hist = histories[item.id]
                if name == "none":
                    rewritten.append(item)
                    chars.append(0)
                    continue
                if name == "summary":
                    text, c = summarise(hist[:-4])
                    cost += c
                    hist = [
                        {
                            "role": "user",
                            "content": f"(summary of the earlier conversation) {text}",
                        },
                        *hist[-4:],
                    ]
                if name == "wide":
                    transform.HISTORY_TURNS = 10
                else:
                    transform.HISTORY_TURNS = 3
                chars.append(len(format_history(hist)))
                t = rewrite_query(item.question, hist, cache=cache)
                guard.record(t.cost_usd)
                cost += t.cost_usd
                rewritten.append(item.model_copy(update={"question": t.text}))
            guard.record(cost)
            m = evaluate(rewritten, rank).overall
            print(
                f"| {'last turn' if back == 1 else '4 turns ago'} | {name} | {m['recall@5']:.2f} | {m['recall@1']:.2f} | {sum(chars) / len(chars):.0f} | ${cost / len(follow):.5f} |",
                flush=True,
            )  # noqa: E501
    transform.HISTORY_TURNS = 3
    print(f"{len(follow)} follow-up questions; {guard.summary()}")
    _ = Any


if __name__ == "__main__":
    main()
