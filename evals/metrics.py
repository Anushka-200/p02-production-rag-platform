"""Retrieval metrics: where in the ranked results does the right place appear?

Gold is (document, locator), not a chunk id, so the numbers stay comparable when the chunking changes.
A result "covers" a gold location when it comes from that document and its anchors include the
gold heading (Markdown/HTML) or its page range includes the gold page (PDF).

Per question we find, for each gold location, the rank of the FIRST result covering it:
    ranks = [3, None]    # first gold found at rank 3, second gold never found
and compute every metric from that list.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from evals.schema import EvalItem, GoldRef, Place

DEFAULT_KS = (1, 5, 10, 50)


class Result(Protocol):
    source_uri: str
    anchors: list[str]
    page_start: int | None
    page_end: int | None


def covers(result: Result, gold: GoldRef) -> bool:
    """A result covers a gold place if it covers the place itself or one of its accepted alternatives."""
    return any(_covers_one(result, p) for p in gold.places())


def _covers_one(result: Result, gold: Place) -> bool:
    if result.source_uri != gold.doc:
        return False
    if gold.locator.startswith("page:"):
        page = int(gold.locator[5:])
        return result.page_start is not None and result.page_start <= page <= (
            result.page_end or result.page_start
        )
    return gold.locator in result.anchors


def first_ranks(results: Sequence[Result], gold: Sequence[GoldRef]) -> list[int | None]:
    """For each gold location: the 1-based rank of the first result that covers it, else None."""
    ranks: list[int | None] = []
    for g in gold:
        ranks.append(next((i for i, r in enumerate(results, start=1) if covers(r, g)), None))
    return ranks


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of the gold locations found in the top k (1.0 = all of them)."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    return float(any(r is not None and r <= k for r in ranks))


def reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """1 / rank of the first correct result (1.0 if first, 0.5 if second, 0 if none)."""
    found = [r for r in ranks if r is not None]
    return 1.0 / min(found) if found else 0.0


def ndcg_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Rewards finding gold locations HIGH in the list. 1.0 = all gold at the very top.

    Each gold location earns 1 / log2(rank + 1) at the rank where it is first covered; the ideal
    puts every gold location at ranks 1, 2, 3... so the score is relative to that ideal.
    """
    if not ranks:
        return 0.0
    dcg = sum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= k)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(ranks), k) + 1))
    return dcg / ideal


@dataclass
class ItemResult:
    item_id: str
    ranks: list[int | None]
    metrics: dict[str, float]


@dataclass
class Report:
    overall: dict[str, float]
    subsets: dict[str, dict[str, float]]  # e.g. "kind=multi_hop" -> metrics
    items: list[ItemResult] = field(default_factory=list)
    n: int = 0


def item_metrics(ranks: Sequence[int | None], ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    m: dict[str, float] = {"mrr": reciprocal_rank(ranks), "ndcg@10": ndcg_at_k(ranks, 10)}
    for k in ks:
        m[f"recall@{k}"] = recall_at_k(ranks, k)
        m[f"hit@{k}"] = hit_at_k(ranks, k)
    return m


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return (
        {name: round(sum(r[name] for r in rows) / len(rows), 4) for name in rows[0]} if rows else {}
    )


def evaluate(
    items: Sequence[EvalItem],
    results_fn: Callable[[EvalItem], Sequence[Any]],
    ks: Sequence[int] = DEFAULT_KS,
) -> Report:
    """Run `results_fn` (question -> ranked results) for every ANSWERABLE item and average the metrics.

    Unanswerable items have no gold location; they are judged by refusal accuracy, not retrieval.
    """
    rows: list[ItemResult] = []
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in items:
        if not item.answerable or not item.gold:
            continue
        ranks = first_ranks(results_fn(item), item.gold)
        metrics = item_metrics(ranks, ks)
        rows.append(ItemResult(item.id, ranks, metrics))
        for label in (
            f"kind={item.kind}",
            f"tenant={item.tenant}",
            f"corpus={item.corpus}",
            *[f"tag={t}" for t in item.tags],
        ):
            groups[label].append(metrics)
    return Report(
        overall=_mean([r.metrics for r in rows]),
        subsets={label: {**_mean(g), "n": len(g)} for label, g in sorted(groups.items())},
        items=rows,
        n=len(rows),
    )
