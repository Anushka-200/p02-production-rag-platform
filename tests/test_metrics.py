import math
from types import SimpleNamespace

import pytest

from evals.metrics import (
    covers,
    evaluate,
    first_ranks,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evals.schema import EvalItem, GoldRef


def res(doc: str, anchors=(), pages=(None, None)):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        source_uri=doc, anchors=list(anchors), page_start=pages[0], page_end=pages[1]
    )


def item(
    id_: str,
    gold: list[tuple[str, str]],
    kind="single",
    tenant="A",
    corpus="gitlab",
    answerable=True,
    tags=(),
):  # type: ignore[no-untyped-def]
    return EvalItem(id=id_, tenant=tenant, kind=kind, question="q", answerable=answerable, corpus=corpus,
                    gold=[GoldRef(doc=d, locator=loc) for d, loc in gold], tags=list(tags))  # fmt: skip


# ---- what counts as a hit ------------------------------------------------------------------------
def test_a_heading_gold_needs_the_same_document_and_the_anchor() -> None:
    g = GoldRef(doc="a.md", locator="time-off")
    assert covers(res("a.md", ["intro", "time-off"]), g)  # a merged chunk covers several anchors
    assert not covers(res("a.md", ["intro"]), g) and not covers(res("b.md", ["time-off"]), g)


def test_a_page_gold_is_covered_by_any_chunk_whose_page_range_includes_it() -> None:
    g = GoldRef(doc="x.pdf", locator="page:6")
    assert covers(res("x.pdf", pages=(5, 6)), g) and covers(res("x.pdf", pages=(6, 6)), g)
    assert not covers(res("x.pdf", pages=(7, 8)), g) and not covers(res("x.pdf"), g)
    assert covers(res("x.pdf", pages=(6, None)), g)


def test_first_ranks_report_where_each_gold_location_first_appears() -> None:
    results = [res("z.md"), res("a.md", ["one"]), res("a.md", ["one"]), res("b.md", ["two"])]
    gold = [
        GoldRef(doc="a.md", locator="one"),
        GoldRef(doc="b.md", locator="two"),
        GoldRef(doc="c.md", locator="x"),
    ]
    assert first_ranks(results, gold) == [
        2,
        4,
        None,
    ]  # the second hit on 'one' (rank 3) does not count again


# ---- the numbers from the learning-step quiz ---------------------------------------------------
QUIZ = [[1], [3], [None], [2], [5]]  # five questions, one gold location each


def test_quiz_recall_at_5_is_0_8_and_mrr_is_0_41() -> None:
    assert sum(recall_at_k(r, 5) for r in QUIZ) / 5 == pytest.approx(0.8)
    assert sum(reciprocal_rank(r) for r in QUIZ) / 5 == pytest.approx(
        (1 + 1 / 3 + 0 + 1 / 2 + 1 / 5) / 5
    )  # 0.4067


def test_recall_and_hit_with_several_gold_locations() -> None:
    assert recall_at_k([2, None], 5) == 0.5 and hit_at_k([2, None], 5) == 1.0  # one of two found
    assert recall_at_k([2, 9], 5) == 0.5 and recall_at_k([2, 9], 10) == 1.0
    assert recall_at_k([], 5) == 0.0


def test_reciprocal_rank_uses_the_first_correct_result() -> None:
    assert (
        reciprocal_rank([4, 2]) == 0.5
        and reciprocal_rank([None, None]) == 0.0
        and reciprocal_rank([1]) == 1.0
    )


def test_ndcg_rewards_ranking_gold_high() -> None:
    assert ndcg_at_k([1, 2], 10) == pytest.approx(1.0)  # perfect: gold at the very top
    # gold at ranks 1 and 10: (1 + 1/log2(11)) / (1 + 1/log2(3))
    assert ndcg_at_k([1, 10], 10) == pytest.approx((1 + 1 / math.log2(11)) / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([1, 10], 5) < ndcg_at_k([1, 10], 10)  # rank 10 is outside the top 5
    assert ndcg_at_k([None], 10) == 0.0 and ndcg_at_k([], 10) == 0.0
    assert ndcg_at_k([3], 10) < ndcg_at_k([1], 10) == pytest.approx(1.0)


# ---- the whole evaluation -----------------------------------------------------------------------
def test_evaluate_averages_over_answerable_questions_and_splits_by_subset() -> None:
    items = [
        item("a-1", [("a.md", "x")], tags=["table"]),
        item("a-2", [("b.md", "y")], kind="multi_hop", tenant="B", corpus="posthog"),
        item("u-1", [], kind="unanswerable", answerable=False),  # not part of retrieval metrics
    ]
    results = {"a-1": [res("a.md", ["x"])], "a-2": [res("z.md"), res("b.md", ["y"])]}
    report = evaluate(items, lambda it: results[it.id])
    assert report.n == 2 and [i.ranks for i in report.items] == [[1], [2]]
    assert report.overall["recall@1"] == 0.5 and report.overall["recall@5"] == 1.0
    assert report.overall["mrr"] == pytest.approx(0.75)
    assert (
        report.subsets["kind=multi_hop"]["mrr"] == 0.5
        and report.subsets["kind=multi_hop"]["n"] == 1
    )
    assert report.subsets["tenant=A"]["recall@1"] == 1.0 and report.subsets["tag=table"]["n"] == 1
    assert "kind=unanswerable" not in report.subsets


def test_evaluate_with_no_answerable_items_is_empty_not_a_crash() -> None:
    report = evaluate([item("u", [], kind="unanswerable", answerable=False)], lambda it: [])
    assert report.n == 0 and report.overall == {} and report.subsets == {}
