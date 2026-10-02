import uuid
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.retrieval.fusion import Fused
from app.retrieval.search import Candidate
from evals.run import (
    AnswerRecord,
    answer_metrics,
    append_log_row,
    gold_in_context,
    percentile,
    reorder_with_reranker,
    split_items,
)
from evals.schema import EvalItem, GoldRef


def item(id_: str, kind="single", answerable=True, gold=(("a.md", "x"),)):  # type: ignore[no-untyped-def]
    return EvalItem(id=id_, tenant="A", kind=kind, question="q", answerable=answerable, corpus="gitlab",
                    gold=[GoldRef(doc=d, locator=loc) for d, loc in gold])  # fmt: skip


def rec(
    id_,
    kind="single",
    answerable=True,
    refused=False,
    cost=0.004,
    ms=None,
    gic=True,
    faith=None,
    corr=None,
):  # type: ignore[no-untyped-def]
    return AnswerRecord(id_, kind, answerable, refused, "text", "", cost,
                        ms or {"retrieve_ms": 50, "rerank_ms": 1000, "context_ms": 10, "answer_ms": 2000},
                        gic, [], faith, corr)  # fmt: skip


def test_percentiles_use_the_nearest_rank_method() -> None:
    xs = list(range(1, 101))
    assert percentile(xs, 0.5) == 50 and percentile(xs, 0.95) == 95 and percentile(xs, 1.0) == 100
    assert percentile([7], 0.95) == 7 and percentile([], 0.5) == 0.0


def test_follow_ups_are_split_off_from_the_headline() -> None:
    head, follow = split_items(
        [item("a"), item("b", kind="follow_up"), item("c", kind="multi_hop")]
    )
    assert [i.id for i in head] == ["a", "c"] and [i.id for i in follow] == ["b"]


def cand(name: str) -> Candidate:
    return Candidate(
        uuid.uuid4(), uuid.uuid4(), f"{name}.md", "S", None, None, [name], name, "", None, 1, 0.0
    )


def test_reranked_ranking_puts_the_reranked_pool_first_then_the_rest() -> None:
    fused = [
        Fused(candidate=cand(n), rrf_score=0.0, rank=i)
        for i, n in enumerate(["a", "b", "c", "d", "e"], start=1)
    ]

    class PreferC:
        name = "fake"

        def score(self, query, passages):  # type: ignore[no-untyped-def]
            return [5.0 if p == "c" else 0.0 for p in passages]

    ranking = reorder_with_reranker(fused, "q", PreferC(), candidates=3, max_words=100)
    assert [c.text for c in ranking] == [
        "c",
        "a",
        "b",
        "d",
        "e",
    ]  # pool [a,b,c] reranked; d, e untouched after it


def test_answer_metrics_count_refusals_context_hits_latency_and_cost() -> None:
    records = [
        rec("1"),
        rec("2", gic=False),
        rec("3", refused=True),  # answerable: 1 ok, 1 gold missing, 1 wrongly refused
        rec("u1", answerable=False, refused=True),
        rec("u2", answerable=False, refused=False),
        rec("f1", kind="follow_up", refused=True),  # follow-ups are excluded from the headline
    ]
    m = answer_metrics(records)
    assert (
        m["n"] == 5
        and m["refusal_accuracy"] == 0.5
        and m["wrongly_refused_rate"] == pytest.approx(1 / 3, abs=1e-3)
    )
    assert m["gold_in_context_rate"] == pytest.approx(
        2 / 3, abs=1e-3
    )  # records 1 and 3 had gic=True
    assert m["pre_llm_ms_p50"] == 1060 and m["latency_ms_p50"] == 3060
    assert m["cost_per_answer_usd"] == 0.004


def test_judge_scores_are_averaged_and_wrong_refusals_count_as_zero_correctness() -> None:
    records = [
        rec("1", faith=1.0, corr=1.0),
        rec("2", faith=0.5, corr=0.5),
        rec("3", refused=True, corr=0.0),  # refused: no faithfulness, correctness 0
    ]
    m = answer_metrics(records)
    assert m["faithfulness"] == 0.75 and m["faithfulness_n"] == 2 and m["correctness"] == 0.5


def test_gold_in_context_handles_headings_and_pdf_pages() -> None:
    src = [SimpleNamespace(document="a.md", anchors=["x", "y"], pages=None),
           SimpleNamespace(document="p.pdf", anchors=["p5-t"], pages="5-6")]  # fmt: skip
    assert gold_in_context(item("1", gold=(("a.md", "y"),)), src)
    assert not gold_in_context(item("2", gold=(("a.md", "z"),)), src)
    assert gold_in_context(item("3", gold=(("p.pdf", "page:6"),)), src)
    assert not gold_in_context(item("4", gold=(("p.pdf", "page:7"),)), src)


def test_the_benchmark_log_gets_one_row_per_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "benchmarks.md"
    path.write_text("# Benchmarks\n")
    report = {"retrieval": {"hybrid+rerank": {"n": 61, "headline": {"recall@5": 0.7, "mrr": 0.5, "ndcg@10": 0.55}}},
              "answers": {"faithfulness": 0.8, "correctness": 0.6, "refusal_accuracy": 0.9, "latency_ms_p50": 3000, "latency_ms_p95": 5000, "cost_per_answer_usd": 0.0035}}  # fmt: skip
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    append_log_row(path, "first", report, s)
    append_log_row(path, "second", report, s)
    text = path.read_text()
    assert (
        text.count("## Eval runs") == 1
        and text.count("| first |") == 1
        and text.count("| second |") == 1
    )
    assert "| 61 | 0.70 | 0.50 | 0.55 | 0.8 | 0.6 | 0.9 | 3000 / 5000 | 0.0035 |" in text


def test_answers_are_saved_as_they_are_made_and_a_resume_skips_the_saved_ones(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """A crash in the judge stage must not throw away money spent on answers."""
    import json
    from types import SimpleNamespace

    from evals import run as R

    calls = []

    def fake_ask(conn, embedder, reranker, question, scope, settings=None):  # type: ignore[no-untyped-def]
        calls.append(question)
        answer = SimpleNamespace(refused=False, text="a", reason="", cost_usd=0.004, citations=[])
        return SimpleNamespace(
            answer=answer,
            timings_ms={"retrieve_ms": 1.0},
            context=SimpleNamespace(sources=[], text="CTX"),
        )

    monkeypatch.setattr(R, "ask", fake_ask)
    guard = SimpleNamespace(record=lambda c: None, spent_usd=0.0)
    items = [item(f"q{i}") for i in range(3)]
    for it, q in zip(items, "abc", strict=True):
        it.question = q
    path = tmp_path / "answers.jsonl"
    first = R.run_answers(None, None, None, items, {"A": None}, None, guard, path)  # type: ignore[arg-type]
    assert len(first) == 3 and len(path.read_text().splitlines()) == 3
    assert (
        json.loads(path.read_text().splitlines()[0])["context"] == "CTX"
    )  # the judge's input is saved too
    calls.clear()
    again = R.run_answers(None, None, None, items, {"A": None}, None, guard, path, resume=True)  # type: ignore[arg-type]
    assert calls == [] and [r.item_id for r in again] == [
        "q0",
        "q1",
        "q2",
    ]  # nothing re-asked, nothing re-paid


def test_a_judge_error_is_recorded_and_does_not_stop_the_run() -> None:
    from evals.judge import JudgeError
    from evals.run import run_judge

    class FailsOnce:
        def faithfulness(self, answer, context):  # type: ignore[no-untyped-def]
            if answer == "bad":
                raise JudgeError("not valid JSON: cut off")
            return SimpleNamespace(score=1.0, cost_usd=0.01)

        def correctness(self, question, answer, gold):  # type: ignore[no-untyped-def]
            return SimpleNamespace(score=1.0, cost_usd=0.002)

    items = [item("a"), item("b")]
    for it in items:
        it.gold_answer = "gold"
    records = [rec("a"), rec("b")]
    records[0].text, records[1].text = "bad", "good"
    run_judge(items, records, FailsOnce(), SimpleNamespace(record=lambda c: None), workers=1)  # type: ignore[arg-type]
    assert "cut off" in records[0].judge_error and records[0].faithfulness is None
    assert records[1].faithfulness == 1.0 and records[1].judge_error == ""
