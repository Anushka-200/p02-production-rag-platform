"""The CI eval: a tiny corpus, the real ingestion and hybrid retrieval, a stand-in embedder, no API calls."""

from evals.mini_eval import HashEmbedder, load_items, run_mini_eval
from tests.test_search import World


def test_the_mini_eval_set_is_what_it_says_it_is() -> None:
    items = load_items()
    assert len(items) == 15
    assert sum(i.answerable for i in items) == 12 and all(i.reviewed for i in items)


def test_the_hashed_embedder_is_deterministic_and_unit_length() -> None:
    e = HashEmbedder()
    a, b = e.embed_query("annual leave days"), e.embed_query("annual leave days")
    assert a == b and abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_hybrid_retrieval_finds_the_gold_place_for_most_questions(db_schema) -> None:  # type: ignore[no-untyped-def]
    who = World(db_schema).person("mini", "ann")
    report = run_mini_eval(db_schema, who.org_id, who.user_id)
    print("MINI EVAL", {k: round(v, 3) for k, v in report.overall.items()})
    assert report.n == 12
    # thresholds sit below what a healthy pipeline scores, so a break (not noise) fails the build
    assert report.overall["recall@5"] >= 0.90
    assert report.overall["mrr"] >= 0.80
