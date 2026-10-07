"""Parent expansion: a small hit is swapped for the larger section it belongs to."""

import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.rag import ask
from app.retrieval.fusion import Fused
from app.retrieval.parents import expand_to_parents
from app.retrieval.rerank import Reranked
from app.retrieval.search import Scope, dense_search
from tests.test_api import FakeEmbedder
from tests.test_search import World, vec

CHILD_A, CHILD_B = "Parental leave is 16 weeks.", "It may be taken within 12 months."
PARENT = "Leave policies. Parental leave is 16 weeks. It may be taken within 12 months. Sick leave is 8 days."


@pytest.fixture
def w(db_schema) -> World:  # type: ignore[no-untyped-def]
    world = World(db_schema)
    world.ann = world.person("acme", "ann")  # type: ignore[attr-defined]
    world.amy = world.person("acme", "amy")  # type: ignore[attr-defined]
    return world


def link(
    w: World,
    who: Scope,
    org: str,
    visibility: str = "org",
    parent_text: str = PARENT,
    uri: str = "leave.md",
) -> None:
    """Two child chunks that share one parent chunk (level 1, never searched)."""
    w.chunk(who, org, parent_text, None, visibility=visibility, level=1, uri=uri)
    w.chunk(who, org, CHILD_A, vec(1), visibility=visibility, uri=uri)
    w.chunk(who, org, CHILD_B, vec(1, 0.1, 2), visibility=visibility, uri=uri)
    parent = w.conn.execute(
        "SELECT id FROM chunks WHERE level = 1 ORDER BY created_at DESC LIMIT 1"
    ).fetchone()[0]
    w.conn.execute(
        "UPDATE chunks SET parent_id = %s WHERE level = 0 AND text IN (%s, %s)",
        (parent, CHILD_A, CHILD_B),
    )
    w.conn.execute(
        "UPDATE chunks SET page_start = 3, page_end = 4, anchors = '{leave}' WHERE id = %s",
        (parent,),
    )


def hits(w: World, who: Scope) -> list[Reranked]:
    found = dense_search(w.conn, vec(1), who, k=10)
    return [Reranked(Fused(c, 0.0, rank=i), None, i) for i, c in enumerate(found, start=1)]


def test_a_hit_is_replaced_by_its_parent_with_the_parents_pages_and_anchors(w: World) -> None:
    link(w, w.ann, "acme")
    out = expand_to_parents(w.conn, hits(w, w.ann)[:1], w.ann)
    c = out[0].fused.candidate
    assert c.text == PARENT and (c.page_start, c.page_end) == (3, 4) and "leave" in c.anchors
    assert out[0].rank == 1  # it keeps the position of the hit it replaced


def test_two_hits_with_the_same_parent_become_one_source(w: World) -> None:
    link(w, w.ann, "acme")
    items = hits(w, w.ann)
    assert len(items) == 2
    out = expand_to_parents(w.conn, items, w.ann)
    assert [i.fused.candidate.text for i in out] == [PARENT]


def test_a_hit_without_a_parent_stays_as_it_is(w: World) -> None:
    w.chunk(w.ann, "acme", CHILD_A, vec(1))
    out = expand_to_parents(w.conn, hits(w, w.ann), w.ann)
    assert [i.fused.candidate.text for i in out] == [CHILD_A]


def test_a_parent_larger_than_the_cap_is_not_used(w: World) -> None:
    link(w, w.ann, "acme", parent_text="word " * 900)
    out = expand_to_parents(
        w.conn, hits(w, w.ann), w.ann, max_parent_tokens=700, count=lambda t: len(t.split())
    )
    assert {i.fused.candidate.text for i in out} == {CHILD_A, CHILD_B}  # the small chunks stay


def test_a_parent_the_asker_may_not_read_is_never_swapped_in(w: World) -> None:
    link(w, w.ann, "acme", visibility="org")
    w.conn.execute("UPDATE chunks SET visibility = 'org' WHERE level = 1")
    items = hits(w, w.ann)  # found by ann
    # the same hits, asked by an anonymous visitor: the org-level parent is invisible to them
    out = expand_to_parents(w.conn, items, Scope(w.ann.org_id, None))
    assert {i.fused.candidate.text for i in out} == {CHILD_A, CHILD_B}


def test_a_parent_in_another_organisation_is_never_read(w: World) -> None:
    link(w, w.ann, "acme")
    bob = w.person("globex", "bob")
    out = expand_to_parents(
        w.conn, hits(w, w.ann), bob
    )  # bob asks with acme's hits: a forged scope
    assert {i.fused.candidate.text for i in out} == {CHILD_A, CHILD_B}


def test_with_the_setting_on_the_answer_model_reads_the_parent_section(w: World) -> None:
    link(w, w.ann, "acme")
    seen: list = []  # type: ignore[type-arg]

    def complete_fn(messages, **kw):  # type: ignore[no-untyped-def]
        seen.append(messages[-1]["content"])
        reply = {"supported": True, "answer": "16 weeks [S1].", "citations": ["S1"]}
        return SimpleNamespace(
            text=json.dumps(reply),
            model="fake",
            cost_usd=0.001,
            input_tokens=1,
            output_tokens=1,
            latency_s=0.1,
        )

    def run(**kw) -> None:  # type: ignore[no-untyped-def]
        settings = Settings(
            _env_file=None,
            reranker_provider="none",
            retrieval_candidates=10,
            query_rewrite=False,
            **kw,
        )  # type: ignore[call-arg]
        ask(
            w.conn,
            FakeEmbedder(),
            None,
            "how long is parental leave?",
            w.ann,
            settings=settings,
            complete_fn=complete_fn,
        )

    run()
    run(parent_expansion=True)
    plain, expanded = seen
    assert "Sick leave is 8 days" not in plain and "Sick leave is 8 days" in expanded
