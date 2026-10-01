import json

import pytest

from app.llm import LLMError, LLMResult
from evals import corpus as C
from evals.draft import (
    PLAN,
    DraftError,
    ask,
    draft_follow_up,
    draft_multi,
    draft_single,
    draft_unanswerable,
    parse_json,
    plan_work,
    run_draft,
)
from evals.schema import EvalItem

SENT_A = "Employees receive twenty vacation days every year."
SENT_B = "Managers approve requests within five working days."
BODY = f"{SENT_A} {SENT_B} " + "Additional policy words describe the process in plain terms. " * 10


def passage(
    uri: str, n: int = 0, corpus: str = "gitlab", tenant: str = "A", text: str = BODY
) -> C.Passage:
    return C.Passage(corpus, tenant, uri, f"sec-{n}", "Title", f"Title > S{n}", text, True)


def result(text: str) -> LLMResult:
    return LLMResult(text, "m", 100, 50, 0.001, 0.1, 1)


def fake_model(messages, **kw):  # type: ignore[no-untyped-def]
    """Answers each kind of prompt with valid JSON (support sentences exist in every passage)."""
    prompt = messages[-1]["content"]
    if "two-turn" in prompt:
        out = {
            "question_1": "How many vacation days?",
            "answer_1": "Twenty.",
            "follow_up": "And who approves them?",
            "follow_up_answer": "Managers.",
            "support": [SENT_A],
        }
    elif "BOTH passages" in prompt:
        out = {
            "possible": True,
            "question": "How many days, and who approves?",
            "gold_answer": "Twenty; managers.",
            "support_a": SENT_A,
            "support_b": SENT_B,
        }
    elif "does NOT answer" in prompt:
        out = {"question": "What is the sabbatical policy?", "key_terms": ["sabbatical", "policy"]}
    else:
        out = {
            "question": "How long is the yearly leave?",
            "gold_answer": "Twenty days.",
            "support": [SENT_A],
        }
    return result(json.dumps(out))


# ---- small helpers -----------------------------------------------------------------------------
def test_parse_json_accepts_code_fences_and_rejects_garbage() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(DraftError, match="not valid JSON"):
        parse_json("Sure! Here is your question")
    with pytest.raises(DraftError, match="object"):
        parse_json("[1, 2]")


def test_verbatim_check_ignores_markup_links_and_spacing() -> None:
    text = "Employees get **twenty** days, see [the policy](https://x.y/z) for details."
    assert C.contains_verbatim(text, "Employees get twenty days, see the policy for details.")
    assert not C.contains_verbatim(text, "Employees get thirty days")
    assert not C.contains_verbatim(text, "  ")


def test_eligibility_wants_real_prose() -> None:
    assert C.eligible(BODY)
    assert not C.eligible("Too short to ask about.")
    links = " ".join(f"[page {i}](/handbook/page-{i})" for i in range(40))
    assert not C.eligible(links)  # a list of links is navigation, not content
    assert not C.eligible(BODY + " {{< shortcode >}}")


# ---- asking the model, with one retry ----------------------------------------------------------
def test_a_bad_answer_is_retried_once_with_the_reason() -> None:
    seen = []

    def flaky(messages, **kw):  # type: ignore[no-untyped-def]
        seen.append(messages[-1]["content"])
        return result("not json at all" if len(seen) == 1 else '{"question": "ok"}')

    out = ask(flaky, "prompt", lambda d: None)
    assert out == {"question": "ok"} and "rejected" in seen[1] and "not valid JSON" in seen[1]


def test_two_bad_answers_give_up() -> None:
    with pytest.raises(DraftError):
        ask(lambda m, **kw: result("nope"), "prompt", lambda d: None)


def test_invented_support_sentences_are_rejected_and_retried() -> None:
    answers = iter(
        [
            {
                "question": "Q?",
                "gold_answer": "A.",
                "support": ["A sentence that is not in the passage at all."],
            },
            {"question": "Q?", "gold_answer": "A.", "support": [SENT_A]},
        ]
    )
    item = draft_single(passage("d.md"), lambda m, **kw: result(json.dumps(next(answers))))
    assert item.support == [SENT_A]  # the second, honest answer was used


def test_hallucinated_support_twice_fails() -> None:
    bad = result(
        json.dumps({"question": "Q?", "gold_answer": "A.", "support": ["Made up sentence."]})
    )
    with pytest.raises(DraftError, match="not in the passage"):
        draft_single(passage("d.md"), lambda m, **kw: bad)


# ---- the five kinds ----------------------------------------------------------------------------
def test_single_item_has_gold_location_and_tags() -> None:
    table = BODY + "\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    item = draft_single(passage("gitlab/x.md", 3, text=table), fake_model)
    assert item.kind == "single" and item.answerable and not item.reviewed
    assert [(g.doc, g.locator) for g in item.gold] == [("gitlab/x.md", "sec-3")]
    assert "table" in item.tags


def test_multi_hop_has_two_gold_locations_or_none_if_unrelated() -> None:
    item = draft_multi(passage("a.md", 1), passage("b.md", 2), fake_model)
    assert item is not None and len(item.gold) == 2 and "multi_hop" in item.tags
    unrelated = lambda m, **kw: result('{"possible": false}')  # noqa: E731
    assert draft_multi(passage("a.md"), passage("b.md"), unrelated) is None


def test_follow_up_carries_the_earlier_turns() -> None:
    item = draft_follow_up(passage("a.md"), fake_model)
    assert item.history == [
        {"role": "user", "content": "How many vacation days?"},
        {"role": "assistant", "content": "Twenty."},
    ]
    assert item.question == "And who approves them?" and item.gold[0].doc == "a.md"


def test_unanswerable_is_flagged_when_the_corpus_seems_to_answer_it() -> None:
    clean = draft_unanswerable(
        passage("a.md"), passage("b.md"), [("x#1", "nothing relevant")], fake_model
    )
    assert (
        clean.kind == "unanswerable"
        and not clean.answerable
        and clean.gold == []
        and clean.flags == []
    )
    leaky = draft_unanswerable(
        passage("a.md"),
        passage("b.md"),
        [("other.md#sab", "our sabbatical policy grants six weeks")],
        fake_model,
    )
    assert (
        len(leaky.flags) == 1
        and "maybe_answerable" in leaky.flags[0]
        and "other.md#sab" in leaky.flags[0]
    )


# ---- the whole plan ----------------------------------------------------------------------------
def groups() -> dict[str, dict[str, list[C.Passage]]]:
    def make(prefix: str, corpus: str, tenant: str, pages: int) -> dict[str, list[C.Passage]]:
        return {
            f"{prefix}/p{i}.md": [passage(f"{prefix}/p{i}.md", j, corpus, tenant) for j in range(3)]
            for i in range(pages)
        }

    return {
        "gitlab": make("gitlab", "gitlab", "A", 80),
        "posthog": make("posthog", "posthog", "B", 40),
        "pdf": make("pdf", "irs-p15b", "A", 30),
    }


def test_the_plan_is_80_answerable_plus_15_unanswerable() -> None:
    assert sum(n for (k, _), n in PLAN.items() if k != "unanswerable") == 80
    assert sum(n for (k, _), n in PLAN.items() if k == "unanswerable") == 15


def test_plan_is_reproducible_and_uses_each_page_once_per_group() -> None:
    first, second = plan_work(groups()), plan_work(groups())
    assert [(k, g, [p.doc for p in ps]) for k, g, ps in first] == [
        (k, g, [p.doc for p in ps]) for k, g, ps in second
    ]
    assert len(first) == 95
    for group in ("gitlab", "posthog"):  # wiki pages: at most one question per page
        pages = [ps[0].doc for _, g, ps in first if g == group]
        assert len(pages) == len(set(pages))
    pdf_pages = [(ps[0].doc, ps[0].locator) for _, g, ps in first if g == "pdf"]
    assert len(pdf_pages) == len(set(pdf_pages))  # PDFs: at most one question per PDF page


def test_run_draft_assigns_stable_ids_and_survives_failures() -> None:
    calls = {"n": 0}

    def sometimes_down(messages, **kw):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] % 10 == 0:
            raise LLMError("provider down")
        return fake_model(messages, **kw)

    items, failures = run_draft(groups(), sometimes_down, workers=1)
    assert 0 < len(failures) < 95 and len(items) + len(failures) == 95
    ids = [i.id for i in items]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) and ids[0].startswith("a-")
    assert all(EvalItem.model_validate_json(i.model_dump_json()) == i for i in items)  # round-trips
    assert sum(1 for i in items if not i.answerable) <= 15 and all(not i.reviewed for i in items)


def test_a_few_pdfs_still_supply_all_their_questions_one_per_page() -> None:
    pdf_pages = {  # 3 PDFs, many pages each (as in the real corpus)
        f"doc{d}.pdf": [passage(f"doc{d}.pdf", n, "irs-p15b") for n in range(1, 25)]
        for d in range(3)
    }
    work = [w for w in plan_work({"pdf": pdf_pages}) if w[1] == "pdf"]
    assert len(work) == 16  # 12 single + 4 unanswerable
    pages = [(ps[0].doc, ps[0].locator) for _, _, ps in work]
    assert len(pages) == len(set(pages))  # every question from a different page
    unanswerable = [ps for kind, _, ps in work if kind == "unanswerable"]
    assert all(
        len(ps) == 2 and ps[0].doc == ps[1].doc and ps[0].locator != ps[1].locator
        for ps in unanswerable
    )


def test_fill_mode_drafts_only_what_is_missing() -> None:
    calls = {"n": 0}

    def counting(messages, **kw):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return fake_model(messages, **kw)

    first, _ = run_draft(groups(), counting, workers=1)
    assert len(first) == 95
    calls["n"] = 0
    again, failures = run_draft(groups(), counting, workers=1, existing=first)
    assert again == [] and failures == [] and calls["n"] == 0  # nothing missing, nothing spent

    missing = [i for i in first if i.kind == "single" and i.tenant == "B"][:3]
    partial = [i for i in first if i not in missing]
    new, _ = run_draft(groups(), counting, workers=1, existing=partial)
    assert len(new) == 3 and all(n.kind == "single" and n.tenant == "B" for n in new)
    assert not {n.id for n in new} & {p.id for p in partial}  # the new ids do not collide
    assert {tuple(g.locator for g in n.gold) for n in new} == {
        tuple(g.locator for g in m.gold) for m in missing
    }


def test_matching_ignores_curly_quotes_and_dashes() -> None:
    passage_text = (
        "Expenses incurred before you establish your HSA aren’t qualified – see the rules."
    )
    assert C.contains_verbatim(
        passage_text,
        "Expenses incurred before you establish your HSA aren't qualified - see the rules",
    )
