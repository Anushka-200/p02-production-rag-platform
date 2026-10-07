"""One question of the evaluation set (a line of evals/qa.jsonl)."""

from typing import Literal

from pydantic import BaseModel, Field


class GoldRef(BaseModel):
    doc: str  # source_uri, e.g. "gitlab/total-rewards/benefits/_index.md"
    locator: str  # a heading anchor ("time-off") or "page:6" for PDFs


class EvalItem(BaseModel):
    id: str
    tenant: Literal["A", "B"]  # which organisation's documents the question is about
    kind: Literal["single", "multi_hop", "follow_up", "unanswerable", "paraphrase", "broad"]
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)  # earlier turns, for follow-ups
    answerable: bool
    gold: list[GoldRef] = Field(default_factory=list)  # where the answer is (empty if unanswerable)
    gold_answer: str = ""
    support: list[str] = Field(default_factory=list)  # sentences copied verbatim from the passage
    source: list[str] = Field(default_factory=list)  # passages the draft was made from: doc#locator
    corpus: str  # gitlab | posthog | irs-p15b | irs-p969 | opm
    tags: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)  # automatic warnings for the reviewer
    reviewed: bool = False  # set to true by the human reviewer (P01's eval never recorded this)
    review_notes: str = ""
    drafted_by: str = ""
