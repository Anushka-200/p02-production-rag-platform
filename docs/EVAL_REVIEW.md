# Eval set review (2026-10-09)

**Who reviewed:** Claude (the model), at the owner's request, because the owner had not read the source documents and could not judge
the gold answers. This is a **model review against the source text, not a human review**: items carry `reviewed: true,
reviewer: "claude"`. It removes the flaws that can be checked against the text (below); it cannot remove a shared blind spot of the
model that drafted the questions and the model that reviewed them. A human spot-check of a random 10% is still worth doing.

**Method (`evals/review_pack.py`, `evals/review_decisions.jsonl`, `evals/apply_review.py`):** for each of the 150 questions a review card
shows the question, the gold answer, the gold passage text, an automatic check that the supporting quotes appear in the gold passages,
and the best passages the production search finds besides the gold (candidate second answers). For doubtful items the corpus was
searched directly. Each decision is keep, fix or drop; every gold place and every added alternative was then verified to match a real
chunk (heading anchor, or page range for PDFs). Originals are kept: `evals/results/qa-before-review.jsonl`, dropped items in
`evals/qa_dropped.jsonl`, the original text of every changed field in each item's `review_notes`.

**Result: 115 kept as they were, 26 fixed, 9 dropped (23% of the drafted questions had a real flaw).**

| flaw found | count (approx.) | example |
|---|---|---|
| gold answer says something the source does not | 5 | a guess about retailers' profit margins; "direct reports felt unheard" |
| question ambiguous (no year, country, person) | 8 | "this year" with 2025 and 2026 limits in the text; "this person"; a baby benefit with no country |
| question misreads the source | 2 | "65 countries" is company-wide, not "where France employees work" |
| gold place wrong or only half needed | 11 | a graph-derived question whose gold passages never mention the field it asks about |
| labelled unanswerable but answerable | 5 | "minimum annual deductible for an HDHP": the table says $1,650 / $3,300 |
| bogus (stitched from unrelated passages) | 2 | contacting billing to connect a payment processor |
| second valid answer place not accepted | 6 | schema-change section also says unused columns are annotated |

Two things that looked like errors were not: the IRS PDF gold pages (my own review card showed the wrong section of a page, so the
question was kept), and several "competing" passages that were about a different topic.

**What changed in the system:** a gold place may now list accepted alternatives (`alt`); finding the place or any alternative counts
(`evals/metrics.py::covers`, `evals/run.py::gold_in_context`). Tests in `tests/test_metrics.py`.

**What it means for earlier numbers:** all retrieval and answer numbers in `docs/benchmarks.md` up to B-035 were measured on the
UNREVIEWED set. They are not comparable with numbers measured after this review: re-measure before comparing (see B-036).
