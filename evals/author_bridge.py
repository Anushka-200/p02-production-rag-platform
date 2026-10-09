"""Hand-authored multi-hop questions (written by the reviewer, 2026-10-09), each tied to two linked pages.

Why authored: the link-based generator produced only 5 usable bridge questions out of 44 (the second page alone usually
answered). Here each question gives only the situation from page A and needs a fact that is on page B; the supporting quote of
BOTH pages must appear verbatim in its passage and both gold places must exist. The script checks that and then appends
the items (reviewed, reviewer=claude, authored=true in review_notes) to evals/qa.jsonl. Idempotent by id.

Run: uv run python -m evals.author_bridge [--dry-run]
"""

import argparse
import json
from pathlib import Path

import psycopg

from app.config import get_settings
from evals import corpus as C
from evals.apply_review import place_exists
from evals.review_pack import passage_index
from evals.schema import EvalItem

G, P = "gitlab/", "posthog/"
# (id, tenant, question, gold answer, (A doc, A locator, A quote), (B doc, B locator, B quote))
ITEMS = [
    (
        "n-m-multi_hop-001",
        "A",
        "If I joined GitLab on October 20th, will I be considered for a stock refresh grant in the annual compensation review?",
        "Probably not: refresh grants are reviewed through the Annual Compensation Review, and to take part in the equity refresh program your hire date must be on or before October 5th.",
        (
            G + "total-rewards/stock-options.md",
            "refresh-grants",
            "eligible GitLab team members will be reviewed for a refresh grant once per year",
        ),
        (
            G + "total-rewards/compensation/compensation-review-cycle/_index.md",
            "eligibility",
            "October 5th to be eligible to participate in the equity refresh program",
        ),
    ),
    (
        "n-m-multi_hop-002",
        "A",
        "My direct report's interim assignment just ended. What do I need to do about their bonus, and where do I confirm when the interim period started?",
        "Review the criteria for eligibility and submit an interim bonus request in Workday. For an interim bonus, confirm the start date in the Workday Job tab under Job History (the Business Title column) and give the start and end dates of the interim period.",
        (
            G + "people-group/promotions-transfers.md",
            "ending-interim-period",
            "the manager should review the Criteria For Eligibility for the Interim Bonus and submit an interim bonus request for the team member",
        ),
        (
            G + "total-rewards/incentives.md",
            "submitting",
            "The manager references the Workday Job tab and clicks into Job History and review the 'Business Title' column to confirm the start date of the interim period",
        ),
    ),
    (
        "n-m-multi_hop-003",
        "A",
        "If I'm worried I might do something unsafe and People Operations points me to a support benefit, when do my sessions reset each year and do unused ones carry over?",
        "The benefit (Modern Health) resets annually on March 20, and unused sessions do not roll over.",
        (
            G + "people-policies/_index.md",
            "responding-to-unsafe-situations",
            "People Operations can assist in providing information about available Modern Health options",
        ),
        (
            G + "total-rewards/benefits/modern-health.md",
            "important-details",
            "Benefits reset annually on March 20",
        ),
    ),
    (
        "n-m-multi_hop-004",
        "A",
        "Which leadership program at GitLab is meant for Managers and Senior Managers, and how many learning modules does it have?",
        "Elevate is the program for Managers and Senior Managers; it is organized into 6 learning modules, including 2 practice labs.",
        (
            G + "people-group/learning-and-development/elevate-programs/_index.md",
            "elevate-programs",
            "For Managers and Senior Managers.",
        ),
        (
            G + "people-group/learning-and-development/elevate-programs/elevate/_index.md",
            "how-is-the-program-structured",
            "Elevate is organized into 6 learning modules, including 2 practice labs.",
        ),
    ),
    (
        "n-m-multi_hop-005",
        "A",
        "Before I take part in the 360 feedback program at GitLab, what should I review, and what does 360 feedback add to the conversation between a manager and a team member?",
        "Review your Individual Growth Plan and ask for specific feedback on your development areas. 360 feedback enriches the manager-team member dialogue with insights from peers, stakeholders and direct reports, surfaces blind spots, supports targeted development and makes career conversations more meaningful.",
        (
            G + "people-group/learning-and-development/career-development/igp-guide.md",
            "bringing-your-plan-to-life-track-reflect-and-adapt",
            "Before participating in the 360 feedback program, review your IGP and ask for specific feedback on your development areas.",
        ),
        (
            G + "people-group/360-feedback.md",
            "why-360-feedback-matters",
            "360 feedback enriches the dialogue between managers and team members by",
        ),
    ),
    (
        "n-m-multi_hop-006",
        "B",
        "When I want to introduce a new software tool at PostHog, how big does the upside have to be, and how much can I spend per month on subscriptions in my own name?",
        "There needs to be a very significant upside to outweigh the cost of a new piece of software. Each person gets a User Limit of $5,000 per month in Brex for individual subscriptions and similar spending; it is a ceiling, not a target.",
        (
            P + "company/adding-tools.md",
            "what-is-it",
            "There needs to be a very significant upside to introducing a new piece of software to outweigh its cost.",
        ),
        (
            P + "people/spending-money.md",
            "how-it-works",
            "You'll be assigned a single User Limit of $5,000 per month in Brex",
        ),
    ),
    (
        "n-m-multi_hop-007",
        "B",
        "When expanding into a second team at an existing customer, what does the sales playbook suggest offering, and what is the format and group size of that training?",
        "Consider offering a training session to get the new team up to speed. The format is two live group sessions for up to 30 people across product, marketing and engineering.",
        (
            P + "growth/sales/expansion-strategies.md",
            "how-to-execute-2",
            "consider offering a training session to get the new team up to speed",
        ),
        (
            P + "growth/sales/customer-training.md",
            "running-product-training-sessions",
            "Two live group sessions for up to 30 people across product, marketing, and engineering.",
        ),
    ),
    (
        "n-m-multi_hop-008",
        "B",
        "When a product team moves a feature to beta at PostHog, who writes the documentation, and what should those docs help people do?",
        "Product teams are responsible for writing the documentation. Docs should help people get started with the feature, understand what it does (a complete list of features and details), and make the most of it through use cases, concepts and answers to common questions.",
        (
            P + "product/releasing-new-products-and-features.md",
            "moving-a-feature-to-beta",
            "Product teams are responsible for writing documentation",
        ),
        (
            P + "engineering/writing-docs.md",
            "what-should-i-write-docs-about",
            "Get started with your product or feature.",
        ),
    ),
    (
        "n-m-multi_hop-009",
        "B",
        "Which PostHog AI capability analyses recordings in batches to find patterns, and in what ways can it be triggered?",
        "Session summaries. It can be triggered by asking PostHog AI directly, from the Session Replay UI or other products, or by Deep research as part of a larger investigation.",
        (
            P + "engineering/ai/ai-platform.md",
            "1-user-facing-products",
            "Batch analysis of session recordings to find patterns",
        ),
        (
            P + "engineering/ai/products.md",
            "how-it-works-1",
            "You can trigger Session summaries in three ways",
        ),
    ),
    (
        "n-m-multi_hop-010",
        "B",
        "How many ICP signups does PostHog get each week, and how large does a signup's potential need to be before someone from sales reaches out?",
        "Hundreds of ICP signups every week; a salesperson should reach out when they think the lead could end up paying more than $20k a year.",
        (
            P + "growth/sales/overview.md",
            "things-we-want-to-be-great-at",
            "We get hundreds of ICP signups to PostHog every week",
        ),
        (
            P + "growth/sales/product-led-sales.md",
            "working-with-the-customer",
            "If you think they have potential to end up paying more than $20k a year then you should reach out to introduce yourself and offer help.",
        ),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    index = passage_index()
    conn = psycopg.connect(get_settings().database_url("owner"), autocommit=True)
    qa = Path("evals/qa.jsonl")
    have = {json.loads(line)["id"] for line in qa.read_text().splitlines()}
    problems, new = [], []
    for id_, tenant, question, answer, a, b in ITEMS:
        for doc, loc, quote in (a, b):
            p = index.get((doc, f"{loc}"))
            if p is None or not C.contains_verbatim(p.text, quote):
                problems.append(f"{id_}: quote not found in {doc}#{loc}: {quote[:70]!r}")
            if not place_exists(conn, doc, loc):
                problems.append(f"{id_}: no such place {doc}#{loc}")
        item = EvalItem(
            id=id_, tenant=tenant, kind="multi_hop", question=question, answerable=True,  # type: ignore[arg-type]
            gold=[{"doc": a[0], "locator": a[1]}, {"doc": b[0], "locator": b[1]}],  # type: ignore[list-item]
            gold_answer=answer, support=[a[2], b[2]], corpus="gitlab" if tenant == "A" else "posthog",
            tags=["multi_hop", "bridge", "authored"], source=[f"{a[0]}#{a[1]}", f"{b[0]}#{b[1]}"],
            reviewed=True, reviewer="claude", drafted_by="claude (authored during the 2026-10-09 review)",
            review_notes="authored from a real cross-link: page A gives the situation, page B holds the answer; both supporting quotes verified verbatim",
        )  # fmt: skip
        if id_ not in have:
            new.append(item)
    print(f"{len(ITEMS)} authored items, {len(new)} new, {len(problems)} problems")
    for p in problems:
        print("  ", p)
    if problems or args.dry_run:
        raise SystemExit(1 if problems else 0)
    with qa.open("a") as f:
        f.writelines(i.model_dump_json() + "\n" for i in new)


if __name__ == "__main__":
    main()
