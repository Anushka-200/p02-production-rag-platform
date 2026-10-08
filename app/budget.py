"""Per-organisation daily spend budget: check before answering, record after, alert once per level.

    spent < alert fraction of the budget   normal
    spent >= alert fraction                a WARNING is logged and shown in the admin console
    spent >= the budget                    new fresh answers are refused (HTTP 429) until tomorrow

Cache hits cost nothing and are recorded as requests only. A request already running when the limit
is crossed still finishes, so the real overshoot is at most (parallel requests x one answer's cost).
The cap is per organisation, so one tenant running out never affects another.
"""

import logging
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import Settings
from app.retrieval.search import Scope

logger = logging.getLogger(__name__)

OK, WARNING, EXHAUSTED = 0, 1, 2


@dataclass
class BudgetStatus:
    spent_usd: float
    limit_usd: float | None  # None = unlimited
    level: int  # OK | WARNING | EXHAUSTED

    @property
    def exhausted(self) -> bool:
        return self.level >= EXHAUSTED


def limit_for(conn: psycopg.Connection, scope: Scope, s: Settings) -> float | None:
    """The org's own cap if it has one, else the server default; 0 means unlimited."""
    row = conn.execute(
        "SELECT daily_budget_usd FROM orgs WHERE id = %s", (scope.org_id,)
    ).fetchone()
    limit = float(row[0]) if row and row[0] is not None else s.org_daily_budget_usd
    return limit if limit > 0 else None


def _level(spent: float, limit: float | None, s: Settings) -> int:
    if limit is None:
        return OK
    if spent >= limit:
        return EXHAUSTED
    return WARNING if spent >= s.budget_alert_fraction * limit else OK


def status(conn: psycopg.Connection, scope: Scope, s: Settings) -> BudgetStatus:
    row = conn.execute(
        "SELECT cost_usd FROM org_usage WHERE org_id = %s AND day = current_date", (scope.org_id,)
    ).fetchone()
    spent = float(row[0]) if row else 0.0
    limit = limit_for(conn, scope, s)
    return BudgetStatus(spent, limit, _level(spent, limit, s))


def record(
    conn: psycopg.Connection, scope: Scope, s: Settings, *, cost_usd: float, input_tokens: int = 0,
    output_tokens: int = 0,
) -> BudgetStatus:  # fmt: skip
    """Add one answer's cost to today's total (needs a WRITE connection). Fires each alert once."""
    row = conn.execute(
        "INSERT INTO org_usage (org_id, requests, input_tokens, output_tokens, cost_usd) "
        "VALUES (%s, 1, %s, %s, %s) ON CONFLICT (org_id, day) DO UPDATE SET "
        "requests = org_usage.requests + 1, "
        "input_tokens = org_usage.input_tokens + EXCLUDED.input_tokens, "
        "output_tokens = org_usage.output_tokens + EXCLUDED.output_tokens, "
        "cost_usd = org_usage.cost_usd + EXCLUDED.cost_usd "
        "RETURNING cost_usd, alert_level",
        (scope.org_id, input_tokens, output_tokens, round(cost_usd, 6)),
    ).fetchone()
    assert row is not None
    spent, already = float(row[0]), int(row[1])
    limit = limit_for(conn, scope, s)
    level = _level(spent, limit, s)
    if level > already:  # crossed a new line today: remember it, say it once
        conn.execute(
            "UPDATE org_usage SET alert_level = %s WHERE org_id = %s AND day = current_date",
            (level, scope.org_id),
        )
        log = logger.error if level == EXHAUSTED else logger.warning
        log(
            "org budget exhausted" if level == EXHAUSTED else "org budget alert",
            extra={"org_id": str(scope.org_id), "spent_usd": round(spent, 4), "limit_usd": limit},
        )
    return BudgetStatus(spent, limit, level)


def usage_report(
    conn: psycopg.Connection, scope: Scope, s: Settings, days: int = 7
) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT day, requests, input_tokens, output_tokens, cost_usd::float AS cost_usd, "
            "alert_level FROM org_usage WHERE org_id = %s ORDER BY day DESC LIMIT %s",
            (scope.org_id, days),
        ).fetchall()
    st = status(conn, scope, s)
    return {
        "today": {"spent_usd": st.spent_usd, "limit_usd": st.limit_usd, "level": st.level},
        "days": rows,
    }


def run_cost(result: Any) -> float:
    """What one answer really cost: the answer call plus query rewriting/expansion and the agent."""
    cost = float(result.answer.cost_usd)
    cost += float(getattr(result.retrieval, "transform_cost_usd", 0.0) or 0.0)
    if result.agent is not None:
        cost += float(result.agent.cost_usd)
    return cost
