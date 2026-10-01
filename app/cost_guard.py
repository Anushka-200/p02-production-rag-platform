"""Cost guard for scripts that call paid models: project first, confirm, then stop on overrun.

Two protections (OWASP "unbounded consumption" at script level):
1. BEFORE: estimate the bill; above a threshold the script must be confirmed (--yes or a typed y).
2. DURING: add up the real cost of every call; stop if it passes a hard limit.
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import litellm

logger = logging.getLogger(__name__)

CONFIRM_ABOVE_USD = 0.50


class SpendDeclined(RuntimeError):
    """The user did not confirm the projected spend."""


class BudgetExceeded(RuntimeError):
    """Real spend passed the hard limit; the script stopped itself."""


def _price_info(model: str) -> dict[str, float] | None:
    return litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/")[-1])


def estimate_llm_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    calls: int = 1,
    cached_input_tokens: int = 0,
) -> float | None:
    """Projected dollars for `calls` calls of this size, or None if the price is unknown.

    `cached_input_tokens` is the part of each call's input served from the prompt cache (billed
    at the cheaper cache-read rate when the model has one; otherwise treated as normal input).
    Cache-write surcharges are ignored: small next to thousands of cache reads.
    """
    info = _price_info(model)
    if not info or "input_cost_per_token" not in info or "output_cost_per_token" not in info:
        return None
    price_in, price_out = info["input_cost_per_token"], info["output_cost_per_token"]
    price_cached = info.get("cache_read_input_token_cost", price_in)
    fresh = input_tokens - cached_input_tokens
    return calls * (
        fresh * price_in + cached_input_tokens * price_cached + output_tokens * price_out
    )


@dataclass
class CostGuard:
    label: str
    projected_usd: float | None  # None = price unknown
    max_usd: float | None = None  # hard stop; default 2x the projection (at least $0.05)
    confirm_above_usd: float = CONFIRM_ABOVE_USD
    assume_yes: bool = False
    ask: Callable[[str], str] = input  # injected so tests do not wait for a keyboard
    say: Callable[[str], None] = print
    spent_usd: float = 0.0
    unknown_cost_calls: int = 0
    _warned: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_usd is None and self.projected_usd is not None:
            self.max_usd = max(2 * self.projected_usd, 0.05)

    def confirm(self) -> None:
        """Print the projection; above the threshold (or if unknown) require a yes."""
        if self.projected_usd is None:
            self.say(f"[cost guard] {self.label}: projected cost UNKNOWN (no price for the model).")
        else:
            limit = f", hard stop at ${self.max_usd:.2f}" if self.max_usd else ""
            self.say(f"[cost guard] {self.label}: projected ${self.projected_usd:.2f}{limit}.")
            if self.projected_usd <= self.confirm_above_usd:
                return
        if self.assume_yes:
            return
        try:
            answer = self.ask(
                f"Projected spend is above ${self.confirm_above_usd:.2f}. Proceed? [y/N] "
            )
        except EOFError:  # no keyboard (CI, pipes): refuse rather than spend silently
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            raise SpendDeclined(f"{self.label}: not confirmed (re-run with --yes to approve)")

    def record(self, cost_usd: float | None) -> None:
        """Add one call's real cost. Raises BudgetExceeded once the hard limit is passed."""
        if cost_usd is None:
            self.unknown_cost_calls += 1
            return
        self.spent_usd += cost_usd
        if self.max_usd is not None and self.spent_usd > self.max_usd:
            raise BudgetExceeded(
                f"{self.label}: spent ${self.spent_usd:.2f}, over the ${self.max_usd:.2f} limit "
                f"(projected ${self.projected_usd or 0:.2f}). Stopped."
            )

    def summary(self) -> str:
        unknown = (
            f", {self.unknown_cost_calls} calls with unknown price"
            if self.unknown_cost_calls
            else ""
        )
        proj = (
            f"projected ${self.projected_usd:.2f}"
            if self.projected_usd is not None
            else "no projection"
        )
        return f"[cost guard] {self.label}: spent ${self.spent_usd:.4f} ({proj}){unknown}"


def add_cost_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yes", action="store_true", help="approve the projected spend")
    parser.add_argument("--max-usd", type=float, default=None, help="hard stop in dollars")


def guard_from_args(args: argparse.Namespace, label: str, projected_usd: float | None) -> CostGuard:
    return CostGuard(label, projected_usd, max_usd=args.max_usd, assume_yes=args.yes)
