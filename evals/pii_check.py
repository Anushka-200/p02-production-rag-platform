"""Hit-rate check for the PII redactor (task 3.11): how many planted personal-data items are caught,
and how many harmless sentences are wrongly changed?

Run: `uv run python -m evals.pii_check`   (labelled cases: evals/pii_cases.jsonl, written by hand)
A case is "caught" when every entity type listed for it was found; a benign case is a "false alarm"
when anything at all is redacted. Informal: 51 hand-written cases, not a benchmark.
"""

import json
from collections import Counter
from pathlib import Path

from app.pii import redact

CASES = Path(__file__).parent / "pii_cases.jsonl"
# Phone numbers are matched loosely by Presidio; a number found as a phone when we labelled it an SSN
# (or the reverse) is still redacted, which is what matters, so these count as equivalent.
EQUIVALENT = {"US_SSN": {"US_SSN", "PHONE_NUMBER"}}


def main() -> None:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    by_type_total: Counter[str] = Counter()
    by_type_hit: Counter[str] = Counter()
    benign = [c for c in cases if not c["pii"]]
    positive = [c for c in cases if c["pii"]]
    missed: list[str] = []
    for case in positive:
        found = set(redact(case["text"]).found)
        for kind in case["pii"]:
            by_type_total[kind] += 1
            if found & EQUIVALENT.get(kind, {kind}):
                by_type_hit[kind] += 1
            else:
                missed.append(f"{kind}: {case['text']}")
    false_alarms = [c["text"] for c in benign if redact(c["text"]).found]
    items = sum(by_type_total.values())
    hits = sum(by_type_hit.values())
    print(f"{'entity':<16} caught")
    for kind in sorted(by_type_total):
        print(f"{kind:<16} {by_type_hit[kind]}/{by_type_total[kind]}")
    print(f"\nrecall           {hits}/{items} = {hits / items:.0%}")
    print(f"false alarms     {len(false_alarms)}/{len(benign)} harmless sentences changed")
    for line in missed:
        print("  MISSED   ", line)
    for line in false_alarms:
        print("  FALSE    ", line)


if __name__ == "__main__":
    main()
