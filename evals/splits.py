"""Dev / test split of the eval set, fixed by a hash of the question id (so it never changes when questions are added).

The rule that keeps tuning honest: choose settings, prompts and models on the DEV questions only. The TEST questions are looked at
ONCE, at the end, to report the final number. Anything tuned while looking at test is no longer a test.
A paraphrase always goes with its base question (same split), so a tuned setting cannot "see" a question through its twin.
"""

import hashlib

TEST_SHARE = 0.30


def base_id(item_id: str) -> str:
    """p-a-single-005 -> a-single-005 (a paraphrase shares its split with the question it was made from)."""
    return item_id[2:] if item_id.startswith("p-") else item_id


def split_of(item_id: str) -> str:
    h = int(hashlib.sha1(base_id(item_id).encode()).hexdigest(), 16) % 1000 / 1000
    return "test" if h < TEST_SHARE else "dev"
