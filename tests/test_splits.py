import json
from collections import Counter

from evals.splits import base_id, split_of


def test_a_paraphrase_shares_the_split_of_its_base_question() -> None:
    assert base_id("p-a-single-005") == "a-single-005" and base_id("a-single-005") == "a-single-005"
    assert split_of("p-a-single-005") == split_of("a-single-005")


def test_the_split_is_stable_and_roughly_70_30_on_the_real_eval_set() -> None:
    ids = [json.loads(line)["id"] for line in open("evals/qa.jsonl")]
    counts = Counter(split_of(i) for i in ids)
    assert counts["test"] + counts["dev"] == len(ids)
    assert 0.2 < counts["test"] / len(ids) < 0.4
    assert [split_of(i) for i in ids] == [split_of(i) for i in ids]
