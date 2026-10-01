import os
from types import SimpleNamespace

import pytest
from litellm import exceptions as exc

from app import embeddings
from app.config import Settings
from app.embeddings import (
    EmbeddingError,
    LiteLLMEmbedder,
    get_embedder,
    model_token_counter,
    vector_literal,
)


def fake_response(vectors: list[list[float]], tokens: int = 10) -> SimpleNamespace:
    # items deliberately out of order: the embedder must sort them by "index"
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)][::-1]
    return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=tokens))


class Script:
    """A fake provider: raises or returns scripted outcomes in order, recording each input."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes, self.inputs = list(outcomes), []

    def __call__(self, model: str, input: list[str]) -> object:  # noqa: A002
        self.inputs.append(list(input))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def embedder(script: Script, **kw: object) -> LiteLLMEmbedder:
    return LiteLLMEmbedder(
        "fake/model",
        2,
        embedding_fn=script,
        cost_fn=lambda **_: 0.001,
        sleep_fn=lambda _: None,
        **kw,
    )


def test_vector_literal_is_postgres_text_form() -> None:
    assert vector_literal([0.5, -1, 2.25]) == "[0.5,-1.0,2.25]"


def test_texts_are_sent_in_batches_and_order_is_preserved() -> None:
    script = Script(
        fake_response([[1.0, 0.0], [2.0, 0.0]]),
        fake_response([[3.0, 0.0], [4.0, 0.0]]),
        fake_response([[5.0, 0.0]]),
    )
    out = embedder(script, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])
    assert script.inputs == [["a", "b"], ["c", "d"], ["e"]]  # 5 texts -> 3 calls
    assert [v[0] for v in out] == [1.0, 2.0, 3.0, 4.0, 5.0]  # even though items came back reversed


def test_cost_and_tokens_accumulate() -> None:
    e = embedder(
        Script(fake_response([[1.0, 0.0]], 7), fake_response([[2.0, 0.0]], 5)), batch_size=1
    )
    e.embed_documents(["a", "b"])
    assert e.total_tokens == 12 and e.total_cost_usd == pytest.approx(0.002) and e.cost_known


def test_unknown_price_is_flagged_not_counted_as_zero() -> None:
    def no_price(**_: object) -> float:
        raise ValueError("unknown model")

    e = LiteLLMEmbedder("m", 2, embedding_fn=Script(fake_response([[1.0, 0.0]])), cost_fn=no_price)
    e.embed_query("x")
    assert e.cost_known is False


def test_temporary_errors_are_retried_then_succeed() -> None:
    limited = exc.RateLimitError("slow down", "p", "m")
    script = Script(limited, limited, fake_response([[1.0, 0.0]]))
    assert embedder(script).embed_query("x") == [1.0, 0.0] and len(script.inputs) == 3


def test_gives_up_after_max_retries() -> None:
    limited = exc.RateLimitError("slow down", "p", "m")
    script = Script(*[limited for _ in range(4)])
    with pytest.raises(EmbeddingError, match="failed after 4 attempts"):
        embedder(script).embed_query("x")


def test_bad_key_fails_immediately() -> None:
    script = Script(exc.AuthenticationError("bad key", "p", "m"))
    with pytest.raises(EmbeddingError, match="AuthenticationError"):
        embedder(script).embed_query("x")
    assert len(script.inputs) == 1


def test_get_embedder_follows_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    made: dict[str, tuple] = {}  # type: ignore[type-arg]
    monkeypatch.setattr(
        embeddings, "LocalEmbedder", lambda *a: made.setdefault("local", a) or SimpleNamespace()
    )
    get_embedder(Settings(_env_file=None, embedding_provider="local"))  # type: ignore[call-arg]
    assert made["local"] == ("BAAI/bge-small-en-v1.5", 384, 16)
    hosted = get_embedder(
        Settings(_env_file=None, embedding_provider="litellm", embedding_model="x/y")  # type: ignore[call-arg]
    )
    assert isinstance(hosted, LiteLLMEmbedder) and hosted.model == "x/y"


# ---- the real local model (downloads ~70 MB once; skipped on CI and when unavailable) ---------
@pytest.fixture(scope="module")
def local():  # type: ignore[no-untyped-def]
    if os.environ.get("CI"):
        pytest.skip("real model tests are skipped on CI")
    try:
        return embeddings.LocalEmbedder()
    except Exception as e:  # no network, no disk space...
        pytest.skip(f"local model unavailable: {type(e).__name__}")


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # vectors are unit length


def test_local_vectors_have_the_right_size_and_unit_length(local) -> None:  # type: ignore[no-untyped-def]
    [v] = local.embed_documents(["Employees receive twenty vacation days."])
    assert len(v) == local.dimension == 384
    assert sum(x * x for x in v) == pytest.approx(1.0, abs=1e-3)


def test_local_similar_meaning_is_closer_than_unrelated(local) -> None:  # type: ignore[no-untyped-def]
    cat, rug, revenue = local.embed_documents(
        ["The cat sat on the mat.", "A kitten is resting on a rug.", "Quarterly revenue grew 12%."]
    )
    assert cosine(cat, rug) > cosine(cat, revenue) + 0.1


def test_local_batching_does_not_change_the_result(local) -> None:  # type: ignore[no-untyped-def]
    texts = [f"Policy number {i} covers topic {i * 7}." for i in range(5)]
    together = local.embed_documents(texts)
    one_by_one = [local.embed_documents([t])[0] for t in texts]
    for a, b in zip(together, one_by_one, strict=True):
        assert cosine(a, b) > 0.999


def test_local_empty_input_returns_empty(local) -> None:  # type: ignore[no-untyped-def]
    assert local.embed_documents([]) == []


def test_model_token_counter_matches_the_model_not_a_guess() -> None:
    try:
        count = model_token_counter("BAAI/bge-small-en-v1.5")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {type(e).__name__}")
    table_row = "| Meal | 50 |"
    assert count(table_row) > len(table_row) / 4 + 2  # tables need more tokens than "chars / 4"
    assert count("word " * 700) > 512  # not silently capped at the model's maximum
