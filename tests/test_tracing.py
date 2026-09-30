from app import tracing
from app.llm import complete
from tests.test_llm import SETTINGS, Script, fake_response


def test_no_client_means_observation_is_a_noop() -> None:
    with tracing.observation(as_type="span", name="x") as obs:
        assert obs is None
    tracing.flush()  # must not raise


def test_complete_records_generation_with_usage_and_cost(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    class FakeObs:
        def update(self, **kw: object) -> None:
            seen["update"] = kw

    class FakeCtx:
        def __enter__(self) -> FakeObs:
            return FakeObs()

        def __exit__(self, *a: object) -> None:
            return None

    def fake_observation(**kw: object) -> FakeCtx:
        seen["start"] = kw
        return FakeCtx()

    monkeypatch.setattr("app.llm.observation", fake_observation)
    complete(
        [{"role": "user", "content": "x"}],
        name="unit-test",
        settings=SETTINGS,
        completion_fn=Script(fake_response("hey")),
        cost_fn=lambda **_: 0.002,
    )
    assert seen["start"]["as_type"] == "generation"  # type: ignore[index]
    assert seen["update"]["usage_details"] == {"input": 10, "output": 5}  # type: ignore[index]
    assert seen["update"]["cost_details"] == {"total": 0.002}  # type: ignore[index]
