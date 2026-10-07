"""Query rewriting through the chat API and the inspector: history is read, the rewrite is used."""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from tests.test_api import FakeStream, SharedConnection, read_events
from tests.test_expand import STANDALONE, TextEmbedder
from tests.test_search import World, vec

ANN = {"X-Dev-User": "ann@x.com"}


class Rig:
    def __init__(self, db_schema, **settings):  # type: ignore[no-untyped-def]
        w = World(db_schema)
        ann = w.person("acme", "ann")
        w.conn.execute("UPDATE users SET role = 'admin' WHERE id = %s", (ann.user_id,))
        w.chunk(ann, "acme", "Team workshops cover twelve topics.", vec(1), uri="workshops.md")
        self.prompts: list = []  # type: ignore[type-arg]
        self.model_calls: list[str] = []

        def complete_fn(messages, **kw):  # type: ignore[no-untyped-def]
            self.model_calls.append(messages[-1]["content"])
            return SimpleNamespace(text=STANDALONE, cost_usd=0.001)

        def stream_fn(messages, **kw):  # type: ignore[no-untyped-def]
            self.prompts.append(messages)
            return FakeStream(["Twelve [S1]."])

        cfg = Settings(
            _env_file=None,
            environment="dev",
            reranker_provider="none",
            retrieval_candidates=10,
            **settings,
        )  # type: ignore[call-arg]
        self.client = TestClient(
            create_app(
                cfg,
                connect=lambda: SharedConnection(db_schema),
                embedder=TextEmbedder(),
                reranker=None,
                stream_fn=stream_fn,
                complete_fn=complete_fn,
            )  # fmt: skip
        )

    def chat(self, question: str, conversation_id: str | None = None) -> dict:  # type: ignore[type-arg]
        body = {
            "question": question,
            **({"conversation_id": conversation_id} if conversation_id else {}),
        }
        r = self.client.post("/chat", json=body, headers=ANN)
        return next(d for n, d in read_events(r) if n == "done")


def test_a_follow_up_in_a_conversation_is_rewritten_and_the_answer_is_found(db_schema) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(db_schema, query_rewrite=True)
    first = rig.chat("What training does the company offer?")
    assert rig.model_calls == []  # a brand-new conversation has no history: no rewrite call
    rig.chat("How many topics do these sessions cover?", first["conversation_id"])
    assert (
        len(rig.model_calls) == 1 and "What training does the company offer?" in rig.model_calls[0]
    )
    final_prompt = rig.prompts[-1][-1]["content"]
    assert f"Question: {STANDALONE}" in final_prompt  # the answer is about the standalone question
    assert "Team workshops cover twelve topics." in final_prompt  # the search used the rewrite


def test_with_rewriting_off_history_is_not_even_read_and_nothing_changes(db_schema) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(db_schema)
    first = rig.chat("What training does the company offer?")
    rig.chat("How many topics do these sessions cover?", first["conversation_id"])
    assert rig.model_calls == []
    assert "Question: How many topics do these sessions cover?" in rig.prompts[-1][-1]["content"]


def test_the_inspector_shows_what_the_transforms_did(db_schema) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(db_schema, query_rewrite=True)
    history = [
        {"role": "user", "content": "What training?"},
        {"role": "assistant", "content": "Team workshops."},
    ]
    r = rig.client.post(
        "/admin/inspect",
        json={
            "question": "How many topics do these sessions cover?",
            "history": history,
            "mode": "dense",
        },
        headers=ANN,
    )
    body = r.json()
    assert (
        r.status_code == 200
        and body["standalone_question"] == STANDALONE
        and body["queries"] == [STANDALONE]
    )
    assert body["candidates"][0]["document"].startswith("workshops.md")
    assert json.dumps(body["transform_cost_usd"]) == "0.001"


def test_the_inspector_rejects_an_oversized_history(db_schema) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(db_schema, query_rewrite=True)
    long_history = [{"role": "user", "content": "x"}] * 13
    r = rig.client.post(
        "/admin/inspect", json={"question": "q?", "history": long_history}, headers=ANN
    )
    assert r.status_code == 422
