"""Chat history and memory through the HTTP API (fake model and embedder, real database)."""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from tests.test_api import FakeStream, SharedConnection, read_events
from tests.test_memory import AxisEmbedder, fake_llm
from tests.test_search import World, vec

ANN = {"X-Dev-User": "ann@x.com"}
AMY = {"X-Dev-User": "amy@x.com"}
EMBEDDER = AxisEmbedder({"tea": (1, 0.0, 2), "Python": (5, 0.0, 6)})


class Rig:
    def __init__(self, db_schema, *, facts=("User likes tea",), memory=True, extractor=None):  # type: ignore[no-untyped-def]
        self.w = World(db_schema)
        ann = self.w.person("acme", "ann")
        self.w.person("acme", "amy")
        self.w.chunk(ann, "acme", "Employees receive twenty vacation days", vec(30), uri="a.md")
        self.prompts: list[list[dict]] = []  # type: ignore[type-arg]
        self.extractor_calls: list = []  # type: ignore[type-arg]

        def stream_fn(messages, **kw):  # type: ignore[no-untyped-def]
            self.prompts.append(messages)
            return FakeStream(["Twenty days [S1]."])

        class Embed(AxisEmbedder):
            def embed_query(self, text: str) -> list[float]:
                # chunk questions land on the chunk vector; memory topics use their own axes
                return super().embed_query(text) if "tea" in text or "Python" in text else vec(30)

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None, environment="dev", reranker_provider="none", retrieval_candidates=10,
            memory_enabled=memory,
        )  # fmt: skip
        self.client = TestClient(
            create_app(
                settings,
                connect=lambda: SharedConnection(db_schema),
                embedder=Embed(EMBEDDER.axes),
                reranker=None,
                stream_fn=stream_fn,
                complete_fn=lambda *a, **k: pytest.fail("no JSON fallback expected"),
                memory_complete_fn=extractor
                or fake_llm({"facts": list(facts)}, self.extractor_calls),
            )  # fmt: skip
        )

    def ask(self, question: str, who=ANN, conversation_id=None):  # type: ignore[no-untyped-def]
        body = {
            "question": question,
            **({"conversation_id": conversation_id} if conversation_id else {}),
        }
        return self.client.post("/chat", json=body, headers=who)

    def done(self, response) -> dict:  # type: ignore[no-untyped-def,type-arg]
        return next(d for n, d in read_events(response) if n == "done")


@pytest.fixture
def rig(db_schema) -> Rig:  # type: ignore[no-untyped-def]
    return Rig(db_schema)


def test_a_chat_is_stored_as_a_conversation_with_both_messages(rig: Rig) -> None:
    done = rig.done(rig.ask("How many vacation days?"))
    conv = done["conversation_id"]
    messages = rig.client.get(f"/conversations/{conv}", headers=ANN).json()
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "How many vacation days?"),
        ("assistant", "Twenty days [S1]."),
    ]
    assert messages[1]["citations"][0]["id"] == "S1"
    listed = rig.client.get("/conversations", headers=ANN).json()
    assert [c["id"] for c in listed] == [conv] and listed[0]["title"] == "How many vacation days?"


def test_a_follow_up_with_the_conversation_id_appends_to_it(rig: Rig) -> None:
    conv = rig.done(rig.ask("first question"))["conversation_id"]
    again = rig.done(rig.ask("second question", conversation_id=conv))
    assert again["conversation_id"] == conv
    assert len(rig.client.get(f"/conversations/{conv}", headers=ANN).json()) == 4


def test_nobody_else_can_read_continue_or_delete_a_conversation(rig: Rig) -> None:
    conv = rig.done(rig.ask("private question"))["conversation_id"]
    assert rig.client.get(f"/conversations/{conv}", headers=AMY).status_code == 404
    assert rig.ask("hijack", who=AMY, conversation_id=conv).status_code == 404
    assert rig.client.delete(f"/conversations/{conv}", headers=AMY).status_code == 404
    assert rig.client.get("/conversations", headers=AMY).json() == []
    assert rig.client.delete(f"/conversations/{conv}", headers=ANN).status_code == 200
    assert rig.client.get(f"/conversations/{conv}", headers=ANN).status_code == 404


def test_a_null_byte_in_the_question_is_stripped_not_a_server_error(rig: Rig) -> None:
    assert rig.done(rig.ask("vacation\x00 days"))["conversation_id"]


def test_memory_is_learned_after_the_answer_and_can_be_listed_and_deleted(rig: Rig) -> None:
    rig.ask("I really like tea, how many vacation days?")
    memories = rig.client.get("/memories", headers=ANN).json()
    assert [m["fact"] for m in memories] == ["User likes tea"]
    assert rig.client.get("/memories", headers=AMY).json() == []  # private to the user
    assert rig.client.delete(f"/memories/{memories[0]['id']}", headers=AMY).status_code == 404
    assert rig.client.delete(f"/memories/{memories[0]['id']}", headers=ANN).status_code == 200
    assert rig.client.get("/memories", headers=ANN).json() == []


def test_delete_all_memories(rig: Rig) -> None:
    rig.ask("I like tea")
    assert rig.client.delete("/memories", headers=ANN).json() == {"deleted": 1}


def test_a_relevant_memory_reaches_the_next_prompt_as_an_unverified_note(rig: Rig) -> None:
    rig.ask("I like tea")
    rig.prompts.clear()
    rig.ask("what tea policy applies?")
    prompt = rig.prompts[0][-1]["content"]
    assert "<user_notes>" in prompt and "User likes tea" in prompt and "NOT sources" in prompt
    rig.prompts.clear()
    rig.ask("how many vacation days?")  # unrelated: the note is not injected
    assert "<user_notes>" not in rig.prompts[0][-1]["content"]


def test_one_users_memory_never_reaches_another_users_prompt(rig: Rig) -> None:
    rig.ask("I like tea")  # ann's memory
    rig.prompts.clear()
    rig.ask("what tea policy applies?", who=AMY)
    assert "User likes tea" not in json.dumps(rig.prompts)


def test_with_memory_off_nothing_is_extracted_or_stored(db_schema) -> None:  # type: ignore[no-untyped-def]
    off = Rig(db_schema, memory=False)
    off.ask("I like tea")
    assert off.extractor_calls == [] and off.client.get("/memories", headers=ANN).json() == []


def test_a_failing_extractor_never_breaks_the_chat(db_schema) -> None:  # type: ignore[no-untyped-def]
    def explode(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("extractor crashed")

    broken = Rig(db_schema, extractor=explode)
    done = broken.done(broken.ask("hello there"))
    assert done["text"].startswith("Twenty")
    assert broken.client.get("/memories", headers=ANN).json() == []
