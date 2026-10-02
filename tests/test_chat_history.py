"""Conversation storage and the prompt window."""

import psycopg
import pytest

from app.chat_history import (
    HistoryError,
    add_message,
    conversation_exists,
    create_conversation,
    delete_conversation,
    get_messages,
    list_conversations,
    recent_messages,
)
from tests.test_search import World


@pytest.fixture
def w(db_schema):  # type: ignore[no-untyped-def]
    world = World(db_schema)
    world.ann = world.person("acme", "ann")  # type: ignore[attr-defined]
    world.amy = world.person("acme", "amy")  # type: ignore[attr-defined]
    return world


def chat(w, who, turns: list[tuple[str, str]]):  # type: ignore[no-untyped-def]
    conv = create_conversation(w.conn, who, "first question")
    for role, content in turns:
        add_message(w.conn, who, conv, role, content)
    return conv


def test_messages_come_back_in_order_with_citations(w) -> None:  # type: ignore[no-untyped-def]
    conv = create_conversation(w.conn, w.ann, "How many days?")
    add_message(w.conn, w.ann, conv, "user", "How many days?")
    add_message(w.conn, w.ann, conv, "assistant", "Twenty [S1]", citations=[{"id": "S1"}])
    got = get_messages(w.conn, w.ann, conv)
    assert [(m["role"], m["content"]) for m in got] == [
        ("user", "How many days?"),
        ("assistant", "Twenty [S1]"),
    ]
    assert got[1]["citations"] == [{"id": "S1"}]


def test_a_conversation_is_private_to_its_user_even_inside_one_org(w) -> None:  # type: ignore[no-untyped-def]
    conv = chat(w, w.ann, [("user", "secret plan")])
    assert not conversation_exists(w.conn, w.amy, conv)
    with pytest.raises(HistoryError):
        add_message(w.conn, w.amy, conv, "user", "injected")
    with pytest.raises(HistoryError):
        get_messages(w.conn, w.amy, conv)
    assert list_conversations(w.conn, w.amy) == []
    assert delete_conversation(w.conn, w.amy, conv) is False
    assert len(get_messages(w.conn, w.ann, conv)) == 1


def test_the_database_itself_refuses_a_message_in_someone_elses_conversation(w) -> None:  # type: ignore[no-untyped-def]
    conv = chat(w, w.ann, [("user", "hi")])
    with pytest.raises(psycopg.errors.ForeignKeyViolation):  # bypasses the Python check on purpose
        w.conn.execute(
            "INSERT INTO messages (conversation_id, org_id, user_id, role, content) "
            "VALUES (%s, %s, %s, 'user', 'x')",
            (conv, w.amy.org_id, w.amy.user_id),
        )


def test_deleting_a_conversation_removes_its_messages(w) -> None:  # type: ignore[no-untyped-def]
    conv = chat(w, w.ann, [("user", "a"), ("assistant", "b")])
    assert delete_conversation(w.conn, w.ann, conv) is True
    assert w.conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


TURNS = [
    ("user", "u1 " * 10),
    ("assistant", "a1 " * 10),
    ("user", "u2 " * 10),
    ("assistant", "a2 " * 10),
    ("user", "u3 " * 10),
]


def window(w, conv, **kw):  # type: ignore[no-untyped-def]
    return recent_messages(w.conn, w.ann, conv, count=lambda t: len(t.split()), **kw)


def test_the_window_keeps_the_newest_messages_that_fit_the_budget(w) -> None:  # type: ignore[no-untyped-def]
    conv = chat(w, w.ann, TURNS)
    got = window(w, conv, max_turns=10, budget_tokens=25)  # room for two 10-word messages
    assert [m["role"] for m in got] == [
        "user"
    ]  # u3 fits; a2 would make a window starting with 'assistant'


def test_the_window_always_starts_with_a_user_message(w) -> None:  # type: ignore[no-untyped-def]
    conv = chat(w, w.ann, TURNS)
    for budget in (10, 20, 30, 40, 50, 1000):
        got = window(w, conv, max_turns=10, budget_tokens=budget)
        assert got and got[0]["role"] == "user"


def test_the_newest_message_is_kept_even_when_it_alone_exceeds_the_budget(w) -> None:  # type: ignore[no-untyped-def]
    conv = chat(w, w.ann, TURNS)
    got = window(w, conv, max_turns=10, budget_tokens=1)
    assert [m["content"] for m in got] == ["u3 " * 10]


def test_the_window_respects_the_turn_limit_and_returns_oldest_first(w) -> None:  # type: ignore[no-untyped-def]
    conv = chat(w, w.ann, TURNS)
    got = window(w, conv, max_turns=1, budget_tokens=1000)  # 1 turn = 2 messages + the new question
    assert [m["content"].split()[0] for m in got] == ["u2", "a2", "u3"]
