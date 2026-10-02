"""Chat transcripts: store every turn, and pick the recent part that fits a prompt.

Algorithm for the window (`recent_messages`): walk back from the newest message, keep messages
while the count of turns and the token budget allow, always keep at least the newest, and make
sure the window starts with a user message (an answer without its question is confusing).

    stored:  U1 A1 U2 A2 U3 A3 U4 (new question)       budget fits 5 messages
    window:           U2 A2 U3 A3 U4

Conversations and messages are private to their user (row-level security), not just to the org.
"""

import json
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.ingest.chunking import TokenCounter
from app.retrieval.context import approx_tokens
from app.retrieval.search import Scope

TITLE_CHARS = 60


class HistoryError(Exception):
    """A conversation that does not exist or belongs to someone else (same answer for both)."""


def create_conversation(conn: psycopg.Connection, who: Scope, title: str = "") -> uuid.UUID:
    row = conn.execute(
        "INSERT INTO conversations (org_id, user_id, title) VALUES (%s, %s, %s) RETURNING id",
        (who.org_id, who.user_id, title[:TITLE_CHARS]),
    ).fetchone()
    return row[0]  # type: ignore[index]


def conversation_exists(conn: psycopg.Connection, who: Scope, conversation_id: uuid.UUID) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM conversations WHERE id = %s AND org_id = %s AND user_id = %s",
            (conversation_id, who.org_id, who.user_id),
        ).fetchone()
        is not None
    )


def add_message(
    conn: psycopg.Connection,
    who: Scope,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    *,
    citations: list[dict[str, Any]] | None = None,
    refused: bool = False,
) -> uuid.UUID:
    if not conversation_exists(conn, who, conversation_id):
        raise HistoryError("conversation not found")
    row = conn.execute(
        "INSERT INTO messages (conversation_id, org_id, user_id, role, content, citations, "
        "refused) "
        "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s) RETURNING id",
        (
            conversation_id, who.org_id, who.user_id, role, content,
            json.dumps(citations or [], default=str), refused,
        ),
    ).fetchone()  # fmt: skip
    conn.execute("UPDATE conversations SET updated_at = now() WHERE id = %s", (conversation_id,))
    return row[0]  # type: ignore[index]


def recent_messages(
    conn: psycopg.Connection,
    who: Scope,
    conversation_id: uuid.UUID,
    *,
    max_turns: int,
    budget_tokens: int,
    count: TokenCounter = approx_tokens,
) -> list[dict[str, str]]:
    """The window, oldest first, as [{"role", "content"}] ready for a chat prompt."""
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id = %s AND org_id = %s "
            "AND user_id = %s ORDER BY created_at DESC, id DESC LIMIT %s",
            (conversation_id, who.org_id, who.user_id, max_turns * 2 + 1),
        ).fetchall()
    window: list[dict[str, str]] = []
    used = 0
    for row in rows:  # newest first
        cost = count(row["content"])
        if window and used + cost > budget_tokens:
            break
        window.append({"role": row["role"], "content": row["content"]})
        used += cost
    window.reverse()
    while window and window[0]["role"] != "user":
        window.pop(0)
    return window


def list_conversations(
    conn: psycopg.Connection, who: Scope, limit: int = 50
) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE org_id = %s AND user_id = %s ORDER BY updated_at DESC LIMIT %s",
            (who.org_id, who.user_id, limit),
        ).fetchall()


def get_messages(
    conn: psycopg.Connection, who: Scope, conversation_id: uuid.UUID
) -> list[dict[str, Any]]:
    if not conversation_exists(conn, who, conversation_id):
        raise HistoryError("conversation not found")
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT id, role, content, citations, refused, created_at FROM messages "
            "WHERE conversation_id = %s AND org_id = %s AND user_id = %s "
            "ORDER BY created_at, id",
            (conversation_id, who.org_id, who.user_id),
        ).fetchall()


def delete_conversation(conn: psycopg.Connection, who: Scope, conversation_id: uuid.UUID) -> bool:
    """Deletes the conversation and its messages (cascade). Memories learned from it are kept."""
    return (
        conn.execute(
            "DELETE FROM conversations WHERE id = %s AND org_id = %s AND user_id = %s",
            (conversation_id, who.org_id, who.user_id),
        ).rowcount
        == 1
    )
