"""Documents a user can see, and who may share them.

Algorithm: a document is `private` (only its owner) or `org` (everyone in the organisation).
Changing the visibility is one UPDATE of the document row; the foreign key from `chunks` is
ON UPDATE CASCADE, so Postgres copies the new value onto every chunk in the same transaction.
Permissions therefore cannot drift: there is no moment when the document is private but its chunks
are still searchable by the org.

Only the owner may change it. Row-level security already hides other people's private documents;
this check also stops an org member from un-sharing (or sharing) a colleague's shared document,
which RLS alone would allow them to see but should not let them change.
"""

import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.retrieval.search import Scope

VISIBILITIES = ("private", "org")


class DocumentError(Exception):
    """A clear, user-safe reason a document action was refused."""


def list_documents(conn: psycopg.Connection, who: Scope) -> list[dict[str, Any]]:
    """Documents `who` may see, newest first. Filtered here AND by RLS: two independent locks."""
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT id, title, source_uri, source_type, visibility, status, owner_id, updated_at "
            "FROM documents WHERE org_id = %s AND (visibility = 'org' OR owner_id = %s) "
            "ORDER BY updated_at DESC LIMIT 500",
            (who.org_id, who.user_id),
        ).fetchall()


def set_visibility(
    conn: psycopg.Connection, who: Scope, document_id: uuid.UUID, visibility: str
) -> None:
    if visibility not in VISIBILITIES:
        raise DocumentError(f"visibility must be one of {VISIBILITIES}")
    row = conn.execute(
        "UPDATE documents SET visibility = %s, updated_at = now() "
        "WHERE id = %s AND owner_id = %s RETURNING id",
        (visibility, document_id, who.user_id),
    ).fetchone()
    if row is None:
        # Not found, someone else's, or hidden by RLS: the same answer, so ids cannot be probed.
        raise DocumentError("document not found, or you are not its owner")
