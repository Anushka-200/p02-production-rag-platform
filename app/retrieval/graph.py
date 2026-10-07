"""GraphRAG slice, step 2: answer a question by FOLLOWING facts instead of matching words.

    "Which tool does the team that handles outages escalate to?"
        1. seeds    entities named in the question          -> "support team"
        2. walk     facts touching a seed, then facts touching what those reached (2 hops)
                      (support team) --escalates_to--> (engineering teams)         [chunk X]
                      (engineering teams) --track_in--> (github issues)            [chunk Y]
        3. return   the chunks that STATE those facts (X and Y), nearest hop first

Chunk search finds passages that look like the question; this finds passages that are CONNECTED to
what the question names, even when they share no words with it. The model still reads real chunks,
so answers keep real citations. Hubs (entities in dozens of facts, like "slack") are neither used
as seeds nor walked through: they connect everything and would flood the result with noise.
Visibility is checked on every fact AND on every returned chunk.
"""

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass

import psycopg

from app.retrieval.search import _COLUMNS, VISIBLE_SQL, Candidate, Scope, _candidates

EDGE_VISIBLE = VISIBLE_SQL.replace("c.", "e.")
STOP = set(
    "a an the of to in on for and or is are was were be do does did how what when where which who "
    "why can i you we they it its my our your their this that these those at by with from as if "
    "not there any many much about into than then so".split()
)


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    chunk_id: uuid.UUID
    hop: int


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def question_tokens(question: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9&+#./-]+", question.casefold()) if t not in STOP}


def entity_degrees(conn: psycopg.Connection, scope: Scope) -> dict[str, int]:
    """name -> number of visible facts that mention it (as subject or object)."""
    rows = conn.execute(
        "SELECT name, count(*) FROM ("
        "  SELECT e.subject AS name FROM graph_edges e "
        f"  WHERE e.org_id = %(org)s AND {EDGE_VISIBLE}"
        "  UNION ALL SELECT e.object FROM graph_edges e "
        f"  WHERE e.org_id = %(org)s AND {EDGE_VISIBLE}"
        ") t GROUP BY name",
        {"org": scope.org_id, "user": scope.user_id},
    ).fetchall()  # fmt: skip
    return {r[0]: int(r[1]) for r in rows}


def find_seeds(
    question: str, degrees: dict[str, int], *, limit: int = 5, max_degree: int = 40
) -> list[str]:
    """Entities whose every word appears in the question; longer (more specific) names first."""
    tokens = question_tokens(question)
    found: list[tuple[int, int, str]] = []
    for name, degree in degrees.items():
        words = [_stem(w) for w in name.split() if w not in STOP]
        if not words or degree > max_degree:
            continue
        if len(words) == 1 and len(words[0]) < 4:
            continue  # "ai", "ui": too short to mean anything on their own
        if all(w in tokens for w in words):
            found.append((-len(words), degree, name))
    return [name for _, _, name in sorted(found)[:limit]]


def walk(
    conn: psycopg.Connection, scope: Scope, seeds: list[str], degrees: dict[str, int], *,
    hops: int = 2, max_degree: int = 40, max_facts: int = 80,
) -> list[Fact]:  # fmt: skip
    """Facts reachable from the seeds in at most `hops` steps (nearest first)."""
    facts: dict[uuid.UUID, Fact] = {}
    visited = set(seeds)
    frontier = set(seeds)
    for hop in range(1, hops + 1):
        if not frontier or len(facts) >= max_facts:
            break
        rows = conn.execute(
            "SELECT e.id, e.subject, e.relation, e.object, e.chunk_id FROM graph_edges e "
            "WHERE e.org_id = %(org)s AND " + EDGE_VISIBLE + " "
            "AND (e.subject = ANY(%(f)s) OR e.object = ANY(%(f)s))",
            {"org": scope.org_id, "user": scope.user_id, "f": sorted(frontier)},
        ).fetchall()  # fmt: skip
        reached: set[str] = set()
        for fid, subject, relation, obj, chunk_id in rows:
            if fid in facts or len(facts) >= max_facts:
                continue
            facts[fid] = Fact(subject, relation, obj, chunk_id, hop)
            reached.update((subject, obj))
        frontier = {n for n in reached - visited if degrees.get(n, 0) <= max_degree}
        visited |= frontier
    return sorted(facts.values(), key=lambda f: f.hop)


def evidence_order(facts: list[Fact]) -> list[uuid.UUID]:
    """Chunk ids ordered by (nearest hop, number of facts the chunk states), best first."""
    best_hop: dict[uuid.UUID, int] = {}
    count: dict[uuid.UUID, int] = defaultdict(int)
    for f in facts:
        best_hop[f.chunk_id] = min(best_hop.get(f.chunk_id, f.hop), f.hop)
        count[f.chunk_id] += 1
    return sorted(best_hop, key=lambda c: (best_hop[c], -count[c], str(c)))


def graph_candidates(
    conn: psycopg.Connection, scope: Scope, question: str, *, hops: int = 2, max_chunks: int = 10,
) -> list[Candidate]:  # fmt: skip
    """Chunks that state the facts around the entities the question names (empty if none)."""
    degrees = entity_degrees(conn, scope)
    if not degrees:
        return []
    seeds = find_seeds(question, degrees)
    if not seeds:
        return []
    order = evidence_order(walk(conn, scope, seeds, degrees, hops=hops))[:max_chunks]
    if not order:
        return []
    rows = conn.execute(
        f"SELECT {_COLUMNS}, 0.0 AS score FROM chunks c JOIN documents d ON d.id = c.document_id "  # noqa: S608
        f"WHERE c.id = ANY(%(ids)s) AND c.org_id = %(org)s AND c.level = 0 AND {VISIBLE_SQL} "
        "AND d.status = 'ready' AND c.version = d.active_version",
        {"ids": order, "org": scope.org_id, "user": scope.user_id},
    ).fetchall()  # fmt: skip
    position = {chunk_id: i for i, chunk_id in enumerate(order)}
    return _candidates(sorted(rows, key=lambda r: position[r[0]]))
