"""Reciprocal Rank Fusion: merge several ranked lists into one by RANK, not by score.

Dense scores (cosine distances) and keyword scores (ts_rank) are on different scales, so adding
them is meaningless. Ranks are comparable: each list gives a chunk weight / (k + rank) points,
and the points add up. A chunk both lists like beats a chunk only one list loves.
"""

from dataclasses import dataclass

from app.retrieval.search import Candidate


@dataclass
class Fused:
    candidate: Candidate  # the chunk (its text, path, pages...), from the list that ranked it best
    rrf_score: float
    dense_rank: int | None = None
    dense_distance: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rank: int = 0  # final position, 1 = best


def fuse(
    dense: list[Candidate],
    keyword: list[Candidate],
    k: int = 60,
    dense_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[Fused]:
    """Merge the two ranked lists. k damps the gap between rank 1 and rank 2 (60 is the usual)."""
    merged: dict[object, Fused] = {}
    for source, items, weight in (
        ("dense", dense, dense_weight),
        ("keyword", keyword, keyword_weight),
    ):
        for c in items:
            f = merged.setdefault(c.chunk_id, Fused(candidate=c, rrf_score=0.0))
            f.rrf_score += weight / (k + c.rank)
            if source == "dense":
                f.dense_rank, f.dense_distance = c.rank, c.score
            else:
                f.keyword_rank, f.keyword_score = c.rank, c.score
    best_rank = lambda f: min(r for r in (f.dense_rank, f.keyword_rank) if r is not None)  # noqa: E731
    # Ties: the chunk with the better single rank first, then by id, so results never reshuffle.
    ordered = sorted(
        merged.values(), key=lambda f: (-f.rrf_score, best_rank(f), str(f.candidate.chunk_id))
    )
    for position, f in enumerate(ordered, start=1):
        f.rank = position
    return ordered


def merge_fused(
    lists: list[list[Fused]], k: int = 60, weights: list[float] | None = None
) -> list[Fused]:
    """Merge several RANKED RESULT LISTS (one per query variant) with the same rank-based maths.

    A chunk found by three phrasings of the question collects points from each, so it beats a chunk
    only one phrasing found. The chunk keeps the dense/keyword details from the list where it ranked
    BEST (so the inspector still explains it); its score and position are the merged ones.
    """
    weights = weights or [1.0] * len(lists)
    merged: dict[object, Fused] = {}
    for items, weight in zip(lists, weights, strict=True):
        for f in items:
            key = f.candidate.chunk_id
            if key not in merged:
                merged[key] = Fused(
                    candidate=f.candidate, rrf_score=0.0, dense_rank=f.dense_rank,
                    dense_distance=f.dense_distance, keyword_rank=f.keyword_rank,
                    keyword_score=f.keyword_score,
                )  # fmt: skip
            merged[key].rrf_score += weight / (k + f.rank)
    ordered = sorted(merged.values(), key=lambda f: (-f.rrf_score, str(f.candidate.chunk_id)))
    for position, f in enumerate(ordered, start=1):
        f.rank = position
    return ordered
