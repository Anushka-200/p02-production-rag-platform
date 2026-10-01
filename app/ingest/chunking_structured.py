"""Chunker B: heading-aware chunks with parents.

What it fixes compared with chunker A (which left ~10% of chunks under 50 tokens):
1. Small neighbouring sections of the same topic branch are MERGED into one chunk.
2. Cuts happen at section boundaries; only a section too big on its own is split (recursively).
3. Every chunk starts with a breadcrumb ("Handbook > Benefits > Time off") so it describes itself.
4. A group of sections bigger than one chunk also gets a PARENT chunk (level 1). Children
   (level 0) are what we embed and search; the parent is what we can hand to the model for
   more context ("parent expansion", Phase 5).

Steps: group sections -> for each group: one child, or parent + several children.
"""

from collections.abc import Callable

from app.ingest.chunking import Chunk, TokenCounter, approx_tokens, split_text
from app.ingest.models import ParsedDocument, Section


class _Atom:
    """A section (or a piece of a big section) on its way into a chunk."""

    def __init__(self, path: tuple[str, ...], anchor: str, page: int | None, body: str) -> None:
        self.path, self.anchor, self.page, self.body = path, anchor, page, body


def _atom(section: Section) -> _Atom:
    return _Atom(section.heading_path, section.anchor, section.page, section.text)


def _common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    prefix = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix), len(path)) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def _render(atoms: list[_Atom], with_headings: bool) -> tuple[str, tuple[str, ...]]:
    """Chunk text and its heading path (the part of the path all atoms share).

    With headings: a breadcrumb line first, then each atom whose own path goes deeper gets a
    heading line showing only the extra part of its path.
    """
    common = _common_prefix([a.path for a in atoms])
    parts: list[str] = []
    if with_headings:
        parts.append(" > ".join(common))
    for atom in atoms:
        if with_headings and atom.path != common:
            parts.append("#" * len(atom.path) + " " + " > ".join(atom.path[len(common) :]))
        parts.append(atom.body)
    return "\n\n".join(parts), common


def _branch(path: tuple[str, ...]) -> tuple[str, ...] | None:
    """The topic branch of a section: title + first heading. Title-only intro text has none."""
    return path[:2] if len(path) >= 2 else None


def _group_sections(
    sections: list[Section],
    parent_tokens: int,
    count: TokenCounter,
    with_headings: bool,
    merge_across_branches: bool,
) -> list[list[Section]]:
    """Merge neighbouring sections of one branch while the group stays within parent_tokens."""
    groups: list[list[Section]] = []
    current: list[Section] = []
    branch: tuple[str, ...] | None = None
    for section in sections:
        b = _branch(section.heading_path)
        if current:
            same_topic = merge_across_branches or b is None or branch is None or b == branch
            candidate = [*current, section]
            text, _ = _render([_atom(s) for s in candidate], with_headings)
            if same_topic and count(text) <= parent_tokens:
                current = candidate
                branch = branch or b  # an intro-only group adopts the branch of the next section
                continue
            groups.append(current)
        current, branch = [section], b
    if current:
        groups.append(current)
    return groups


def _split_big_atom(
    section: Section, limit: int, overlap: int, count: TokenCounter, with_headings: bool
) -> list[_Atom]:
    """A section too big for one chunk: split its body, leaving room for the breadcrumb."""
    empty = _Atom(section.heading_path, section.anchor, section.page, "")
    overhead = count(_render([empty], with_headings)[0]) + 2
    piece_limit = max(limit - overhead, limit // 2)
    return [
        _Atom(section.heading_path, section.anchor, section.page, piece)
        for piece in split_text(section.text, piece_limit, overlap, count)
    ]


def _make_chunk(
    atoms: list[_Atom],
    text: str,
    path: tuple[str, ...],
    level: int,
    parent_index: int | None,
    index: int,
    count: TokenCounter,
) -> Chunk:
    pages = [a.page for a in atoms if a.page is not None]
    return Chunk(
        text=text,
        heading_path=path,
        anchor=atoms[0].anchor,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        token_count=count(text),
        index=index,
        level=level,
        anchors=tuple(dict.fromkeys(a.anchor for a in atoms)),  # unique, in order
        parent_index=parent_index,
    )


def chunk_structured(
    doc: ParsedDocument,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
    parent_tokens: int = 1200,
    count: TokenCounter = approx_tokens,
    prefix_headings: bool = True,
    merge_across_branches: bool = False,
) -> list[Chunk]:
    out: list[Chunk] = []
    groups = _group_sections(
        doc.sections, parent_tokens, count, prefix_headings, merge_across_branches
    )
    for group in groups:
        atoms = [_atom(s) for s in group]
        full_text, full_path = _render(atoms, prefix_headings)
        total = count(full_text)

        if total <= chunk_tokens:  # small group: ONE chunk, no parent needed
            out.append(_make_chunk(atoms, full_text, full_path, 0, None, len(out), count))
            continue

        parent_index = None
        if total <= parent_tokens:  # medium group: the whole group is the parent
            parent_index = len(out)
            out.append(_make_chunk(atoms, full_text, full_path, 1, None, parent_index, count))

        # Pack the group's sections into children. Whole sections are the unit, so cuts fall
        # on heading boundaries; a section too big alone is split by the recursive splitter.
        current: list[_Atom] = []

        def flush(parent: int | None = parent_index, bucket: list[_Atom] = current) -> None:
            if bucket:
                text, path = _render(bucket, prefix_headings)
                out.append(_make_chunk(list(bucket), text, path, 0, parent, len(out), count))
                bucket.clear()

        for section in group:
            atom = _atom(section)
            if count(_render([atom], prefix_headings)[0]) > chunk_tokens:
                flush()
                for piece in _split_big_atom(
                    section, chunk_tokens, overlap_tokens, count, prefix_headings
                ):
                    text, path = _render([piece], prefix_headings)
                    out.append(_make_chunk([piece], text, path, 0, parent_index, len(out), count))
                continue
            if current and count(_render([*current, atom], prefix_headings)[0]) > chunk_tokens:
                flush()
            current.append(atom)
        flush()
    return out


Chunker = Callable[..., list[Chunk]]
