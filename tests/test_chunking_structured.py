import statistics
from pathlib import Path

import pytest

from app.config import Settings
from app.ingest.chunkers import chunk_document
from app.ingest.chunking import approx_tokens, chunk_recursive
from app.ingest.chunking_structured import chunk_structured
from app.ingest.common import slugify
from app.ingest.models import ParsedDocument, Section
from app.ingest.parsers import parse_markdown
from app.ingest.preprocess import preprocess

TOPICS = "vacation sick parental travel meals hotels laptops passwords remote equipment".split()
SENTENCES = [f"Sentence {i} explains {TOPICS[i % 10]} rules clearly." for i in range(30)]
PARAGRAPH = " ".join(SENTENCES)  # about 330 tokens


def sec(path: tuple[str, ...], text: str, page: int | None = None) -> Section:
    return Section(path, f"p{page}-{slugify(path[-1])}" if page else slugify(path[-1]), text, page)


def doc_of(*sections: Section, kind: str = "md") -> ParsedDocument:
    return ParsedDocument("D", kind, list(sections))


def test_tiny_sections_of_one_branch_merge_into_one_chunk() -> None:
    doc = doc_of(
        sec(("D", "Ben"), "Benefits intro."),
        sec(("D", "Ben", "Time off"), "Twenty vacation days."),
        sec(("D", "Ben", "Expenses"), "Meals are capped at fifty."),
    )
    [chunk] = chunk_structured(doc, chunk_tokens=100, parent_tokens=300)
    assert chunk.level == 0 and chunk.parent_index is None
    assert chunk.heading_path == ("D", "Ben")  # the part of the path all sections share
    assert chunk.anchors == ("ben", "time-off", "expenses")
    assert chunk.text.startswith("D > Ben\n\nBenefits intro.")  # breadcrumb first
    assert "### Time off\n\nTwenty vacation days." in chunk.text  # sub-section keeps its heading


def test_sections_of_different_branches_are_not_merged_unless_asked() -> None:
    doc = doc_of(sec(("D", "A"), "Alpha text."), sec(("D", "B"), "Beta text."))
    assert len(chunk_structured(doc, chunk_tokens=100)) == 2
    assert len(chunk_structured(doc, chunk_tokens=100, merge_across_branches=True)) == 1


def test_title_intro_joins_the_first_branch_only() -> None:
    doc = doc_of(sec(("D",), "Welcome."), sec(("D", "A"), "Alpha."), sec(("D", "B"), "Beta."))
    chunks = chunk_structured(doc, chunk_tokens=100)
    assert [c.anchors for c in chunks] == [("d", "a"), ("b",)]


def test_section_too_big_for_one_chunk_is_split_with_breadcrumb_and_overlap() -> None:
    doc = doc_of(sec(("D", "Big"), PARAGRAPH))
    chunks = chunk_structured(doc, chunk_tokens=100, overlap_tokens=12, parent_tokens=200)
    assert len(chunks) > 3 and all(c.level == 0 and c.parent_index is None for c in chunks)
    assert all(c.token_count <= 100 for c in chunks)  # the breadcrumb fits inside the limit
    assert all(c.text.startswith("D > Big\n\n") for c in chunks)
    for sentence in SENTENCES:
        assert any(sentence in c.text for c in chunks)  # nothing lost


def test_medium_group_gets_a_parent_and_children_that_point_to_it() -> None:
    bodies = [f"Section {i} " + "word " * 25 for i in range(4)]  # about 35 tokens each
    doc = doc_of(*[sec(("D", "Topic", f"Part {i}"), b) for i, b in enumerate(bodies)])
    chunks = chunk_structured(doc, chunk_tokens=100, overlap_tokens=10, parent_tokens=300)
    parent, *children = chunks
    assert parent.level == 1 and parent.index == 0 and parent.parent_index is None
    assert len(children) == 2 and all(c.level == 0 and c.parent_index == 0 for c in children)
    assert all(c.token_count <= 100 for c in children)
    assert all(f"Section {i}" in parent.text for i in range(4))  # the parent holds everything
    assert [a for c in children for a in c.anchors] == ["part-0", "part-1", "part-2", "part-3"]


def test_chunk_indexes_are_sequential_and_parents_come_first() -> None:
    doc = doc_of(*[sec(("D", f"T{i}", "Sub"), "word " * 60) for i in range(3)])
    chunks = chunk_structured(doc, chunk_tokens=40, parent_tokens=500)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        if c.parent_index is not None:
            assert c.parent_index < c.index and chunks[c.parent_index].level == 1


def test_big_table_is_split_by_rows_with_the_header_repeated() -> None:
    table = "| item | cap |\n|---|---|\n" + "\n".join(
        f"| row number {i} | {i} |" for i in range(40)
    )
    doc = doc_of(sec(("D", "Limits"), "Intro.\n\n" + table))
    chunks = chunk_structured(doc, chunk_tokens=70, overlap_tokens=0, parent_tokens=120)
    parts = [c for c in chunks if "| item | cap |" in c.text]
    assert len(parts) > 1 and all(c.text.count("| item | cap |") == 1 for c in parts)
    rows = [ln for c in chunks for ln in c.text.splitlines() if ln.startswith("| row")]
    assert rows == [f"| row number {i} | {i} |" for i in range(40)]


def test_headings_can_be_switched_off_for_experiments() -> None:
    doc = doc_of(sec(("D", "A"), "Alpha text."))
    [chunk] = chunk_structured(doc, prefix_headings=False)
    assert chunk.text == "Alpha text."


def test_page_range_covers_all_merged_pdf_sections() -> None:
    doc = doc_of(
        sec(("D", "Rules"), "First part.", page=5),
        sec(("D", "Rules"), "Second part.", page=6),
        kind="pdf",
    )
    [chunk] = chunk_structured(doc, chunk_tokens=100)
    assert (chunk.page_start, chunk.page_end) == (5, 6)


def test_strategy_is_chosen_from_settings() -> None:
    doc = doc_of(sec(("D", "A"), "Alpha text."), sec(("D", "B"), "Beta text."))
    structured = Settings(_env_file=None, chunk_strategy="structured")  # type: ignore[call-arg]
    recursive = Settings(_env_file=None, chunk_strategy="recursive")  # type: ignore[call-arg]
    assert all("D > " in c.text for c in chunk_document(doc, structured, count=approx_tokens))
    assert all("D > " not in c.text for c in chunk_document(doc, recursive, count=approx_tokens))


# ---- real corpus: does B actually fix chunker A's tiny-chunk problem? --------------------------
G = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "gitlab-handbook"
    / "content"
    / "handbook"
)


@pytest.mark.skipif(not G.exists(), reason="corpus not downloaded")
def test_real_slice_structured_has_fewer_tiny_chunks_than_recursive() -> None:
    docs = [
        preprocess(parse_markdown(f.read_bytes(), f.name))
        for folder in ["total-rewards", "people-policies", "hiring", "values"]
        for f in (G / folder).rglob("*.md")
    ]
    a = [c for d in docs for c in chunk_recursive(d, 400, 50)]
    b = [c for d in docs for c in chunk_structured(d, 400, 50, 1200)]
    children = [c for c in b if c.level == 0]

    def tiny(chunks: list) -> float:  # type: ignore[type-arg]
        return sum(c.token_count < 50 for c in chunks) / len(chunks)

    assert tiny(children) < tiny(a) / 3, (tiny(children), tiny(a))
    assert statistics.median(c.token_count for c in children) > statistics.median(
        c.token_count for c in a
    )
    assert sum(c.token_count > 400 for c in children) / len(children) < 0.02
    assert any(c.level == 1 for c in b)  # parents exist
    assert all(approx_tokens(c.text) > 0 for c in b)
