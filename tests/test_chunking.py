from pathlib import Path

import pytest

from app.ingest.chunking import approx_tokens, chunk_recursive, split_text
from app.ingest.models import ParsedDocument, Section
from app.ingest.parsers import parse_markdown
from app.ingest.preprocess import preprocess

TOPICS = "vacation sick parental travel meals hotels laptops passwords remote equipment".split()
# each sentence is about 10 tokens, so a 50-token limit holds ~4 of them
SENTENCES = [f"Sentence {i} explains {TOPICS[i % 10]} rules clearly." for i in range(30)]
PARAGRAPH = " ".join(SENTENCES)


def doc_of(*texts: str, page: int | None = None) -> ParsedDocument:
    sections = [
        Section(("Doc", f"S{i}"), f"s{i}", t, page=page) for i, t in enumerate(texts, start=1)
    ]
    return ParsedDocument("Doc", "md", sections)


def test_small_section_is_one_chunk_with_metadata() -> None:
    doc = ParsedDocument(
        "D", "pdf", [Section(("D", "Time off"), "p5-time-off", "Short text.", page=5)]
    )
    [chunk] = chunk_recursive(doc, chunk_tokens=50, overlap_tokens=10)
    assert chunk.text == "Short text."
    assert (chunk.heading_path, chunk.anchor, chunk.page_start, chunk.page_end) == (
        ("D", "Time off"),
        "p5-time-off",
        5,
        5,
    )
    assert chunk.index == 0 and chunk.level == 0


def test_long_paragraph_splits_at_sentences_within_limit_and_loses_nothing() -> None:
    chunks = split_text(PARAGRAPH, chunk_tokens=50, overlap_tokens=0)
    assert len(chunks) > 3
    assert all(approx_tokens(c) <= 50 for c in chunks)
    assert all(c.endswith(".") for c in chunks)  # always cut between sentences
    assert " ".join(chunks) == PARAGRAPH  # with no overlap, joining gives back the original


def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    chunks = split_text(PARAGRAPH, chunk_tokens=50, overlap_tokens=12)
    assert all(approx_tokens(c) <= 50 for c in chunks)
    for before, after in zip(chunks, chunks[1:], strict=False):
        last_sentence = before.split(". ")[-1]
        assert after.startswith(last_sentence)  # next chunk begins with the previous chunk's end
    for sentence in SENTENCES:  # every sentence is still present
        assert any(sentence in c for c in chunks)


def test_table_is_never_cut_through_a_row_and_repeats_its_header() -> None:
    table = "| item | cap |\n|---|---|\n" + "\n".join(
        f"| row number {i} | {i} |" for i in range(40)
    )
    chunks = split_text("Intro before the table.\n\n" + table, chunk_tokens=60, overlap_tokens=0)
    table_chunks = [c for c in chunks if "| item | cap |" in c]
    assert len(table_chunks) > 1
    assert all(c.count("| item | cap |") == 1 for c in table_chunks)  # header once per part
    rows = [ln for c in chunks for ln in c.splitlines() if ln.startswith("| row")]
    assert rows == [f"| row number {i} | {i} |" for i in range(40)]  # all rows, intact, in order


def test_code_block_that_fits_stays_whole() -> None:
    code = "```python\nx = 1\n\ny = 2\n```"
    chunks = split_text(PARAGRAPH + "\n\n" + code, chunk_tokens=60, overlap_tokens=0)
    assert sum(code in c for c in chunks) == 1  # fence, blank line and all, in one chunk


def test_sentence_without_breaks_falls_back_to_words() -> None:
    wall = " ".join(f"word{i}" for i in range(200))  # one endless "sentence"
    chunks = split_text(wall, chunk_tokens=40, overlap_tokens=0)
    assert all(approx_tokens(c) <= 40 for c in chunks)
    assert " ".join(chunks).split() == wall.split()  # no word lost, none repeated


def test_single_row_bigger_than_the_limit_stays_whole_and_terminates() -> None:
    table = "| a | b |\n|---|---|\n| " + "x" * 400 + " | y |\n| small | row |"
    chunks = split_text(table, chunk_tokens=40, overlap_tokens=0)
    assert any("x" * 400 in c for c in chunks)  # never cut inside a row
    assert any("| small | row |" in c for c in chunks)


def test_overlap_never_crosses_a_section_boundary() -> None:
    doc = doc_of(PARAGRAPH, PARAGRAPH.replace("Sentence", "Clause"))
    chunks = chunk_recursive(doc, chunk_tokens=50, overlap_tokens=12)
    second = [c for c in chunks if c.anchor == "s2"]
    assert len(second) > 1 and all("Sentence" not in c.text for c in second)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_pluggable_counter_changes_what_fits() -> None:
    words = lambda t: len(t.split())  # noqa: E731  (a tiny counter: 1 word = 1 token)
    chunks = split_text(PARAGRAPH, chunk_tokens=25, overlap_tokens=0, count=words)
    assert all(words(c) <= 25 for c in chunks) and len(chunks) > 3


def test_every_chunk_inherits_page_and_path() -> None:
    doc = doc_of(PARAGRAPH, page=7)
    chunks = chunk_recursive(doc, chunk_tokens=50, overlap_tokens=0)
    assert all(c.page_start == c.page_end == 7 and c.heading_path == ("Doc", "S1") for c in chunks)


# ---- real corpus -----------------------------------------------------------------------------
G = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "gitlab-handbook"
    / "content"
    / "handbook"
)


@pytest.mark.skipif(not G.exists(), reason="corpus not downloaded")
def test_real_handbook_slice_chunks_respect_the_limit() -> None:
    limit = 400
    chunks = []
    for folder in ["total-rewards", "people-policies", "hiring"]:
        for f in (G / folder).rglob("*.md"):
            doc = preprocess(parse_markdown(f.read_bytes(), f.name))
            chunks += chunk_recursive(doc, chunk_tokens=limit, overlap_tokens=50)
    assert len(chunks) > 500
    too_big = [c for c in chunks if c.token_count > limit]
    # only a single table row or code line bigger than the limit may exceed it
    assert len(too_big) / len(chunks) < 0.01, f"{len(too_big)} chunks over the limit"
