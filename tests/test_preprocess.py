import re
from pathlib import Path

import pytest

from app.ingest.models import ParsedDocument, Section
from app.ingest.parsers import parse
from app.ingest.preprocess import (
    clean_text,
    drop_duplicate_sections,
    preprocess,
    remove_page_furniture,
    split_blocks,
    split_table,
)

TOPICS = (
    "vacation sick parental travel meals hotels laptops passwords remote equipment tuition".split()
)


def pdf_doc(pages: int, footer: str = "{n} Publication 15-B (2026)") -> ParsedDocument:
    # Real pages differ in their words, not just in a digit, so each page gets its own topic.
    sections = [
        Section(
            heading_path=("Guide",),
            anchor=f"p{n}",
            text=f"This page explains {TOPICS[n % len(TOPICS)]} rules in detail.\n\n"
            + footer.format(n=n),
            page=n,
        )
        for n in range(1, pages + 1)
    ]
    return ParsedDocument("Guide", "pdf", sections, {"pages": pages})


def test_repeated_footer_removed_but_content_kept() -> None:
    doc, removed = remove_page_furniture(pdf_doc(10))
    assert removed == 10
    assert all("Publication" not in s.text for s in doc.sections)
    assert all("This page explains" in s.text for s in doc.sections)


def test_footer_with_page_number_on_either_side_is_caught() -> None:
    doc = pdf_doc(10)
    for s in doc.sections[::2]:  # even/odd pages put the number on opposite sides
        s.text = s.text.replace(
            f"{s.page} Publication 15-B (2026)", f"Publication 15-B (2026) {s.page}"
        )
    cleaned, removed = remove_page_furniture(doc)
    assert removed == 10 and all("Publication" not in s.text for s in cleaned.sections)


def test_short_pdfs_and_non_pdfs_are_left_alone() -> None:
    assert remove_page_furniture(pdf_doc(3))[1] == 0
    md = ParsedDocument("T", "md", [Section(("T",), "t", "x")], {"pages": 99})
    assert remove_page_furniture(md)[1] == 0


def test_a_short_phrase_on_few_pages_is_not_furniture() -> None:
    doc = pdf_doc(10, footer="Note.")
    doc.sections[0].text += "\n\nExample."  # appears on 1 of 10 pages only
    cleaned, _ = remove_page_furniture(doc)
    assert "Example." in cleaned.sections[0].text


def test_duplicate_sections_dropped_only_when_long_enough() -> None:
    long_text = "word " * 25
    sections = [
        Section(("T", "A"), "a", long_text),
        Section(("T", "B"), "b", long_text.upper()),  # same text, different case
        Section(("T", "C"), "c", "See above."),
        Section(("T", "D"), "d", "See above."),  # too short to dedupe
    ]
    doc, dropped = drop_duplicate_sections(ParsedDocument("T", "md", sections))
    assert dropped == 1 and [s.anchor for s in doc.sections] == ["a", "c", "d"]


def test_clean_text_removes_invisible_characters() -> None:
    assert clean_text("a​b c  \nnext﻿") == "ab c\nnext"


def test_preprocess_records_stats_and_drops_emptied_sections() -> None:
    doc = pdf_doc(10)
    doc.sections.append(Section(("Guide",), "p11", "11 Publication 15-B (2026)", page=11))
    doc.metadata["pages"] = 11
    out = preprocess(doc)
    assert out.metadata["preprocess"]["furniture_paragraphs_removed"] == 11
    assert len(out.sections) == 10  # the page that held only a footer is gone


# ---- blocks and tables ----------------------------------------------------------------------
TEXT = """Intro paragraph
continues here.

| item | cap |
|---|---|
| meal | 50 |

```python
# not | a table
x = 1

y = 2
```

Closing paragraph."""


def test_split_blocks_separates_paragraphs_tables_and_code() -> None:
    blocks = split_blocks(TEXT)
    assert [b.kind for b in blocks] == ["paragraph", "table", "code", "paragraph"]
    assert blocks[0].text == "Intro paragraph\ncontinues here."
    assert blocks[1].text.splitlines()[0] == "| item | cap |"
    assert "y = 2" in blocks[2].text  # a blank line inside code does not split it


def test_table_directly_followed_by_text_ends_the_table() -> None:
    blocks = split_blocks("| a | b |\n|---|---|\n| 1 | 2 |\nNext line")
    assert [b.kind for b in blocks] == ["table", "paragraph"]


def test_split_table_repeats_header_in_every_part() -> None:
    table = "| item | cap |\n|---|---|\n" + "\n".join(f"| row{i} | {i} |" for i in range(20))
    parts = split_table(table, max_size=80)
    assert len(parts) > 1
    assert all(p.startswith("| item | cap |\n|---|---|") for p in parts)
    assert all(len(p) <= 80 for p in parts)
    rows = [line for p in parts for line in p.splitlines()[2:]]
    assert rows == [f"| row{i} | {i} |" for i in range(20)]  # no row lost, none repeated, in order


def test_split_table_keeps_small_tables_whole_and_never_cuts_a_row() -> None:
    small = "| a |\n|---|\n| 1 |"
    assert split_table(small, 1000) == [small]
    huge_row = "| a |\n|---|\n| " + "x" * 200 + " |\n| y |"
    parts = split_table(huge_row, 50)
    assert any("x" * 200 in p for p in parts)  # the oversized row stays whole


# ---- real data -------------------------------------------------------------------------------
HR = Path(__file__).resolve().parents[1] / "data" / "raw" / "hr-pdfs"


@pytest.mark.skipif(
    not (HR / "irs-p15b-fringe-benefits.pdf").exists(), reason="PDFs not downloaded"
)
def test_real_irs_pdf_loses_footers_but_keeps_content() -> None:
    raw = (HR / "irs-p15b-fringe-benefits.pdf").read_bytes()
    before = parse("pdf", raw, "p15b.pdf")
    after = preprocess(before)
    n = after.metadata["preprocess"]["furniture_paragraphs_removed"]
    assert 30 <= n <= 40  # about one footer per page on a 36-page file
    footer = re.compile(r"^(\d+ )?Publication 15-B \(2026\)( \d+)?$")
    paragraphs = [p for s in after.sections for p in s.text.split("\n\n")]
    assert not any(footer.match(p.strip()) for p in paragraphs)  # footer lines are gone
    # the cover page still mentions the publication inside a real sentence: that stays
    assert any("Catalog Number" in p for p in paragraphs)
    assert "Fringe Benefit Exclusion Rules" in " ".join(
        " ".join(s.heading_path) for s in after.sections
    )
    before_words = sum(len(s.text.split()) for s in before.sections)
    after_words = sum(len(s.text.split()) for s in after.sections)
    assert after_words > 0.95 * before_words  # we removed footers, not real content
