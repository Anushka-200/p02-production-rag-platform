import re
from pathlib import Path

import pytest

from app.ingest.parsers import parse, parse_html, parse_markdown, parse_text, slugify

MD = """---
title: Benefits
description: x
---

Intro paragraph before any heading.

## Time off

Unlimited vacation.

### Parental leave

16 weeks paid.

## Equipment

| item | limit |
|---|---|
| laptop | 2000 |
"""


def test_markdown_sections_paths_and_front_matter() -> None:
    doc = parse_markdown(MD, "benefits.md")
    assert doc.title == "Benefits" and doc.metadata["description"] == "x"
    paths = [s.heading_path for s in doc.sections]
    assert paths == [
        ("Benefits",),
        ("Benefits", "Time off"),
        ("Benefits", "Time off", "Parental leave"),
        ("Benefits", "Equipment"),
    ]
    assert doc.sections[2].anchor == "parental-leave"
    assert "16 weeks paid." in doc.sections[2].text


def test_markdown_tables_survive_as_text() -> None:
    doc = parse_markdown(MD)
    assert "| laptop | 2000 |" in doc.sections[-1].text


def test_hash_lines_inside_code_fences_are_not_headings() -> None:
    doc = parse_markdown("# Doc\n\n## Real\n\n```bash\n# just a comment\necho hi\n```\n")
    assert [s.heading_path[-1] for s in doc.sections] == ["Real"]
    assert "# just a comment" in doc.sections[0].text


def test_h1_equal_to_title_is_the_root_not_a_child() -> None:
    doc = parse_markdown("---\ntitle: Guide\n---\n# Guide\n\ntext\n\n## Part\n\nbody\n")
    assert [s.heading_path for s in doc.sections] == [("Guide",), ("Guide", "Part")]


def test_repeated_headings_get_distinct_anchors() -> None:
    doc = parse_markdown("# T\n\n## Notes\n\na\n\n## Notes\n\nb\n")
    assert [s.anchor for s in doc.sections] == ["notes", "notes-1"]


def test_hugo_shortcodes_removed_but_alert_title_and_text_kept() -> None:
    md = (
        '# T\n\n{{% alert title="Warning" color="warning" %}}\nDo not do this.\n'
        "{{% /alert %}}\n{{< engineering/design-document-header >}}\n"
    )
    text = parse_markdown(md).sections[0].text
    assert "Warning: " in text and "Do not do this." in text
    assert "{{" not in text and "%}" not in text


def test_mdx_imports_and_jsx_tags_removed_inner_text_kept() -> None:
    md = (
        'import Foo from "x"\n\n# T\n\n'
        'See <PrivateLink url="https://a.b"> how to </PrivateLink> now.\n\n<SmallTeam name="x" />\n'
    )
    text = parse_markdown(md).sections[0].text
    assert "import" not in text and "<" not in text
    assert "how to" in text


def test_code_fence_content_is_not_stripped() -> None:
    md = "# T\n\n```jsx\n<Button onClick={go}>hi</Button>\n```\n"
    assert "<Button" in parse_markdown(md).sections[0].text


def test_unclosed_code_fence_does_not_swallow_the_rest_of_the_page() -> None:
    md = '# T\n\n```json\n"plan": "x"\n\n{{% /alert %}}\nAfter the forgotten fence.\n'
    text = parse_markdown(md).sections[0].text
    assert "{{%" not in text and "After the forgotten fence." in text


def test_malformed_front_matter_does_not_crash() -> None:
    doc = parse_markdown("---\ntitle: [unclosed\n---\n\n# Hello\n\nbody\n", "x.md")
    assert doc.sections and doc.title == "Hello"


def test_bytes_with_bom_crlf_and_bad_utf8() -> None:
    raw = b"\xef\xbb\xbf# T\r\n\r\ncaf\xe9 body\r\n"
    doc = parse_markdown(raw)
    assert "body" in doc.sections[0].text and "\r" not in doc.sections[0].text


def test_text_parser_and_empty_input() -> None:
    doc = parse_text(b"  hello world  ", "notes.txt")
    assert doc.title == "notes" and doc.sections[0].text == "hello world"
    assert parse_text(b"   ", "x.txt").sections == []


HTML = """<html><head><title>Policy</title><style>p{}</style></head><body>
<nav><a href="/">Home</a> | <a href="/about">About</a></nav>
<main>
<h1>Policy</h1><p>Intro text.</p>
<h2>Leave</h2><ul><li><p>Vacation: 20 days</p></li><li>Sick: 10 days</li></ul>
<h2>Limits</h2><table><tr><th>item</th><th>cap</th></tr><tr><td>meal</td><td>50</td></tr></table>
</main><footer>Copyright</footer><script>alert(1)</script></body></html>"""


def test_html_drops_boilerplate_and_keeps_structure() -> None:
    doc = parse_html(HTML)
    assert doc.title == "Policy"
    all_text = " ".join(s.text for s in doc.sections)
    assert "Home" not in all_text and "Copyright" not in all_text and "alert" not in all_text
    leave = next(s for s in doc.sections if s.heading_path[-1] == "Leave")
    assert leave.text.splitlines() == ["- Vacation: 20 days", "- Sick: 10 days"]  # no duplicates
    limits = next(s for s in doc.sections if s.heading_path[-1] == "Limits")
    assert "| meal | 50 |" in limits.text


def test_parse_dispatch_and_unknown_type() -> None:
    assert parse("md", b"# A\n\nb").source_type == "md"
    with pytest.raises(ValueError, match="no parser"):
        parse("exe", b"")


def test_slugify() -> None:
    assert slugify("Parental Leave & Return!") == "parental-leave-return"


# ---- real corpus smoke test: every handbook file must parse without error -------------------
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
GITLAB = ROOT / "gitlab-handbook" / "content" / "handbook"
POSTHOG = ROOT / "posthog-handbook" / "contents" / "handbook"


@pytest.mark.skipif(not (GITLAB.exists() and POSTHOG.exists()), reason="corpora not downloaded")
def test_every_real_handbook_file_parses() -> None:
    files = [*GITLAB.rglob("*.md"), *POSTHOG.rglob("*.md"), *POSTHOG.rglob("*.mdx")]
    assert len(files) > 4000
    empty, leftovers = [], []
    for f in files:
        doc = parse_markdown(f.read_bytes(), f.name)
        if not doc.sections:
            empty.append(f.name)
        for s in doc.sections:
            outside_code = re.sub(
                r"```.*?```", "", s.text, flags=re.DOTALL
            )  # examples in code are kept
            if "{{%" in outside_code or "{{<" in outside_code:
                leftovers.append(f.name)
    # a few pages are legitimately empty (index stubs); most must have content
    assert len(empty) / len(files) < 0.15, f"too many empty: {len(empty)}"
    assert not leftovers, f"shortcodes left in: {leftovers[:5]}"
