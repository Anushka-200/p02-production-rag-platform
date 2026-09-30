"""What every parser returns, whatever the file format."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    heading_path: tuple[str, ...]  # e.g. ("Benefits", "Parental leave"); first item is the title
    anchor: str  # slug of the heading: the "page" locator for Markdown/HTML sources
    text: str  # body text; Markdown formatting (lists, tables) is kept on purpose
    page: int | None = None  # PDFs/DOCX fill this in (task 1.4)


@dataclass
class ParsedDocument:
    title: str
    source_type: str  # md | txt | html | pdf | docx
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)  # front matter etc.
