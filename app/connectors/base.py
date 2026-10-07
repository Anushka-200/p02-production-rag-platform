"""What every connector provides: a list of pages, each with a fingerprint, nothing more.

A connector is the thin layer between a SOURCE (a git repo, Confluence, Notion, SharePoint) and the
sync engine. It only has to answer one question cheaply: "what pages exist right now, and what is
each page's fingerprint?". Reading a page's content is a separate, lazy step, so a sync that finds
nothing changed reads nothing.

    git:   fingerprint = the blob id (git already hashes every file)
    wiki:  fingerprint = the page's version number or last-modified stamp from the API
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class SourceItem:
    uri: str  # the page's stable identity inside the organisation (also its citation name)
    source_hash: str  # changes if and only if the content changes
    read: Callable[[], bytes]  # fetch the content (lazy: called only for new/changed pages)
    modified: Callable[[], datetime | None]  # when the SOURCE last changed it (lazy)
    visibility: str = "org"  # who may read it: private | org | public (permission sync)


class Connector(Protocol):
    name: str  # stable id, e.g. "git:gitlab-handbook"

    @property
    def ref(self) -> str:
        """What state of the source this listing describes (a git commit, a timestamp)."""

    def items(self) -> Iterator[SourceItem]:
        """Every page that exists now."""


@dataclass
class SyncReport:
    connector: str
    ref: str = ""
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: int = 0
    deleted: list[str] = field(default_factory=list)
    permission_changes: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (uri, reason)
    aborted: str = ""  # a safety stop: nothing was changed
    dry_run: bool = False

    def summary(self) -> str:
        return (
            f"{self.connector} @ {self.ref[:10]}: +{len(self.added)} added, ~{len(self.changed)} "
            f"changed, -{len(self.deleted)} deleted, {len(self.permission_changes)} permission "
            f"changes, {self.unchanged} unchanged, {len(self.failed)} failed"
            + (f"  [ABORTED: {self.aborted}]" if self.aborted else "")
            + ("  [dry run]" if self.dry_run else "")
        )
