"""A connector for a git repository (a docs site, a handbook): the "living wiki" demo source.

Pages are the files under a path prefix at one commit. Git already gives every file a content
fingerprint (its blob id), so listing the pages costs one `git ls-tree` and reads no file. Content
and "last changed" dates are fetched lazily, only for pages the sync decides it needs.

Permissions: a git repo has no per-page access control, so access is a RULE: paths under a given
prefix are private/public (the first matching rule wins; everything else gets the default). That
mirrors how a wiki restricts a folder or space, and changing a rule is a permission change.

Safety: the repository path is checked, all commands are fixed argument lists (no shell), and the
revision and prefix may not start with "-" (so they can never be read as an option).
"""

import re
import subprocess
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath

from app.connectors.base import SourceItem

_SHA = re.compile(r"^[0-9a-f]{40}$")
PAGE_SUFFIXES = (".md", ".mdx", ".txt", ".html")


class GitError(RuntimeError):
    pass


def _run(repo: Path, args: tuple[str, ...]) -> bytes:
    try:
        done = subprocess.run(  # noqa: S603 (fixed argument list, never a shell)
            ["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=120
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        raise GitError(f"git {' '.join(args[:2])} failed: {stderr[:200]!r}") from exc
    return done.stdout


def _git_bytes(repo: Path, *args: str) -> bytes:
    return _run(repo, args)


def _git_text(repo: Path, *args: str) -> str:
    return _run(repo, args).decode("utf-8", errors="replace")


class GitConnector:
    def __init__(
        self,
        repo: str | Path,
        rev: str,
        *,
        name: str,
        path_prefix: str = "",
        uri_prefix: str = "",
        suffixes: tuple[str, ...] = PAGE_SUFFIXES,
        default_visibility: str = "org",
        visibility_rules: list[tuple[str, str]] | None = None,
    ) -> None:
        self.repo = Path(repo)
        if not (self.repo / ".git").exists():
            raise GitError(f"{self.repo} is not a git repository")
        for value in (rev, path_prefix):
            if value.startswith("-"):
                raise GitError("revisions and paths may not start with '-'")
        self.name = name
        self.path_prefix = path_prefix.strip("/")
        self.uri_prefix = uri_prefix.strip("/")
        self.suffixes = suffixes
        self.default_visibility = default_visibility
        self.visibility_rules = visibility_rules or []
        sha = _git_text(self.repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
        if not _SHA.match(sha):
            raise GitError(f"cannot resolve revision {rev!r}")
        self._sha = sha

    @property
    def ref(self) -> str:
        return self._sha

    def visibility_for(self, relative_path: str) -> str:
        for prefix, visibility in self.visibility_rules:
            if relative_path.startswith(prefix):
                return visibility
        return self.default_visibility

    def _uri(self, path: str) -> str:
        relative = path[len(self.path_prefix) :].lstrip("/") if self.path_prefix else path
        return f"{self.uri_prefix}/{relative}" if self.uri_prefix else relative

    def items(self) -> Iterator[SourceItem]:
        args = ["ls-tree", "-r", "-z", "--full-tree", self._sha]
        if self.path_prefix:
            args += ["--", self.path_prefix]
        listing = _git_text(self.repo, *args)
        for entry in filter(None, listing.split("\0")):
            meta, _, path = entry.partition("\t")
            mode, kind, blob = meta.split()[:3]
            if kind != "blob" or not path.lower().endswith(self.suffixes):
                continue
            relative = path[len(self.path_prefix) :].lstrip("/") if self.path_prefix else path
            yield SourceItem(
                uri=self._uri(path),
                source_hash=blob,
                read=self._reader(blob),
                modified=self._modified(path),
                visibility=self.visibility_for(relative),
            )

    def _reader(self, blob: str) -> Callable[[], bytes]:
        return lambda: _git_bytes(self.repo, "cat-file", "blob", blob)

    def _modified(self, path: str) -> Callable[[], datetime | None]:
        def when() -> datetime | None:
            stamp = _git_text(self.repo, "log", "-1", "--format=%cI", self._sha, "--", path).strip()
            return datetime.fromisoformat(stamp) if stamp else None

        return when


def page_name(uri: str) -> str:
    return PurePosixPath(uri).name
