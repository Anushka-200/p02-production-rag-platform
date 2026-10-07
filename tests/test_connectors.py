"""Connectors and incremental sync: a real temporary git repo, the real ingestion pipeline."""

import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.connectors.base import SourceItem
from app.connectors.git import GitConnector, GitError
from app.connectors.sync import make_pipeline_ingest, plan_sync, sync_connector
from tests.test_api import FakeEmbedder
from tests.test_search import World

COUNT = lambda text: len(text.split())  # noqa: E731


def git(repo: Path, *args: str, date: str = "2026-01-10T10:00:00+00:00") -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    out = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    )
    return out.stdout.strip()


def write(repo: Path, files: dict[str, str | None]) -> None:
    for name, content in files.items():
        path = repo / name
        if content is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def commit(
    repo: Path, files: dict[str, str | None], message: str, date: str = "2026-01-10T10:00:00+00:00"
) -> str:
    write(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message, date=date)
    return git(repo, "rev-parse", "HEAD")


PAGE = "# {t}\n\n## Policy\n\n{t} policy text. " + "Words fill the page so it is a real chunk. " * 6


def page(title: str) -> str:
    return PAGE.format(t=title)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "handbook"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "Test")
    commit(r, {"content/people/leave.md": page("Leave"), "content/people/pay.md": page("Pay"),
               "content/people/private/salary.md": page("Salary"), "content/readme.png": "binary", "other/x.md": page("Other")}, "first")  # fmt: skip
    return r


def connector(repo: Path, rev: str = "HEAD", **kw: Any) -> GitConnector:
    return GitConnector(
        repo, rev, name="git:test", path_prefix="content/people", uri_prefix="hb", **kw
    )


# ---- the git connector ------------------------------------------------------------------------
def test_it_lists_only_page_files_under_the_prefix_with_stable_uris_and_content_fingerprints(
    repo: Path,
) -> None:
    items = {i.uri: i for i in connector(repo).items()}
    assert sorted(items) == [
        "hb/leave.md",
        "hb/pay.md",
        "hb/private/salary.md",
    ]  # no png, nothing outside the prefix
    expected = git(repo, "rev-parse", "HEAD:content/people/leave.md")
    assert items["hb/leave.md"].source_hash == expected  # git's own blob id: free change detection


def test_content_and_dates_are_fetched_lazily_and_correctly(repo: Path) -> None:
    commit(
        repo,
        {"content/people/leave.md": page("Leave v2")},
        "edit",
        date="2026-03-05T12:00:00+00:00",
    )
    item = next(i for i in connector(repo).items() if i.uri == "hb/leave.md")
    assert b"Leave v2" in item.read()
    assert item.modified() == datetime.fromisoformat("2026-03-05T12:00:00+00:00")
    pay = next(i for i in connector(repo).items() if i.uri == "hb/pay.md")
    assert pay.modified() == datetime.fromisoformat(
        "2026-01-10T10:00:00+00:00"
    )  # untouched since the first commit


def test_a_past_revision_shows_the_past_state(repo: Path) -> None:
    first = git(repo, "rev-parse", "HEAD")
    commit(repo, {"content/people/new.md": page("New"), "content/people/pay.md": None}, "later")
    assert sorted(i.uri for i in connector(repo, first).items()) == [
        "hb/leave.md",
        "hb/pay.md",
        "hb/private/salary.md",
    ]
    assert sorted(i.uri for i in connector(repo).items()) == [
        "hb/leave.md",
        "hb/new.md",
        "hb/private/salary.md",
    ]


def test_visibility_rules_apply_first_match_wins_else_the_default(repo: Path) -> None:
    c = connector(
        repo, visibility_rules=[("private/", "private"), ("", "public")], default_visibility="org"
    )
    assert {i.uri: i.visibility for i in c.items()} == {
        "hb/leave.md": "public",
        "hb/pay.md": "public",
        "hb/private/salary.md": "private",
    }
    assert {i.visibility for i in connector(repo).items()} == {"org"}


@pytest.mark.parametrize(
    ("rev", "prefix"),
    [("--upload-pack=evil", ""), ("HEAD", "--output=/tmp/x"), ("no-such-branch", "")],
)
def test_unsafe_or_unknown_revisions_and_paths_are_refused(
    repo: Path, rev: str, prefix: str
) -> None:
    with pytest.raises(GitError):
        GitConnector(repo, rev, name="x", path_prefix=prefix)


def test_a_folder_that_is_not_a_repository_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        GitConnector(tmp_path, "HEAD", name="x")


# ---- planning (pure) --------------------------------------------------------------------------------
def item(uri: str, h: str, vis: str = "org") -> SourceItem:
    return SourceItem(uri, h, lambda: b"x", lambda: None, vis)


def test_the_plan_sorts_every_page_into_exactly_one_group() -> None:
    known = {
        "same": ("same", "111", "org", uuid.uuid4(), "active"),
        "edited": ("edited", "222", "org", uuid.uuid4(), "active"),
        "gone": ("gone", "333", "org", uuid.uuid4(), "active"),
        "was-deleted": ("was-deleted", "444", "org", None, "deleted"),
        "locked": ("locked", "555", "org", uuid.uuid4(), "active"),
        "long-gone": ("long-gone", "666", "org", None, "deleted"),
    }
    current = {
        u.uri: u
        for u in [
            item("same", "111"),
            item("edited", "999"),
            item("new", "777"),
            item("was-deleted", "444"),
            item("locked", "555", "private"),
        ]
    }  # noqa: E501
    added, changed, permissions, deleted, unchanged = plan_sync(known, current)
    assert sorted(i.uri for i in added) == [
        "new",
        "was-deleted",
    ]  # a deleted page that returns is added again
    assert [i.uri for i in changed] == ["edited"] and [i.uri for i in permissions] == ["locked"]
    assert (
        deleted == ["gone"] and unchanged == 1
    )  # "long-gone" was already deleted: not reported again


# ---- the sync, against the real database -----------------------------------------------------------
class Rig:
    def __init__(self, db_schema: Any, repo: Path) -> None:
        self.w = World(db_schema)
        self.ann = self.w.person("acme", "ann")
        self.repo = repo
        self.settings = Settings(_env_file=None, retrieval_candidates=10)  # type: ignore[call-arg]
        self.ingested: list[str] = []
        self.fail: set[str] = set()
        real = make_pipeline_ingest(
            db_schema,
            self.ann.org_id,
            self.ann.user_id,
            settings=self.settings,
            embedder=FakeEmbedder(),
            count=COUNT,
        )  # type: ignore[arg-type]

        def ingest(it: SourceItem, raw: bytes) -> uuid.UUID | None:
            if it.uri in self.fail:
                raise RuntimeError("the pipeline broke on this page")
            self.ingested.append(it.uri)
            return real(it, raw)

        self.ingest = ingest

    def sync(self, rev: str = "HEAD", **kw: Any):  # type: ignore[no-untyped-def]
        sync_kw = {k: kw.pop(k) for k in ("dry_run", "allow_mass_delete") if k in kw}
        return sync_connector(
            self.w.conn,
            connector(self.repo, rev, **kw),
            org_id=self.ann.org_id,
            owner_id=self.ann.user_id,
            ingest=self.ingest,
            **sync_kw,
        )

    def sql(self, query: str, *args: Any) -> list[tuple[Any, ...]]:
        return self.w.conn.execute(query, args).fetchall()


@pytest.fixture
def rig(db_schema, repo: Path) -> Rig:  # type: ignore[no-untyped-def]
    return Rig(db_schema, repo)


def test_the_first_sync_adds_every_page_with_chunks_and_the_sources_own_date(rig: Rig) -> None:
    report = rig.sync()
    assert (
        sorted(report.added) == ["hb/leave.md", "hb/pay.md", "hb/private/salary.md"]
        and not report.failed
    )
    docs = rig.sql(
        "SELECT source_uri, status, active_version, source_updated_at FROM documents ORDER BY source_uri"
    )
    assert [d[:3] for d in docs] == [
        ("hb/leave.md", "ready", 1),
        ("hb/pay.md", "ready", 1),
        ("hb/private/salary.md", "ready", 1),
    ]
    assert all(d[3] == datetime.fromisoformat("2026-01-10T10:00:00+00:00") for d in docs)
    assert rig.sql("SELECT count(*) FROM chunks WHERE level = 0")[0][0] >= 3


def test_syncing_again_reads_nothing_and_changes_nothing(rig: Rig) -> None:
    rig.sync()
    first = len(rig.ingested)
    report = rig.sync()
    assert (
        len(rig.ingested) == first
        and report.unchanged == 3
        and not (report.added or report.changed or report.deleted)
    )


def test_only_the_edited_page_is_re_indexed_as_a_new_version(rig: Rig) -> None:
    rig.sync()
    rig.ingested.clear()
    commit(
        rig.repo,
        {"content/people/leave.md": page("Leave NEW"), "content/people/pay.md": page("Pay")},
        "edit",
        date="2026-02-01T09:00:00+00:00",
    )
    report = rig.sync()
    assert report.changed == ["hb/leave.md"] and rig.ingested == [
        "hb/leave.md"
    ]  # pay.md was rewritten with identical bytes: untouched
    assert rig.sql("SELECT active_version FROM documents WHERE source_uri = 'hb/leave.md'") == [
        (2,)
    ]
    assert {
        r[0]
        for r in rig.sql(
            "SELECT version FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.source_uri = 'hb/leave.md'"
        )
    } == {2}
    assert (
        "NEW"
        in rig.sql(
            "SELECT string_agg(text, ' ') FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.source_uri = 'hb/leave.md'"
        )[0][0]
    )
    assert rig.sql("SELECT source_updated_at FROM documents WHERE source_uri = 'hb/leave.md'")[0][
        0
    ] == datetime.fromisoformat("2026-02-01T09:00:00+00:00")


def test_a_page_removed_from_the_source_disappears_with_everything_derived_from_it(
    rig: Rig,
) -> None:
    rig.sync()
    doc = rig.sql("SELECT id FROM documents WHERE source_uri = 'hb/pay.md'")[0][0]
    chunk = rig.sql("SELECT id FROM chunks WHERE document_id = %s AND level = 0 LIMIT 1", doc)[0][0]
    rig.w.conn.execute(
        "INSERT INTO graph_edges (org_id, owner_id, visibility, chunk_id, subject, relation, object) VALUES (%s,%s,'org',%s,'pay team','uses','payroll tool')",
        (rig.ann.org_id, rig.ann.user_id, chunk),
    )  # noqa: E501
    commit(rig.repo, {"content/people/pay.md": None}, "remove pay")
    report = rig.sync()
    assert report.deleted == ["hb/pay.md"]
    assert rig.sql("SELECT count(*) FROM documents WHERE source_uri = 'hb/pay.md'")[0][0] == 0
    assert rig.sql("SELECT count(*) FROM chunks WHERE document_id = %s", doc)[0][0] == 0
    assert rig.sql("SELECT count(*) FROM graph_edges")[0][0] == 0  # its facts went with it
    assert rig.sql("SELECT state, document_id FROM connector_items WHERE uri = 'hb/pay.md'") == [
        ("deleted", None)
    ]


def test_a_renamed_page_is_a_delete_plus_an_add(rig: Rig) -> None:
    rig.sync()
    git(rig.repo, "mv", "content/people/pay.md", "content/people/salary-and-pay.md")
    git(rig.repo, "commit", "-q", "-m", "rename")
    report = rig.sync()
    assert report.deleted == ["hb/pay.md"] and report.added == ["hb/salary-and-pay.md"]


def test_a_permission_change_flips_visibility_without_re_embedding(rig: Rig) -> None:
    rig.sync(visibility_rules=[])
    rig.ingested.clear()
    report = rig.sync(
        visibility_rules=[("private/", "private")]
    )  # the source now restricts that folder
    assert (
        report.permission_changes == ["hb/private/salary.md"] and rig.ingested == []
    )  # nothing re-read or re-embedded
    assert rig.sql(
        "SELECT visibility FROM documents WHERE source_uri = 'hb/private/salary.md'"
    ) == [("private",)]
    assert {
        r[0]
        for r in rig.sql(
            "SELECT c.visibility FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.source_uri = 'hb/private/salary.md'"
        )
    } == {"private"}
    assert (
        rig.sync(visibility_rules=[("private/", "private")]).permission_changes == []
    )  # and it stays settled


def test_a_page_that_fails_is_reported_and_retried_next_time_while_the_rest_succeed(
    rig: Rig,
) -> None:
    rig.fail = {"hb/pay.md"}
    report = rig.sync()
    assert [u for u, _ in report.failed] == ["hb/pay.md"] and sorted(report.added) == [
        "hb/leave.md",
        "hb/private/salary.md",
    ]
    assert (
        rig.sql("SELECT count(*) FROM connector_items WHERE uri = 'hb/pay.md'")[0][0] == 0
    )  # no record: it will be tried again
    rig.fail = set()
    assert rig.sync().added == ["hb/pay.md"]


def test_the_mass_delete_guard_stops_a_sync_that_would_wipe_the_corpus(
    rig: Rig, repo: Path
) -> None:
    files = {f"content/people/p{i}.md": page(f"Page {i}") for i in range(30)}
    commit(repo, files, "many pages")
    rig.sync()
    commit(repo, {name: None for name in list(files)[:25]}, "a bad checkout lost most pages")
    report = rig.sync()
    assert "would delete 25" in report.aborted and not report.deleted
    assert rig.sql("SELECT count(*) FROM documents")[0][0] == 33  # nothing was removed
    forced = rig.sync(allow_mass_delete=True)
    assert len(forced.deleted) == 25 and rig.sql("SELECT count(*) FROM documents")[0][0] == 8


def test_a_dry_run_reports_the_plan_and_changes_nothing(rig: Rig) -> None:
    report = rig.sync(dry_run=True)
    assert len(report.added) == 3 and report.dry_run and rig.ingested == []
    assert (
        rig.sql("SELECT count(*) FROM documents")[0][0] == 0
        and rig.sql("SELECT count(*) FROM connector_runs")[0][0] == 0
    )


def test_every_real_sync_leaves_an_audit_row(rig: Rig) -> None:
    rig.sync()
    commit(rig.repo, {"content/people/leave.md": page("Leave 3")}, "edit")
    rig.sync()
    runs = rig.sql(
        "SELECT added, changed, unchanged, deleted, finished_at IS NOT NULL, length(source_ref) FROM connector_runs ORDER BY started_at"
    )
    assert runs == [(3, 0, 0, 0, True, 40), (0, 1, 2, 0, True, 40)]


def test_two_connectors_in_one_org_do_not_touch_each_others_pages(rig: Rig) -> None:
    rig.sync()
    other = GitConnector(
        rig.repo, "HEAD", name="git:other", path_prefix="other", uri_prefix="other"
    )
    report = sync_connector(
        rig.w.conn, other, org_id=rig.ann.org_id, owner_id=rig.ann.user_id, ingest=rig.ingest
    )
    assert (
        report.added == ["other/x.md"] and not report.deleted
    )  # the first connector's pages are not "deleted" by this one
    assert rig.sql("SELECT count(*) FROM documents")[0][0] == 4


def test_search_results_carry_the_sources_last_updated_date_for_citations(rig: Rig) -> None:
    from app.retrieval.search import Scope, keyword_search

    commit(
        rig.repo,
        {"content/people/leave.md": page("Leave zebra")},
        "edit",
        date="2026-03-05T12:00:00+00:00",
    )
    rig.sync()
    scope = Scope(rig.ann.org_id, rig.ann.user_id, None)  # type: ignore[arg-type]
    hit = next(
        c
        for c in keyword_search(rig.w.conn, "zebra policy", scope)
        if c.source_uri == "hb/leave.md"
    )
    assert hit.updated == datetime.fromisoformat("2026-03-05T12:00:00+00:00")
