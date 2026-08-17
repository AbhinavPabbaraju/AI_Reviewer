"""Indexing a checkout as it is on disk, uncommitted edits and all.

The property that matters is narrow and easy to lose: a file edited but not
committed must be served with the content on disk and a blob sha derived from
that content. Serve the committed sha instead and the parse cache hits against
the wrong version, which produces stale symbols for precisely the file under
review -- a bug that would look like a bad model rather than a bad index.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.infra.source.working_tree import WorkingTreeSource
from tests.conftest import GitRepo

MODULE = "pkg/service.py"
ORIGINAL = "def handle(value):\n    return value + 1\n"
EDITED = "def handle(value):\n    return value + 2\n"


def _entries(entries: object) -> dict[str, str]:
    assert isinstance(entries, list | tuple)
    return {entry.path: entry.blob_sha for entry in entries}  # type: ignore[attr-defined]


class TestWorkingTreeSource:
    async def test_lists_tracked_files_at_their_committed_shas(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL, "README.md": "hi\n"})
        source = WorkingTreeSource(repo.path)

        entries = _entries(await source.list_files(repo.url, "HEAD"))
        assert set(entries) == {MODULE, "README.md"}
        assert await source.read_blob(repo.url, entries[MODULE]) == ORIGINAL.encode()

    async def test_an_uncommitted_edit_is_what_gets_served(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """The whole reason this adapter exists."""
        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL})
        (repo.path / MODULE).write_text(EDITED)

        source = WorkingTreeSource(repo.path)
        entries = _entries(await source.list_files(repo.url, "HEAD"))
        assert await source.read_blob(repo.url, entries[MODULE]) == EDITED.encode()

    async def test_an_edited_file_gets_a_different_blob_sha(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """The sha is the parse cache's key. If it did not move with the
        content, the cache would serve the committed parse for the edited file
        and the review would be of code nobody wrote."""
        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL})
        source = WorkingTreeSource(repo.path)
        committed = _entries(await source.list_files(repo.url, "HEAD"))[MODULE]

        (repo.path / MODULE).write_text(EDITED)
        edited = _entries(await source.list_files(repo.url, "HEAD"))[MODULE]

        assert edited != committed

    async def test_the_sha_is_gits_own_blob_hash(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """Computed in-process to avoid a subprocess per dirty file, so it has
        to agree with git or the two are different identifier spaces."""
        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL})
        (repo.path / MODULE).write_text(EDITED)

        source = WorkingTreeSource(repo.path)
        computed = _entries(await source.list_files(repo.url, "HEAD"))[MODULE]
        expected = repo._run("hash-object", MODULE).strip()
        assert computed == expected

    async def test_untracked_files_are_invisible(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """Scratch work is not part of the repository, and reviewing it would
        mean reviewing notes. git already decides this; we do not re-decide."""
        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL})
        (repo.path / "scratch.py").write_text("x = 1\n")

        source = WorkingTreeSource(repo.path)
        assert "scratch.py" not in _entries(await source.list_files(repo.url, "HEAD"))

    async def test_a_deleted_file_is_skipped_rather_than_raising(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """Tracked but gone from disk: a deletion in flight. The diff already
        describes it, and there is nothing left to index."""
        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL, "pkg/other.py": "y = 2\n"})
        (repo.path / MODULE).unlink()

        source = WorkingTreeSource(repo.path)
        entries = _entries(await source.list_files(repo.url, "HEAD"))
        assert MODULE not in entries
        assert "pkg/other.py" in entries

    async def test_sizes_come_from_disk(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """The size filter runs before content is read, so it has to describe
        the file being indexed rather than its committed ancestor."""
        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL})
        long_body = ORIGINAL + "# padding\n" * 100
        (repo.path / MODULE).write_text(long_body)

        source = WorkingTreeSource(repo.path)
        entries = await source.list_files(repo.url, "HEAD")
        [entry] = [e for e in entries if e.path == MODULE]
        assert entry.size_bytes == len(long_body.encode())


class TestWorkingTreeHeadFiles:
    async def test_reads_the_file_on_disk(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        from app.infra.review.head_files import WorkingTreeHeadFiles

        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL})
        (repo.path / MODULE).write_text(EDITED)

        files = WorkingTreeHeadFiles(repo.path)
        assert await files.read(MODULE) == EDITED

    async def test_a_missing_file_reads_as_absent(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        from app.infra.review.head_files import WorkingTreeHeadFiles

        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL})
        assert await WorkingTreeHeadFiles(repo.path).read("pkg/nope.py") is None

    async def test_paths_cannot_escape_the_root(
        self, make_git_repo: Callable[[], GitRepo], tmp_path: Path
    ) -> None:
        """``CodeSpan`` rejects traversal at the type boundary, but this is
        where a path derived from model output becomes a filesystem read, and
        one check at the boundary beats being sure about every path upstream."""
        from app.infra.review.head_files import WorkingTreeHeadFiles

        repo = make_git_repo()
        repo.commit({MODULE: ORIGINAL})
        secret = tmp_path / "secret.txt"
        secret.write_text("password\n")

        files = WorkingTreeHeadFiles(repo.path)
        assert await files.read("../secret.txt") is None
