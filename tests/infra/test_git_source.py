"""The git-backed source provider, against real throwaway repositories."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.infra.source.git_source import GitCommandError, GitSourceProvider
from tests.conftest import GitRepo


class TestListFiles:
    async def test_lists_blobs_with_shas_and_sizes(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        sha = repo.commit(
            {"app/main.py": "print(1)\n", "README.md": "# hi\n"}, "init"
        )
        provider = GitSourceProvider()
        entries = {e.path: e for e in await provider.list_files(repo.url, sha)}
        assert set(entries) == {"app/main.py", "README.md"}
        assert entries["app/main.py"].size_bytes == len("print(1)\n")
        assert len(entries["app/main.py"].blob_sha) == 40

    async def test_reflects_commit_history(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        first = repo.commit({"a.py": "1\n"}, "first")
        second = repo.commit({"b.py": "2\n"}, "second")
        at_first = {e.path for e in await GitSourceProvider().list_files(repo.url, first)}
        provider = GitSourceProvider()
        at_second = {e.path for e in await provider.list_files(repo.url, second)}
        assert at_first == {"a.py"}
        assert at_second == {"a.py", "b.py"}

    async def test_blob_sha_changes_with_content(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        first = repo.commit({"a.py": "one\n"}, "first")
        second = repo.commit({"a.py": "two\n"}, "second")
        provider = GitSourceProvider()
        s1 = (await provider.list_files(repo.url, first))[0].blob_sha
        s2 = (await provider.list_files(repo.url, second))[0].blob_sha
        assert s1 != s2


class TestReadBlob:
    async def test_reads_content_by_sha(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        sha = repo.commit({"app/x.py": "hello world\n"}, "init")
        provider = GitSourceProvider()
        entry = (await provider.list_files(repo.url, sha))[0]
        content = await provider.read_blob(repo.url, entry.blob_sha)
        assert content == b"hello world\n"

    async def test_missing_object_raises(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        repo.commit({"a.py": "1\n"}, "init")
        provider = GitSourceProvider()
        with pytest.raises(GitCommandError):
            await provider.read_blob(repo.url, "0" * 40)


class TestCloneReuse:
    async def test_clone_is_reused_across_calls(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        sha = repo.commit({"a.py": "1\n"}, "init")
        provider = GitSourceProvider()
        # two calls must not fail on re-clone into the same directory
        await provider.list_files(repo.url, sha)
        again = await provider.list_files(repo.url, sha)
        assert again[0].path == "a.py"
