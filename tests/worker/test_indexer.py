"""End-to-end indexing against real git repos, incrementally (sec. 4.1-4.2)."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from app.infra.parsing.registry import default_parsers
from app.infra.source.git_source import GitSourceProvider
from app.infra.store.memory import InMemoryIndexStore
from tests.conftest import GitRepo
from worker.pipeline.indexer import Indexer

_APP = {
    "app/util.py": "def helper(x):\n    return x * 2\n",
    "app/store.py": (
        "from app.util import helper\n\n"
        "class Store:\n"
        "    def save(self, item):\n"
        "        return helper(item)\n"
        "    def load(self, key):\n"
        "        return self.save(key)\n"
    ),
}


def _indexer(store: InMemoryIndexStore) -> Indexer:
    return Indexer(
        source=GitSourceProvider(),
        parsers=default_parsers(),
        cache=store,
        store=store,
    )


class TestBasicIndex:
    async def test_indexes_and_resolves(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        sha = repo.commit(_APP, "init")
        store = InMemoryIndexStore()
        result = await _indexer(store).index(
            repository_id=uuid4(), repo_url=repo.url, commit_sha=sha
        )
        assert result.files_indexed == 2
        assert result.symbols > 0
        assert result.edges > 0
        assert result.resolution.resolution_rate == 1.0
        assert result.chunks_total == result.chunks_new  # first run: nothing reused

    async def test_excludes_noise(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        sha = repo.commit(
            {
                **_APP,
                ".gitignore": "secret.py\n",
                "secret.py": "KEY = 'x'\n",
                "node_modules/dep.py": "x = 1\n",
                "data.bin": b"\x00\x01binary",
                "README.md": "# docs\n",
            },
            "init",
        )
        store = InMemoryIndexStore()
        result = await _indexer(store).index(
            repository_id=uuid4(), repo_url=repo.url, commit_sha=sha
        )
        indexed = {f.path for f in store.snapshot(result.snapshot_id).files}
        assert indexed == {"app/util.py", "app/store.py"}


class TestIncremental:
    async def test_single_file_change_reuses_the_rest(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        repo_id = uuid4()
        store = InMemoryIndexStore()
        indexer = _indexer(store)
        first = repo.commit(_APP, "init")
        await indexer.index(repository_id=repo_id, repo_url=repo.url, commit_sha=first)

        changed = dict(_APP)
        changed["app/store.py"] = changed["app/store.py"].replace(
            "return helper(item)", "return helper(item) + 1"
        )
        second = repo.commit(changed, "tweak")
        result = await indexer.index(
            repository_id=repo_id, repo_url=repo.url, commit_sha=second
        )
        assert result.plan.modified == ("app/store.py",)
        assert result.plan.unchanged == ("app/util.py",)
        assert result.files_parsed == 1  # only the changed file re-parsed
        assert result.files_reused == 1
        assert result.chunks_reused > 0  # util's chunks carried over by hash

    async def test_reindexing_same_commit_reuses_everything(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        repo_id = uuid4()
        store = InMemoryIndexStore()
        indexer = _indexer(store)
        sha = repo.commit(_APP, "init")
        await indexer.index(repository_id=repo_id, repo_url=repo.url, commit_sha=sha)
        result = await indexer.index(
            repository_id=repo_id, repo_url=repo.url, commit_sha=sha
        )
        assert result.files_parsed == 0
        assert result.files_reused == result.files_indexed
        assert result.chunks_reused == result.chunks_total

    async def test_removed_file_leaves_the_snapshot(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        repo_id = uuid4()
        store = InMemoryIndexStore()
        indexer = _indexer(store)
        extended = {**_APP, "app/extra.py": "def gone():\n    return 0\n"}
        first = repo.commit(extended, "init")
        await indexer.index(repository_id=repo_id, repo_url=repo.url, commit_sha=first)
        repo.remove("app/extra.py")
        second = repo.commit({}, "drop extra")
        result = await indexer.index(
            repository_id=repo_id, repo_url=repo.url, commit_sha=second
        )
        assert result.plan.removed == ("app/extra.py",)
        indexed = {f.path for f in store.snapshot(result.snapshot_id).files}
        assert "app/extra.py" not in indexed

    async def test_cold_cache_falls_back_to_reparse(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """A warm store but cold parse cache (e.g. Redis eviction) must still
        produce a correct, complete snapshot -- just without the reuse win."""
        repo = make_git_repo()
        repo_id = uuid4()
        store = InMemoryIndexStore()
        sha = repo.commit(_APP, "init")
        # First run populates store + cache together.
        await Indexer(
            source=GitSourceProvider(), parsers=default_parsers(),
            cache=InMemoryIndexStore(), store=store,
        ).index(repository_id=repo_id, repo_url=repo.url, commit_sha=sha)
        # Second run: same store (has the snapshot) but a brand-new empty cache.
        cold_cache = InMemoryIndexStore()
        result = await Indexer(
            source=GitSourceProvider(), parsers=default_parsers(),
            cache=cold_cache, store=store,
        ).index(repository_id=repo_id, repo_url=repo.url, commit_sha=sha)
        assert result.plan.unchanged == ("app/store.py", "app/util.py")
        assert result.files_reused == 0  # cache was cold
        assert result.files_indexed == 2  # but the snapshot is still complete
        assert result.resolution.resolution_rate == 1.0
