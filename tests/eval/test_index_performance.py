"""The M1 latency exit gate: cold index of a 5k-file repo, and a one-file push.

ROADMAP.md M1: "index a 5k-file repo in < 4 min cold; a single-file push
re-indexes in < 10 s." Like the resolution rate, these are *measured* here
against a real git repository built on the fly -- real clone, real tree-sitter
parsing, real incremental plan -- and the measurement is printed so a run that
merely squeaks under budget is visible before it starts failing.

The incremental number is the one that matters architecturally. Cold indexing is
a batch job nobody watches; the freshness SLO (index within 60 s of a push) is
paid every push, and it is only affordable because the parse cache is keyed by
blob sha: one changed file is one cache miss, and the other 4,999 files are
served from cache without being fetched or parsed at all.

Marked ``slow`` and deselected by default (see ``pyproject.toml``); run the gate
with ``pytest -m slow``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest

from app.domain.indexing.ports import IndexStorePort, ParseCachePort
from app.infra.parsing.registry import default_parsers
from app.infra.source.git_source import GitSourceProvider
from app.infra.store.memory import InMemoryIndexStore
from app.infra.store.postgres import PostgresIndexStore
from tests.conftest import GitRepo
from tests.pg import create_repository
from worker.pipeline.indexer import Indexer, IndexResult

pytestmark = pytest.mark.slow

SCALE_FILES = 5_000
COLD_BUDGET_SECONDS = 4 * 60
INCREMENTAL_BUDGET_SECONDS = 10

_PACKAGES = 50
"""Files are spread over packages so the symbol graph has real cross-module
edges to resolve; one flat directory of 5k modules would understate the
resolver's share of the cost."""


def _python_module(package: int, index: int) -> str:
    """A module that imports a sibling, defines a class and functions, and calls
    across package boundaries -- so indexing it costs what real code costs."""
    neighbour = (index + 1) % (SCALE_FILES // _PACKAGES)
    return f'''"""Generated module {package}/{index}."""

from pkg{package}.mod{neighbour} import Helper{package}_{neighbour}


class Helper{package}_{index}:
    """Does a small amount of work."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def compute(self, value: int) -> int:
        return self._seed + value

    def delegate(self, value: int) -> int:
        other = Helper{package}_{neighbour}(self._seed)
        return other.compute(value)


def build{index}(seed: int) -> Helper{package}_{index}:
    return Helper{package}_{index}(seed)


def run{index}(value: int) -> int:
    return build{index}(1).delegate(value)
'''


def _typescript_module(package: int, index: int) -> str:
    neighbour = (index + 1) % (SCALE_FILES // _PACKAGES)
    return f"""import {{ Helper{package}_{neighbour} }} from "./mod{neighbour}";

export class Helper{package}_{index} {{
  constructor(private readonly seed: number) {{}}

  compute(value: number): number {{
    return this.seed + value;
  }}

  delegate(value: number): number {{
    const other = new Helper{package}_{neighbour}(this.seed);
    return other.compute(value);
  }}
}}

export function build{index}(seed: number): Helper{package}_{index} {{
  return new Helper{package}_{index}(seed);
}}
"""


def _generate_repo_files() -> dict[str, str]:
    """A 5k-file repo, four fifths Python and one fifth TypeScript."""
    per_package = SCALE_FILES // _PACKAGES
    files: dict[str, str] = {}
    for package in range(_PACKAGES):
        for index in range(per_package):
            if index % 5 == 4:
                files[f"web{package}/mod{index}.ts"] = _typescript_module(
                    package, index
                )
            else:
                files[f"pkg{package}/mod{index}.py"] = _python_module(package, index)
        files[f"pkg{package}/__init__.py"] = f'"""Package {package}."""\n'
    return files


def _indexer(
    store: InMemoryIndexStore,
    source: GitSourceProvider,
    index_store: IndexStorePort | None = None,
) -> Indexer:
    """``store`` always backs the parse cache; ``index_store`` overrides where
    the snapshot is persisted, so the same run can be re-timed against Postgres
    without changing anything else about it."""
    cache: ParseCachePort = store
    return Indexer(
        source=source,
        parsers=default_parsers(),
        cache=cache,
        store=index_store or store,
    )


def _report(label: str, result: IndexResult, seconds: float, budget: int) -> None:
    print(
        f"\n[{label}] {seconds:.1f}s of a {budget}s budget | "
        f"files indexed {result.files_indexed} "
        f"(parsed {result.files_parsed}, reused from cache {result.files_reused}) | "
        f"symbols {result.symbols} edges {result.edges} | "
        f"chunks {result.chunks_total} (new {result.chunks_new}, "
        f"reused {result.chunks_reused}) | "
        f"resolution rate {result.resolution.resolution_rate:.2%}"
    )


class TestIndexingLatency:
    """Both criteria in one test: the incremental number is only meaningful
    against a store and cache warmed by a real cold run of the same repo."""

    async def test_cold_and_incremental_indexing_meet_budget(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        files = _generate_repo_files()
        assert len(files) >= SCALE_FILES
        head = repo.commit(files, "generated corpus")

        store = InMemoryIndexStore()
        source = GitSourceProvider()
        repository_id = uuid4()

        started = time.perf_counter()
        cold = await _indexer(store, source).index(
            repository_id=repository_id, repo_url=repo.url, commit_sha=head
        )
        cold_seconds = time.perf_counter() - started
        _report("cold", cold, cold_seconds, COLD_BUDGET_SECONDS)

        assert cold.files_indexed >= SCALE_FILES
        assert cold_seconds < COLD_BUDGET_SECONDS, (
            f"cold index of {cold.files_indexed} files took {cold_seconds:.1f}s, "
            f"over the {COLD_BUDGET_SECONDS}s M1 budget"
        )

        # A single-file push: one blob changes, everything else must come back
        # from the content-addressed cache rather than being re-parsed.
        touched = "pkg0/mod0.py"
        head = repo.commit(
            {touched: files[touched] + "\n\ndef added(value: int) -> int:\n    return value\n"},
            "single-file push",
        )

        started = time.perf_counter()
        warm = await _indexer(store, source).index(
            repository_id=repository_id, repo_url=repo.url, commit_sha=head
        )
        warm_seconds = time.perf_counter() - started
        _report("incremental", warm, warm_seconds, INCREMENTAL_BUDGET_SECONDS)

        assert warm.plan.modified == (touched,)
        assert warm.files_parsed == 1, "only the changed blob may be re-parsed"
        assert warm.files_reused == cold.files_indexed - 1
        assert warm.chunks_reused > 0, "unchanged chunks must not be re-produced"
        assert warm_seconds < INCREMENTAL_BUDGET_SECONDS, (
            f"single-file re-index took {warm_seconds:.1f}s, over the "
            f"{INCREMENTAL_BUDGET_SECONDS}s M1 budget"
        )


class TestPostgresIndexingLatency:
    """The same two budgets, with snapshots persisted to Postgres.

    ``TestIndexingLatency`` above measures parsing, resolution and chunking with
    an in-memory store, so it says nothing about what persistence costs -- and
    persistence is where a naive adapter would blow the budget, because a cold
    index of this repository writes roughly 90,000 rows across four tables. The
    store loads them with ``COPY`` into temp tables and four ``INSERT ... SELECT``
    statements for exactly that reason; this is the measurement that says whether
    the reason was real.

    The parse cache stays in memory (it is Redis in production, not Postgres),
    so the delta between the two classes is persistence and nothing else.

    Skips without ``ARGUS_TEST_DATABASE_URL``; see ``tests/pg.py``.
    """

    async def test_cold_and_incremental_indexing_meet_budget(
        self, make_git_repo: Callable[[], GitRepo], pg_pool: Any
    ) -> None:
        repo = make_git_repo()
        files = _generate_repo_files()
        head = repo.commit(files, "generated corpus")

        cache = InMemoryIndexStore()
        source = GitSourceProvider()
        repository_id = await create_repository(pg_pool)
        store = PostgresIndexStore(pg_pool)

        started = time.perf_counter()
        cold = await _indexer(cache, source, store).index(
            repository_id=repository_id, repo_url=repo.url, commit_sha=head
        )
        cold_seconds = time.perf_counter() - started
        _report("cold (postgres)", cold, cold_seconds, COLD_BUDGET_SECONDS)

        assert cold.files_indexed >= SCALE_FILES
        assert cold_seconds < COLD_BUDGET_SECONDS, (
            f"cold index of {cold.files_indexed} files took {cold_seconds:.1f}s "
            f"with Postgres persistence, over the {COLD_BUDGET_SECONDS}s budget"
        )

        touched = "pkg0/mod0.py"
        head = repo.commit(
            {touched: files[touched] + "\n\ndef added(value: int) -> int:\n    return value\n"},
            "single-file push",
        )

        started = time.perf_counter()
        warm = await _indexer(cache, source, store).index(
            repository_id=repository_id, repo_url=repo.url, commit_sha=head
        )
        warm_seconds = time.perf_counter() - started
        _report("incremental (postgres)", warm, warm_seconds, INCREMENTAL_BUDGET_SECONDS)

        assert warm.files_parsed == 1, "only the changed blob may be re-parsed"
        assert warm_seconds < INCREMENTAL_BUDGET_SECONDS, (
            f"single-file re-index took {warm_seconds:.1f}s with Postgres "
            f"persistence, over the {INCREMENTAL_BUDGET_SECONDS}s budget"
        )

        # The payoff the schema is shaped around: the second snapshot shares
        # almost every chunk row (and every embedding) with the first.
        distinct = await pg_pool.fetchval(
            "SELECT count(*) FROM chunks WHERE repository_id = $1", repository_id
        )
        reused = await pg_pool.fetchval(
            "SELECT chunks_reused FROM index_snapshots WHERE id = $1", warm.snapshot_id
        )
        print(
            f"[postgres] {reused} of {warm.chunks_total} chunks already stored; "
            f"{distinct} distinct chunk rows for the repository after two commits"
        )
        assert reused > 0, "the second index must reuse stored chunks"
