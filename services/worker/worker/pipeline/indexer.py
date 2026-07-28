"""Stage I/II orchestrator: clone -> filter -> parse -> resolve -> chunk -> persist.

This is the worker-side wiring of the domain algorithms and infra adapters. It
owns no parsing or resolution logic of its own -- that all lives behind ports --
and its single job is to sequence them incrementally and honestly report what
happened (ARCHITECTURE sec. 4.1-4.2).

The incremental contract it implements, end to end:

1. list the tree at ``commit_sha`` and cheaply pre-filter by path and size, so
   excluded blobs are never fetched;
2. diff the surviving ``path -> blob_sha`` map against the previous snapshot;
3. for changed files, fetch, content-filter, parse and chunk; for unchanged
   files, take the parsed unit straight from the content-addressed cache;
4. re-resolve the symbol graph *globally* (cheap, pure, and always fresh, so a
   moved symbol never leaves a stale edge behind);
5. persist the snapshot and report resolution, chunk-reuse, and timing metrics.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from app.domain.indexing.chunking import chunk_parsed_file
from app.domain.indexing.filtering import DEFAULT_MAX_BYTES, FileFilter
from app.domain.indexing.incremental import IndexPlan, plan_index
from app.domain.indexing.models import (
    Chunk,
    Language,
    ParsedUnit,
    SourceFile,
    Symbol,
)
from app.domain.indexing.ports import (
    FileEntry,
    IndexStorePort,
    ParseCachePort,
    ParserPort,
    SnapshotWrite,
    SourceProviderPort,
)
from app.domain.indexing.resolution import ResolutionStats, Resolver

__all__ = ["IndexResult", "Indexer"]

_GITIGNORE_PATH = ".gitignore"
_DEFAULT_PARSER_VERSION = "treesitter/1"
# M1 has no embeddings (that is M2); snapshots record which model *will* embed
# their chunks so the schema's NOT NULL column is satisfied honestly.
_NO_EMBEDDING = "none"


@dataclass(frozen=True, slots=True)
class IndexResult:
    """The report of one indexing run. Every field is a measured fact -- these are
    what the M1 exit criteria are checked against, not asserted."""

    snapshot_id: UUID
    repository_id: UUID
    commit_sha: str
    plan: IndexPlan
    files_indexed: int
    files_excluded: int
    files_parsed: int
    files_reused: int
    symbols: int
    edges: int
    chunks_total: int
    chunks_new: int
    chunks_reused: int
    resolution: ResolutionStats
    duration_ms: int


class Indexer:
    """Indexes a repository at a commit into the symbol graph and chunk store."""

    def __init__(
        self,
        *,
        source: SourceProviderPort,
        parsers: Mapping[Language, ParserPort],
        cache: ParseCachePort,
        store: IndexStorePort,
        max_bytes: int | None = None,
        parser_version: str = _DEFAULT_PARSER_VERSION,
        embedding_model: str = _NO_EMBEDDING,
    ) -> None:
        self._source = source
        self._parsers = dict(parsers)
        self._cache = cache
        self._store = store
        self._max_bytes = max_bytes
        self._parser_version = parser_version
        self._embedding_model = embedding_model

    async def index(
        self, *, repository_id: UUID, repo_url: str, commit_sha: str
    ) -> IndexResult:
        started = time.perf_counter()

        entries = await self._source.list_files(repo_url, commit_sha)
        sizes = {entry.path: entry.size_bytes for entry in entries}
        file_filter = await self._build_filter(repo_url, entries)

        current_blobs: dict[str, str] = {
            entry.path: entry.blob_sha
            for entry in entries
            if file_filter.should_fetch(entry.path, entry.size_bytes)
        }
        plan = await self._plan(repository_id, current_blobs)

        units: list[ParsedUnit] = []
        excluded = 0
        reused = 0
        to_fetch: list[str] = [*plan.added, *plan.modified]

        # Unchanged files: reuse the parsed unit (symbols + chunks) from cache.
        for path in plan.unchanged:
            unit = await self._cache.get(repository_id, current_blobs[path])
            if unit is not None:
                units.append(unit)
                reused += 1
            else:  # cold cache (e.g. fresh store): fall back to a full parse
                to_fetch.append(path)

        for path in to_fetch:
            unit = await self._index_file(
                repository_id, repo_url, path, current_blobs[path], file_filter, sizes
            )
            if unit is None:
                excluded += 1
            else:
                units.append(unit)

        parsed_files = [unit.parsed for unit in units]
        resolution = Resolver(parsed_files).resolve()

        symbols: list[Symbol] = [s for pf in parsed_files for s in pf.symbols]
        chunks: list[Chunk] = [c for unit in units for c in unit.chunks]
        files: list[SourceFile] = [pf.file for pf in parsed_files]
        known_hashes = await self._store.known_chunk_hashes(repository_id)
        chunks_new = sum(1 for c in chunks if c.content_hash not in known_hashes)

        snapshot = SnapshotWrite(
            id=uuid4(),
            repository_id=repository_id,
            commit_sha=commit_sha,
            parent_snapshot_id=await self._parent_snapshot_id(repository_id),
            parser_version=self._parser_version,
            embedding_model=self._embedding_model,
            files=tuple(files),
            symbols=tuple(symbols),
            edges=resolution.edges,
            chunks=tuple(chunks),
        )
        await self._store.save(snapshot)

        return IndexResult(
            snapshot_id=snapshot.id,
            repository_id=repository_id,
            commit_sha=commit_sha,
            plan=plan,
            files_indexed=len(files),
            files_excluded=excluded,
            files_parsed=len(to_fetch) - excluded,
            files_reused=reused,
            symbols=len(symbols),
            edges=len(resolution.edges),
            chunks_total=len(chunks),
            chunks_new=chunks_new,
            chunks_reused=len(chunks) - chunks_new,
            resolution=resolution.stats,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )

    # -- steps ----------------------------------------------------------- #

    async def _index_file(
        self,
        repository_id: UUID,
        repo_url: str,
        path: str,
        blob_sha: str,
        file_filter: FileFilter,
        sizes: Mapping[str, int],
    ) -> ParsedUnit | None:
        """Fetch, content-filter, parse and chunk one file. ``None`` means the
        content check excluded it (binary, minified, generated)."""
        content = await self._source.read_blob(repo_url, blob_sha)
        decision = file_filter.evaluate(path, len(content), content)
        if not decision.included or decision.language is None:
            return None

        cached = await self._cache.get(repository_id, blob_sha)
        if cached is not None and cached.parsed.file.path == path:
            return cached

        source_file = SourceFile(
            path=path,
            language=decision.language,
            blob_sha=blob_sha,
            size_bytes=len(content),
            line_count=content.count(b"\n") + 1,
            is_test=decision.is_test,
            is_generated=decision.is_generated,
        )
        parser = self._parsers[decision.language]
        parsed = parser.parse(source_file, content)
        text = content.decode("utf-8", "replace")
        chunks = chunk_parsed_file(repository_id, parsed, text)
        unit = ParsedUnit(parsed=parsed, chunks=tuple(chunks))
        await self._cache.put(repository_id, unit)
        return unit

    async def _build_filter(
        self, repo_url: str, entries: Sequence[FileEntry]
    ) -> FileFilter:
        gitignore_blob = next(
            (e for e in entries if e.path == _GITIGNORE_PATH), None
        )
        max_bytes = self._max_bytes or DEFAULT_MAX_BYTES
        if gitignore_blob is None:
            return FileFilter(max_bytes=max_bytes)
        raw = await self._source.read_blob(repo_url, gitignore_blob.blob_sha)
        return FileFilter.from_gitignore_text(
            raw.decode("utf-8", "replace"), max_bytes=max_bytes
        )

    async def _plan(
        self, repository_id: UUID, current_blobs: Mapping[str, str]
    ) -> IndexPlan:
        latest = await self._store.latest_snapshot(repository_id)
        previous: Mapping[str, str] = (
            await self._store.snapshot_file_blobs(latest.id) if latest else {}
        )
        return plan_index(current_blobs, previous)

    async def _parent_snapshot_id(self, repository_id: UUID) -> UUID | None:
        latest = await self._store.latest_snapshot(repository_id)
        return latest.id if latest is not None else None
