"""Ports for Stage I/II: source access, parsing, and index storage.

Same rule as ``app.domain.ports`` (M0): the domain declares what indexing needs;
``infra`` supplies it. Concretely this boundary buys three things --

* the resolver and chunker are unit-tested against hand-built ``ParsedFile``
  objects, with no tree-sitter grammar and no git in the loop;
* incrementality is a *content-addressed parse cache* (``ParseCachePort``) that a
  deployment can back with Redis while tests back it with a dict;
* the store of record (``IndexStorePort``) can be Postgres in production and an
  in-memory fake in the M1 tests, without the pipeline knowing the difference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

from app.domain.base import Frozen
from app.domain.indexing.models import (
    Chunk,
    Embedding,
    Language,
    ParsedFile,
    ParsedUnit,
    SourceFile,
    Symbol,
    SymbolEdge,
)

__all__ = [
    "EmbeddingCachePort",
    "FileEntry",
    "IndexStorePort",
    "ParseCachePort",
    "ParserPort",
    "SnapshotRef",
    "SnapshotWrite",
    "SourceProviderPort",
]


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One entry from a git tree listing. ``size_bytes`` comes from the tree
    (``git ls-tree -l``) so the size filter can reject a blob without ever
    fetching it -- a bare blobless clone has not downloaded the content yet."""

    path: str
    blob_sha: str
    size_bytes: int


@runtime_checkable
class SourceProviderPort(Protocol):
    """Read-only access to repository contents at a commit.

    Cloned code is untrusted input (ARCHITECTURE sec. 8): implementations clone
    bare, blobless, with no hooks and no submodule recursion, and never execute
    anything from the tree.
    """

    async def list_files(
        self, repo_url: str, commit_sha: str
    ) -> Sequence[FileEntry]:
        """Every blob reachable at ``commit_sha``, with sizes. Trees, symlinks
        and submodule gitlinks are excluded."""
        ...

    async def read_blob(self, repo_url: str, blob_sha: str) -> bytes:
        """Fetch one blob's raw bytes. Separated from :meth:`list_files` so the
        caller can filter on path and size before paying to download content."""
        ...


@runtime_checkable
class ParserPort(Protocol):
    """A per-language parser. Synchronous: parsing is CPU-bound and does no I/O,
    so making it ``async`` would only add ceremony."""

    @property
    def language(self) -> Language: ...

    def parse(self, file: SourceFile, source: bytes) -> ParsedFile:
        """Extract symbols, imports and (unresolved) references from one file.

        The parser assigns the module fqn and every symbol fqn -- these follow
        language-specific conventions it alone knows -- but it does **not**
        resolve references; that is the language-agnostic resolver's job.
        """
        ...


@runtime_checkable
class ParseCachePort(Protocol):
    """Content-addressed parse cache: the engine of incremental re-indexing.

    Keyed by ``blob_sha`` (the git object id), which *is* a content hash, so a
    cache hit is always valid and a changed file is always a miss. A single-file
    push therefore re-parses exactly one blob (ARCHITECTURE sec. 4.1); every
    other file is a hit. The cached unit carries both the parsed structure and the
    file's chunks, so neither is recomputed on reuse.
    """

    async def get(self, repository_id: UUID, blob_sha: str) -> ParsedUnit | None: ...

    async def put(self, repository_id: UUID, unit: ParsedUnit) -> None:
        """Store ``unit`` under ``unit.parsed.file.blob_sha``."""
        ...


@runtime_checkable
class EmbeddingCachePort(Protocol):
    """Content-addressed embedding cache: the reason a re-index is nearly free.

    Keyed by ``(model, content_hash)`` -- the model belongs in the key because
    vectors from two models are not comparable, so an upgrade must miss rather
    than silently mix embedding spaces in one ANN index (ARCHITECTURE sec. 4.1).
    Unlike the parse cache this is worth persisting across repositories: the same
    vendored file in two repos hashes identically only if its path and repo id
    match, so sharing is safe by construction of the chunk hash.
    """

    async def get_many(
        self, model: str, content_hashes: Sequence[str]
    ) -> Mapping[str, Embedding]:
        """Cached vectors for the hashes that have one. Missing keys are absent
        from the result rather than None-valued -- the caller's next step is
        "embed the difference", and a sparse map makes that a set operation."""
        ...

    async def put_many(
        self, model: str, vectors: Mapping[str, Embedding]
    ) -> None: ...


class SnapshotRef(Frozen):
    """A pointer to a previously stored snapshot, used to diff file blobs."""

    id: UUID
    commit_sha: str


class SnapshotWrite(Frozen):
    """The complete graph for one indexing run, persisted atomically.

    ``chunks`` are every chunk in the snapshot (reused and freshly produced
    alike); the store deduplicates by ``content_hash``. ``embeddings`` maps
    ``content_hash -> vector`` and may be empty (an indexer configured without an
    embedder still writes a complete, queryable symbol graph -- the vectors only
    supplement it).
    """

    id: UUID
    repository_id: UUID
    commit_sha: str
    parent_snapshot_id: UUID | None
    parser_version: str
    embedding_model: str
    files: tuple[SourceFile, ...]
    symbols: tuple[Symbol, ...]
    edges: tuple[SymbolEdge, ...]
    chunks: tuple[Chunk, ...]
    embeddings: Mapping[str, Embedding] = {}


@runtime_checkable
class IndexStorePort(Protocol):
    """The index store of record. Backs incremental diffing and persistence."""

    async def latest_snapshot(self, repository_id: UUID) -> SnapshotRef | None:
        """The most recent ready snapshot for the repo, or ``None`` on first
        index."""
        ...

    async def snapshot_file_blobs(self, snapshot_id: UUID) -> Mapping[str, str]:
        """``path -> blob_sha`` for a prior snapshot, to compute the add/modify/
        remove plan against the current tree."""
        ...

    async def known_chunk_hashes(self, repository_id: UUID) -> AbstractSet[str]:
        """Every ``content_hash`` already stored for the repo, so the run can
        report how many chunks it reused versus produced."""
        ...

    async def save(self, snapshot: SnapshotWrite) -> None: ...
