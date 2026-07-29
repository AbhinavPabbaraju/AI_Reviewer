"""In-memory ``ParseCachePort`` + ``IndexStorePort`` for tests and the eval harness.

This is a real, complete implementation of the storage ports -- not a stub. It is
what the M1 pipeline tests run against, mirroring the architecture's philosophy of
exercising the whole pipeline through fakes (ARCHITECTURE sec. 3): a Postgres/
pgvector adapter will implement the same two protocols for production, and the
indexer will not be able to tell the difference. Keeping the store behind the port
is what makes that swap free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from uuid import UUID

from app.domain.indexing.models import Embedding, ParsedUnit
from app.domain.indexing.ports import SnapshotRef, SnapshotWrite

__all__ = ["InMemoryEmbeddingCache", "InMemoryIndexStore"]


class InMemoryIndexStore:
    """Backs both storage ports. Thread-unsafe by design: one index job owns it
    at a time, exactly as a per-job transaction would in Postgres."""

    def __init__(self) -> None:
        self._parsed: dict[UUID, dict[str, ParsedUnit]] = {}
        self._snapshots: dict[UUID, SnapshotWrite] = {}
        self._latest: dict[UUID, UUID] = {}
        self._chunk_hashes: dict[UUID, set[str]] = {}

    # -- ParseCachePort -------------------------------------------------- #

    async def get(self, repository_id: UUID, blob_sha: str) -> ParsedUnit | None:
        return self._parsed.get(repository_id, {}).get(blob_sha)

    async def put(self, repository_id: UUID, unit: ParsedUnit) -> None:
        self._parsed.setdefault(repository_id, {})[unit.blob_sha] = unit

    # -- IndexStorePort -------------------------------------------------- #

    async def latest_snapshot(self, repository_id: UUID) -> SnapshotRef | None:
        snapshot_id = self._latest.get(repository_id)
        if snapshot_id is None:
            return None
        snapshot = self._snapshots[snapshot_id]
        return SnapshotRef(id=snapshot.id, commit_sha=snapshot.commit_sha)

    async def snapshot_file_blobs(self, snapshot_id: UUID) -> Mapping[str, str]:
        snapshot = self._snapshots[snapshot_id]
        return {file.path: file.blob_sha for file in snapshot.files}

    async def known_chunk_hashes(self, repository_id: UUID) -> AbstractSet[str]:
        return frozenset(self._chunk_hashes.get(repository_id, set()))

    async def save(self, snapshot: SnapshotWrite) -> None:
        self._snapshots[snapshot.id] = snapshot
        self._latest[snapshot.repository_id] = snapshot.id
        hashes = self._chunk_hashes.setdefault(snapshot.repository_id, set())
        hashes.update(chunk.content_hash for chunk in snapshot.chunks)

    # -- test/eval introspection ---------------------------------------- #

    def snapshot(self, snapshot_id: UUID) -> SnapshotWrite:
        """Read back a persisted snapshot (tests assert on the stored graph)."""
        return self._snapshots[snapshot_id]


class InMemoryEmbeddingCache:
    """In-memory :class:`EmbeddingCachePort`, keyed by ``(model, hash)``.

    Counts its own lookups: "did the cache actually work" is an operational
    question the M2 budget depends on, and a cache nobody measures is a cache
    nobody notices has stopped working.
    """

    def __init__(self) -> None:
        self._vectors: dict[tuple[str, str], Embedding] = {}
        self.hits = 0
        self.misses = 0

    async def get_many(
        self, model: str, content_hashes: Sequence[str]
    ) -> Mapping[str, Embedding]:
        found: dict[str, Embedding] = {}
        for content_hash in content_hashes:
            vector = self._vectors.get((model, content_hash))
            if vector is None:
                self.misses += 1
            else:
                self.hits += 1
                found[content_hash] = vector
        return found

    async def put_many(self, model: str, vectors: Mapping[str, Embedding]) -> None:
        for content_hash, vector in vectors.items():
            self._vectors[(model, content_hash)] = vector

    def __len__(self) -> int:
        return len(self._vectors)
