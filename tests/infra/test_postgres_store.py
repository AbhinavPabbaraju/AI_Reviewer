"""``PostgresIndexStore`` against a real database, in parity with the fake.

The parity tests are the point of the file. ``InMemoryIndexStore`` is what the
M1 pipeline tests and the M6 eval harness run against, so every behaviour it has
that Postgres does not is a bug the test suite is structurally unable to see.
Asking both the same questions and comparing answers is what keeps the fake
honest -- and it is cheap, because they implement the same port.

The tests that are *not* parity tests cover things the fake has no opinion about
because it cannot: constraint enforcement, snapshot replacement, and whether a
re-index destroys embeddings that were already paid for.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from app.domain.indexing.ports import IndexStorePort, SnapshotWrite
from app.infra.store.memory import InMemoryIndexStore
from app.infra.store.postgres import PostgresIndexStore
from tests.eval.corpus.indexed import IndexedCorpus, build_corpus
from tests.eval.corpus.python_corpus import PY_CORPUS, PY_TEST_FILES
from tests.pg import create_repository


@pytest.fixture(scope="module")
async def corpus() -> IndexedCorpus:
    return await build_corpus(PY_CORPUS, PY_TEST_FILES, typescript=False)


@pytest.fixture
async def store(pg_pool: Any, corpus: IndexedCorpus) -> PostgresIndexStore:
    await create_repository(pg_pool, corpus.repository_id)
    return PostgresIndexStore(pg_pool)


def _rewrite(snapshot: SnapshotWrite, **changes: Any) -> SnapshotWrite:
    return snapshot.model_copy(update=changes)


class TestPortParity:
    """Both implementations answer every port method identically."""

    async def test_implements_the_port(self, store: PostgresIndexStore) -> None:
        assert isinstance(store, IndexStorePort)

    async def test_latest_snapshot_is_none_before_any_index(
        self, store: PostgresIndexStore, corpus: IndexedCorpus
    ) -> None:
        memory = InMemoryIndexStore()
        assert await store.latest_snapshot(corpus.repository_id) is None
        assert await memory.latest_snapshot(corpus.repository_id) is None

    async def test_snapshot_round_trips_identically(
        self, store: PostgresIndexStore, corpus: IndexedCorpus
    ) -> None:
        memory = InMemoryIndexStore()
        await store.save(corpus.snapshot)
        await memory.save(corpus.snapshot)

        pg_ref = await store.latest_snapshot(corpus.repository_id)
        mem_ref = await memory.latest_snapshot(corpus.repository_id)
        assert pg_ref is not None and mem_ref is not None
        assert pg_ref == mem_ref
        assert pg_ref.id == corpus.snapshot.id

        assert await store.snapshot_file_blobs(pg_ref.id) == (
            await memory.snapshot_file_blobs(mem_ref.id)
        )
        assert await store.known_chunk_hashes(corpus.repository_id) == (
            await memory.known_chunk_hashes(corpus.repository_id)
        )

    async def test_file_blobs_cover_every_indexed_file(
        self, store: PostgresIndexStore, corpus: IndexedCorpus
    ) -> None:
        await store.save(corpus.snapshot)
        blobs = await store.snapshot_file_blobs(corpus.snapshot.id)
        assert blobs == {
            file.path: file.blob_sha for file in corpus.snapshot.files
        }

    async def test_unknown_repository_has_no_chunks(
        self, store: PostgresIndexStore
    ) -> None:
        assert await store.known_chunk_hashes(uuid4()) == frozenset()


class TestSnapshotReplacement:
    """0001 makes (repo, commit, model, parser) unique; a repeat is a redo."""

    async def test_reindexing_a_commit_replaces_its_snapshot(
        self, store: PostgresIndexStore, corpus: IndexedCorpus, pg_pool: Any
    ) -> None:
        await store.save(corpus.snapshot)
        replacement = _rewrite(corpus.snapshot, id=uuid4())
        await store.save(replacement)

        count = await pg_pool.fetchval(
            "SELECT count(*) FROM index_snapshots WHERE repository_id = $1",
            corpus.repository_id,
        )
        assert count == 1
        latest = await store.latest_snapshot(corpus.repository_id)
        assert latest is not None and latest.id == replacement.id

    async def test_a_second_commit_is_a_second_snapshot(
        self, store: PostgresIndexStore, corpus: IndexedCorpus
    ) -> None:
        await store.save(corpus.snapshot)
        second = _rewrite(
            corpus.snapshot,
            id=uuid4(),
            commit_sha="1" * 40,
            parent_snapshot_id=corpus.snapshot.id,
        )
        await store.save(second)

        latest = await store.latest_snapshot(corpus.repository_id)
        assert latest is not None
        assert latest.id == second.id
        assert latest.commit_sha == "1" * 40
        # The first snapshot's file list is still readable: incremental indexing
        # diffs against it, so replacing it would break the next push.
        assert await store.snapshot_file_blobs(corpus.snapshot.id)

    async def test_replacement_keeps_embeddings_already_paid_for(
        self, store: PostgresIndexStore, corpus: IndexedCorpus, pg_pool: Any
    ) -> None:
        """The invariant 0002 exists to protect.

        An indexer configured without an embedder writes a complete symbol graph
        and *no* vectors. If that run replaced an embedded snapshot's chunks with
        NULL embeddings, a single unembedded re-index would silently disable
        semantic retrieval for the whole repository -- and nothing downstream
        would report an error, only worse packs.
        """
        await store.save(corpus.snapshot)
        embedded_before = await pg_pool.fetchval(
            "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
        )
        assert embedded_before > 0

        await store.save(
            _rewrite(corpus.snapshot, id=uuid4(), embeddings={})
        )

        assert (
            await pg_pool.fetchval(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            )
            == embedded_before
        )


class TestPersistedShape:
    """What actually landed in the tables."""

    async def test_every_symbol_edge_and_chunk_is_written(
        self, store: PostgresIndexStore, corpus: IndexedCorpus, pg_pool: Any
    ) -> None:
        await store.save(corpus.snapshot)
        snapshot = corpus.snapshot

        assert (
            await pg_pool.fetchval(
                "SELECT count(*) FROM files WHERE snapshot_id = $1", snapshot.id
            )
            == len(snapshot.files)
        )
        assert (
            await pg_pool.fetchval(
                "SELECT count(*) FROM symbols WHERE snapshot_id = $1", snapshot.id
            )
            == len(snapshot.symbols)
        )
        # Edges are scoped through their source symbol, which is what makes a
        # snapshot's graph self-contained without an edge-level snapshot column.
        assert (
            await pg_pool.fetchval(
                """
                SELECT count(*) FROM symbol_edges e
                JOIN symbols s ON s.id = e.src_symbol_id
                WHERE s.snapshot_id = $1
                """,
                snapshot.id,
            )
            == len(snapshot.edges)
        )
        assert (
            await pg_pool.fetchval(
                "SELECT count(*) FROM snapshot_chunks WHERE snapshot_id = $1",
                snapshot.id,
            )
            == len({chunk.content_hash for chunk in snapshot.chunks})
        )

    async def test_unresolved_edges_keep_their_textual_target(
        self, store: PostgresIndexStore, corpus: IndexedCorpus, pg_pool: Any
    ) -> None:
        """ARCHITECTURE sec. 4.2: an unresolved reference is kept, never dropped.
        Persistence is where that promise is easiest to break silently."""
        expected = sum(
            1 for edge in corpus.snapshot.edges if edge.dst_fqn is None
        )
        assert expected > 0, "corpus no longer exercises unresolved edges"
        await store.save(corpus.snapshot)
        stored = await pg_pool.fetchval(
            """
            SELECT count(*) FROM symbol_edges e
            JOIN symbols s ON s.id = e.src_symbol_id
            WHERE s.snapshot_id = $1
              AND e.dst_symbol_id IS NULL
              AND e.dst_unresolved_name IS NOT NULL
            """,
            corpus.snapshot.id,
        )
        assert stored == expected

    async def test_confidence_survives_the_round_trip(
        self, store: PostgresIndexStore, corpus: IndexedCorpus, pg_pool: Any
    ) -> None:
        """Confidence is what expansion weights neighbours by, so a column that
        rounded or defaulted it would degrade retrieval invisibly."""
        await store.save(corpus.snapshot)
        rows = await pg_pool.fetch(
            """
            SELECT DISTINCT e.confidence FROM symbol_edges e
            JOIN symbols s ON s.id = e.src_symbol_id
            WHERE s.snapshot_id = $1
            """,
            corpus.snapshot.id,
        )
        stored = {round(float(row["confidence"]), 4) for row in rows}
        expected = {round(edge.confidence, 4) for edge in corpus.snapshot.edges}
        assert stored == expected

    async def test_chunks_reused_is_measured_not_assumed(
        self, store: PostgresIndexStore, corpus: IndexedCorpus, pg_pool: Any
    ) -> None:
        """The DDL says ``chunks_reused`` proves incrementality works. First
        index reuses nothing; re-indexing the same content reuses all of it."""
        await store.save(corpus.snapshot)
        assert (
            await pg_pool.fetchval(
                "SELECT chunks_reused FROM index_snapshots WHERE id = $1",
                corpus.snapshot.id,
            )
            == 0
        )

        second = _rewrite(corpus.snapshot, id=uuid4(), commit_sha="2" * 40)
        await store.save(second)
        distinct = len({chunk.content_hash for chunk in corpus.snapshot.chunks})
        assert (
            await pg_pool.fetchval(
                "SELECT chunks_reused FROM index_snapshots WHERE id = $1", second.id
            )
            == distinct
        )


class TestTenancy:
    async def test_a_second_repository_sees_none_of_the_first(
        self, store: PostgresIndexStore, corpus: IndexedCorpus, pg_pool: Any
    ) -> None:
        await store.save(corpus.snapshot)
        other: UUID = await create_repository(pg_pool)
        assert await store.latest_snapshot(other) is None
        assert await store.known_chunk_hashes(other) == frozenset()
