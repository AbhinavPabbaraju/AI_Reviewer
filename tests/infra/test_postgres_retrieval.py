"""``PostgresSymbolIndex`` / ``PostgresVectorStore`` in parity with the fakes.

Same argument as ``test_postgres_store.py``, with one extra stake: the M2
retrieval-quality gate runs against ``InMemorySymbolIndex``, so any divergence
here is a divergence between the number the gate reports and the system that
actually ships. Parity is checked on the *ordered* results, not on sets, because
fusion breaks score ties by input order -- two adapters returning the same rows
in a different order build different packs at the budget boundary.

The non-parity tests cover the thing the fakes cannot express at all: a
repository accumulates one symbol row per commit indexed, so reads have to be
scoped to a snapshot as well as to a tenant.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.domain.contracts import CodeSpan
from app.domain.retrieval.ports import SymbolIndexPort
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore
from app.infra.retrieval.postgres import (
    PostgresSymbolIndex,
    PostgresVectorStore,
    latest_ready_snapshot,
)
from app.infra.store.postgres import PostgresIndexStore
from tests.eval.corpus.indexed import (
    EMBEDDING_DIMENSIONS,
    IndexedCorpus,
    build_corpus,
)
from tests.eval.corpus.python_corpus import PY_CORPUS, PY_TEST_FILES
from tests.pg import create_repository

# A handful of fqns that exercise every retrieval path: a class, a method, a
# module-level function, a module, and one that does not exist.
_PROBE_FQNS = (
    "shop.models.Entity",
    "shop.models.Entity.validate",
    "shop.store.memory.MemoryRepository.put",
    "shop.services.notifications.notify_all",
    "shop.models",
    "shop.does.not.exist",
)


@pytest.fixture(scope="module")
async def corpus() -> IndexedCorpus:
    return await build_corpus(PY_CORPUS, PY_TEST_FILES, typescript=False)


@pytest.fixture
async def persisted(pg_pool: Any, corpus: IndexedCorpus) -> IndexedCorpus:
    """The corpus, written to Postgres once per test. Both adapter fixtures
    depend on this rather than persisting it themselves, so a test can ask for
    the symbol index and the vector store together."""
    await create_repository(pg_pool, corpus.repository_id)
    await PostgresIndexStore(pg_pool).save(corpus.snapshot)
    return corpus


@pytest.fixture
async def indexes(
    pg_pool: Any, persisted: IndexedCorpus
) -> tuple[PostgresSymbolIndex, InMemorySymbolIndex]:
    postgres = await PostgresSymbolIndex.for_latest(pg_pool, persisted.repository_id)
    assert postgres is not None
    return postgres, InMemorySymbolIndex(persisted.snapshot)


@pytest.fixture
async def stores(
    pg_pool: Any, persisted: IndexedCorpus
) -> tuple[PostgresVectorStore, InMemoryVectorStore]:
    postgres = await PostgresVectorStore.for_latest(pg_pool, persisted.repository_id)
    assert postgres is not None
    return postgres, InMemoryVectorStore(persisted.snapshot)


class TestSymbolIndexParity:
    async def test_implements_the_port(
        self, indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex]
    ) -> None:
        postgres, _ = indexes
        assert isinstance(postgres, SymbolIndexPort)

    async def test_symbols_in_spans_match(
        self,
        indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex],
        corpus: IndexedCorpus,
    ) -> None:
        postgres, memory = indexes
        key = corpus.repository_key
        for fqn in _PROBE_FQNS[:-1]:
            span = corpus.span_of(fqn)
            assert list(await postgres.symbols_in_spans(key, [span])) == list(
                await memory.symbols_in_spans(key, [span])
            ), f"symbols_in_spans diverged for {fqn}"

    async def test_multi_hunk_spans_match(
        self,
        indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex],
        corpus: IndexedCorpus,
    ) -> None:
        """One query for N hunks, not N queries -- and the same answer."""
        postgres, memory = indexes
        key = corpus.repository_key
        spans = [corpus.span_of(fqn) for fqn in _PROBE_FQNS[:-1]]
        assert list(await postgres.symbols_in_spans(key, spans)) == list(
            await memory.symbols_in_spans(key, spans)
        )

    async def test_a_hunk_anchors_on_the_enclosing_symbol(
        self,
        indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex],
        corpus: IndexedCorpus,
    ) -> None:
        """Overlap, not containment: three changed lines in the middle of a
        method must return the whole method."""
        postgres, memory = indexes
        method = corpus.symbols["shop.models.Entity.validate"]
        middle = CodeSpan(
            path=method.span.path,
            line_start=method.span.line_start + 1,
            line_end=method.span.line_start + 1,
        )
        found = await postgres.symbols_in_spans(corpus.repository_key, [middle])
        assert method.fqn in {symbol.fqn for symbol in found}
        assert list(found) == list(
            await memory.symbols_in_spans(corpus.repository_key, [middle])
        )

    async def test_edges_touching_match(
        self,
        indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex],
        corpus: IndexedCorpus,
    ) -> None:
        postgres, memory = indexes
        key = corpus.repository_key
        # Compared as multisets: both walk edges in either direction and neither
        # port promises an order, but they must agree on the *set* of edges and
        # on how many of each.
        def key_of(edge: Any) -> tuple[str, str, str, str, float]:
            # `dst_fqn` and `dst_unresolved_name` are mutually exclusive and one
            # is always None, so they are flattened for sorting rather than
            # compared as None.
            return (
                str(edge.kind),
                edge.src_fqn,
                edge.dst_fqn or "",
                edge.dst_unresolved_name or "",
                round(edge.confidence, 4),
            )

        pg_edges = sorted(
            key_of(e) for e in await postgres.edges_touching(key, list(_PROBE_FQNS))
        )
        mem_edges = sorted(
            key_of(e) for e in await memory.edges_touching(key, list(_PROBE_FQNS))
        )
        assert pg_edges
        assert pg_edges == mem_edges

    async def test_chunks_for_symbols_match_in_order(
        self,
        indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex],
        corpus: IndexedCorpus,
    ) -> None:
        postgres, memory = indexes
        key = corpus.repository_key
        pg_chunks = await postgres.chunks_for_symbols(key, list(_PROBE_FQNS))
        mem_chunks = await memory.chunks_for_symbols(key, list(_PROBE_FQNS))
        assert pg_chunks
        assert [c.content_hash for c in pg_chunks] == [
            c.content_hash for c in mem_chunks
        ]
        assert list(pg_chunks) == list(mem_chunks)

    async def test_chunks_by_hash_match_in_order(
        self,
        indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex],
        corpus: IndexedCorpus,
    ) -> None:
        postgres, memory = indexes
        key = corpus.repository_key
        hashes = [chunk.content_hash for chunk in corpus.snapshot.chunks[:15]]
        assert list(await postgres.chunks_by_hash(key, hashes)) == list(
            await memory.chunks_by_hash(key, hashes)
        )

    async def test_empty_inputs_return_nothing(
        self,
        indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex],
        corpus: IndexedCorpus,
    ) -> None:
        postgres, memory = indexes
        key = corpus.repository_key
        assert list(await postgres.edges_touching(key, [])) == list(
            await memory.edges_touching(key, [])
        )
        assert list(await postgres.chunks_for_symbols(key, [])) == list(
            await memory.chunks_for_symbols(key, [])
        )


class TestVectorStoreParity:
    async def test_ann_returns_the_same_ranking(
        self,
        stores: tuple[PostgresVectorStore, InMemoryVectorStore],
        corpus: IndexedCorpus,
    ) -> None:
        """The two stores select the same chunks and score them the same, to the
        precision pgvector actually has.

        ``vector`` is a **float4** type. The in-memory adapter computes cosine in
        Python float64, so the two agree to about 1e-7 and no further -- which is
        invisible until several chunks tie *exactly* in float64 (three of them do
        here, being structurally identical methods). Python then breaks the tie
        on ``content_hash``; pgvector breaks it on distances that are no longer
        equal after the float4 round-trip. Both orderings are correct, so this
        asserts what is actually guaranteed -- same members, same scores within
        float4 epsilon -- rather than an ordering neither store promises.
        """
        postgres, memory = stores
        embedder = DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS)
        [vector] = await embedder.embed(
            [corpus.symbols["shop.models.Entity.validate"].fqn]
        )
        pg_hits = await postgres.search(corpus.repository_key, vector, limit=10)
        mem_hits = await memory.search(corpus.repository_key, vector, limit=10)
        assert pg_hits
        assert {hit.chunk_id for hit in pg_hits} == {
            hit.chunk_id for hit in mem_hits
        }

        mem_by_id = {hit.chunk_id: hit for hit in mem_hits}
        for pg_hit in pg_hits:
            mem_hit = mem_by_id[pg_hit.chunk_id]
            assert pg_hit.score == pytest.approx(mem_hit.score, abs=1e-6)
            assert pg_hit.path == mem_hit.path
            assert pg_hit.symbol_fqn == mem_hit.symbol_fqn

        # Ranking still has to be monotonic in score: ties may be ordered
        # differently, but a lower-scoring chunk must never outrank a higher one.
        scores = [hit.score for hit in pg_hits]
        assert scores == sorted(scores, reverse=True)

    async def test_excluded_paths_are_honoured(
        self,
        stores: tuple[PostgresVectorStore, InMemoryVectorStore],
        corpus: IndexedCorpus,
    ) -> None:
        postgres, memory = stores
        embedder = DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS)
        [vector] = await embedder.embed(["validate entity"])
        excluded = ["shop/models.py"]
        pg_hits = await postgres.search(
            corpus.repository_key, vector, limit=10, exclude_paths=excluded
        )
        mem_hits = await memory.search(
            corpus.repository_key, vector, limit=10, exclude_paths=excluded
        )
        assert pg_hits
        assert all(hit.path not in excluded for hit in pg_hits)
        assert {hit.chunk_id for hit in pg_hits} == {
            hit.chunk_id for hit in mem_hits
        }

    async def test_limit_is_respected(
        self,
        stores: tuple[PostgresVectorStore, InMemoryVectorStore],
        corpus: IndexedCorpus,
    ) -> None:
        postgres, _ = stores
        embedder = DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS)
        [vector] = await embedder.embed(["repository put"])
        assert len(await postgres.search(corpus.repository_key, vector, limit=3)) == 3


class TestScoping:
    async def test_another_tenant_gets_nothing(
        self,
        indexes: tuple[PostgresSymbolIndex, InMemorySymbolIndex],
        stores: tuple[PostgresVectorStore, InMemoryVectorStore],
        corpus: IndexedCorpus,
    ) -> None:
        """An unfiltered cross-tenant read is a data leak, so the adapters answer
        a foreign repository id with silence rather than with rows."""
        postgres, memory = indexes
        vectors, _ = stores
        stranger = str(uuid4())
        span = corpus.span_of("shop.models.Entity")
        assert await postgres.symbols_in_spans(stranger, [span]) == []
        assert await memory.symbols_in_spans(stranger, [span]) == []
        assert await postgres.edges_touching(stranger, list(_PROBE_FQNS)) == []
        assert await postgres.chunks_for_symbols(stranger, list(_PROBE_FQNS)) == []
        assert await vectors.search(stranger, [0.0] * EMBEDDING_DIMENSIONS) == []

    async def test_reads_see_one_snapshot_not_the_union_of_all(
        self, pg_pool: Any, corpus: IndexedCorpus
    ) -> None:
        """The reason 0002 exists.

        Every index run re-creates the repository's symbols. Without snapshot
        scoping, a repository indexed twice would return each symbol twice and
        each edge twice, and a context pack would be built from code drawn from
        two different commits.
        """
        await create_repository(pg_pool, corpus.repository_id)
        store = PostgresIndexStore(pg_pool)
        await store.save(corpus.snapshot)
        second = corpus.snapshot.model_copy(
            update={
                "id": uuid4(),
                "commit_sha": "3" * 40,
                "parent_snapshot_id": corpus.snapshot.id,
            }
        )
        await store.save(second)

        latest = await latest_ready_snapshot(pg_pool, corpus.repository_id)
        assert latest is not None
        assert latest.id == second.id
        assert latest.commit_sha == "3" * 40
        assert latest.embedding_model == corpus.snapshot.embedding_model

        index = await PostgresSymbolIndex.for_latest(pg_pool, corpus.repository_id)
        assert index is not None
        span = corpus.span_of("shop.models.Entity")
        found = await index.symbols_in_spans(corpus.repository_key, [span])
        assert len(found) == len({symbol.fqn for symbol in found})

        memory = InMemorySymbolIndex(corpus.snapshot)
        assert list(found) == list(
            await memory.symbols_in_spans(corpus.repository_key, [span])
        )

    async def test_for_latest_is_none_before_the_first_index(
        self, pg_pool: Any
    ) -> None:
        """A fresh installation is a real state, not an error."""
        never_indexed = await create_repository(pg_pool)
        assert await PostgresSymbolIndex.for_latest(pg_pool, never_indexed) is None
        assert await PostgresVectorStore.for_latest(pg_pool, never_indexed) is None
