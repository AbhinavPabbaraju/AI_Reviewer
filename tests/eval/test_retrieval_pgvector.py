"""The M2 exit criterion, measured against Postgres + pgvector.

ROADMAP M2: "on 30 hand-built queries, the file a human would need appears in
the context pack >= 90% of the time; **p95 retrieval < 800 ms**." The recall half
has been green since the retriever landed. The latency half could not honestly be
claimed, because it had only ever been measured against in-memory adapters whose
ANN search is a linear scan over a few hundred chunks -- a floor, not a forecast,
as ``test_retrieval_quality.py`` says in as many words.

This file replaces that floor with a measurement: the same 40 queries, the same
corpora, the same ``ContextRetriever``, wired to ``PostgresSymbolIndex`` and
``PostgresVectorStore`` over a real database. Every number includes its round
trips.

It also re-runs the recall gate against Postgres. That is not redundant with the
in-memory gate -- it is the check that the adapter swap did not quietly change
what gets retrieved. Recall measured on the fake and recall measured on the real
store must agree, or one of them is measuring a system nobody ships.

Skips when ``ARGUS_TEST_DATABASE_URL`` is unset; see ``tests/pg.py``.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

import pytest

from app.domain.retrieval.models import Provenance
from app.domain.retrieval.retriever import ContextRetriever, RetrievalConfig
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.retrieval.postgres import PostgresSymbolIndex, PostgresVectorStore
from app.infra.store.postgres import PostgresIndexStore
from tests.eval.corpus.indexed import (
    EMBEDDING_DIMENSIONS,
    IndexedCorpus,
    QueryOutcome,
    build_corpus,
    run_queries,
)
from tests.eval.corpus.python_corpus import PY_CORPUS, PY_TEST_FILES
from tests.eval.corpus.retrieval_queries import PY_QUERIES, TS_QUERIES, RetrievalQuery
from tests.eval.corpus.typescript_corpus import TS_CORPUS, TS_TEST_FILES
from tests.eval.test_retrieval_quality import (
    GATE_TOKEN_BUDGET,
    P95_BUDGET_MS,
    RECALL_TARGET,
)
from tests.pg import create_repository


async def _serve(
    pool: Any, corpus: IndexedCorpus, token_budget: int
) -> ContextRetriever:
    """Persist a corpus and return a retriever reading it back out of Postgres."""
    await create_repository(pool, corpus.repository_id)
    await PostgresIndexStore(pool).save(corpus.snapshot)

    index = await PostgresSymbolIndex.for_latest(pool, corpus.repository_id)
    vectors = await PostgresVectorStore.for_latest(pool, corpus.repository_id)
    assert index is not None and vectors is not None, "the snapshot did not persist"
    return ContextRetriever(
        index=index,
        vectors=vectors,
        embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
        config=RetrievalConfig(token_budget=token_budget),
    )


@pytest.fixture(scope="module")
async def corpora() -> list[tuple[IndexedCorpus, Sequence[RetrievalQuery]]]:
    return [
        (await build_corpus(PY_CORPUS, PY_TEST_FILES, typescript=False), PY_QUERIES),
        (await build_corpus(TS_CORPUS, TS_TEST_FILES, typescript=True), TS_QUERIES),
    ]


@pytest.fixture
async def outcomes(
    pg_pool: Any, corpora: list[tuple[IndexedCorpus, Sequence[RetrievalQuery]]]
) -> list[QueryOutcome]:
    results: list[QueryOutcome] = []
    for corpus, queries in corpora:
        retriever = await _serve(pg_pool, corpus, GATE_TOKEN_BUDGET)
        results.extend(await run_queries(retriever, corpus, queries))
    return results


class TestRetrievalAgainstPgvector:
    def test_p95_latency_is_within_the_m2_budget(
        self, outcomes: list[QueryOutcome]
    ) -> None:
        """The M2 exit criterion. Measured, not forecast."""
        timings = sorted(outcome.duration_ms for outcome in outcomes)
        p95 = timings[max(0, round(0.95 * len(timings)) - 1)]
        print(
            f"\n[pgvector] retrieval latency p50 "
            f"{statistics.median(timings):.1f} ms, p95 {p95:.1f} ms, "
            f"max {timings[-1]:.1f} ms over {len(timings)} queries "
            f"(Postgres + pgvector HNSW, round trips included)"
        )
        assert p95 < P95_BUDGET_MS, (
            f"p95 retrieval {p95:.1f} ms exceeds the {P95_BUDGET_MS} ms M2 budget"
        )

    def test_needed_file_is_retrieved(self, outcomes: list[QueryOutcome]) -> None:
        hits = sum(1 for outcome in outcomes if outcome.hit)
        recall = hits / len(outcomes)
        print(
            f"[pgvector] needed-file recall {recall:.1%} ({hits}/{len(outcomes)} "
            f"queries) at a {GATE_TOKEN_BUDGET}-token budget"
        )
        for outcome in outcomes:
            if not outcome.hit:
                print(
                    f"    miss: changing {outcome.query.changed_symbol} did not "
                    f"retrieve {outcome.query.needs}"
                )
        assert recall >= RECALL_TARGET

    def test_structure_still_carries_the_pack(
        self, outcomes: list[QueryOutcome]
    ) -> None:
        """ADR-002 has to survive the storage swap. An adapter whose graph
        queries quietly under-returned would show up here as the semantic share
        climbing, not as a test error."""
        graph = sum(
            len(outcome.pack.by_provenance(Provenance.GRAPH)) for outcome in outcomes
        )
        semantic = sum(
            len(outcome.pack.by_provenance(Provenance.SEMANTIC))
            for outcome in outcomes
        )
        share = graph / (graph + semantic) if graph + semantic else 0.0
        print(
            f"[pgvector] provenance: {graph} graph items vs {semantic} semantic "
            f"({share:.0%} structural)"
        )
        assert share > 0.5

    def test_every_pack_anchors_on_the_change(
        self, outcomes: list[QueryOutcome]
    ) -> None:
        for outcome in outcomes:
            assert outcome.pack.by_provenance(Provenance.ANCHOR), (
                f"no anchor for {outcome.query.changed_symbol}"
            )


class TestAdapterEquivalence:
    """Postgres and the in-memory fake must retrieve the same thing.

    The strongest statement this suite can make: not "both are above the bar"
    but "both build the same pack". Anything weaker leaves room for the eval
    harness (which runs on the fake) to certify behaviour production does not
    have.
    """

    async def test_packs_are_identical_to_the_in_memory_adapters(
        self,
        pg_pool: Any,
        corpora: list[tuple[IndexedCorpus, Sequence[RetrievalQuery]]],
    ) -> None:
        from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore

        for corpus, queries in corpora:
            postgres = await _serve(pg_pool, corpus, GATE_TOKEN_BUDGET)
            memory = ContextRetriever(
                index=InMemorySymbolIndex(corpus.snapshot),
                vectors=InMemoryVectorStore(corpus.snapshot),
                embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
                config=RetrievalConfig(token_budget=GATE_TOKEN_BUDGET),
            )
            for query in queries:
                hunk = corpus.span_of(query.changed_symbol)
                key = corpus.repository_key
                pg_pack = await postgres.retrieve(repository_id=key, hunks=[hunk])
                mem_pack = await memory.retrieve(repository_id=key, hunks=[hunk])
                assert [item.chunk.content_hash for item in pg_pack.items] == [
                    item.chunk.content_hash for item in mem_pack.items
                ], f"packs diverged for {query.changed_symbol}"
                assert pg_pack.stats.tokens_used == mem_pack.stats.tokens_used
                assert (
                    pg_pack.stats.dropped_by_budget
                    == mem_pack.stats.dropped_by_budget
                )
