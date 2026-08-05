"""The M2 exit gate: does the pack contain what a reviewer would need?

ROADMAP M2: "on 30 hand-built queries, the file a human would need appears in
the context pack >= 90% of the time; p95 retrieval < 800 ms." Both are measured
here and printed, per the project's rule that a milestone is done when its exit
criterion is *measured*, not when the code exists.

Two honesty notes about what these numbers do and do not prove:

* **Recall is measured over the whole pipeline** -- real parsers, the real
  resolver, real chunking, real fusion and budget. A resolver regression that
  stops emitting ``CALLS`` edges shows up here as a recall drop, which is the
  point of reusing the M1 corpora as the retrieval corpus.
* **The latency number is a floor, not a forecast.** It is measured against the
  in-memory adapters, whose ANN search is a linear scan over a few hundred
  chunks; production is pgvector HNSW over millions, across a network. The
  budget is asserted here to catch algorithmic blowups (an N+1 expansion, an
  accidental full scan per hunk); the real p95 has to be re-measured against the
  Postgres adapter when it lands.

Run ``pytest tests/eval/test_retrieval_quality.py -s`` to see the report.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import pytest

from app.domain.retrieval.models import Provenance
from app.domain.retrieval.retriever import ContextRetriever, RetrievalConfig
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore
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

RECALL_TARGET = 0.90
"""ROADMAP M2: the needed file appears in the pack at least this often."""

P95_BUDGET_MS = 800
QUERY_TARGET = 30
"""Hand-built queries the roadmap asks for."""

GATE_TOKEN_BUDGET = 300
"""The budget the recall gate runs at, and the reason the gate can fail.

These corpora are ~20 files each, so a production-sized 12k-token pack holds
*everything* depth-2 expansion reaches and recall is trivially 100% -- a gate
that cannot fail measures nothing. Scaling the budget down to 300 tokens
restores the selectivity a real repository imposes for free: ~40 candidates
compete for ~12 slots, so the fusion ranking has to actually be right. Recall at
a production-shaped budget is reported alongside, for scale.
"""

PRODUCTION_TOKEN_BUDGET = 12_000


def _retriever(corpus: IndexedCorpus, token_budget: int) -> ContextRetriever:
    """The in-memory adapter set. ``test_retrieval_pgvector.py`` wires the same
    ``ContextRetriever`` to Postgres and runs these same queries through it."""
    return ContextRetriever(
        index=InMemorySymbolIndex(corpus.snapshot),
        vectors=InMemoryVectorStore(corpus.snapshot),
        embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
        config=RetrievalConfig(token_budget=token_budget),
    )


async def _run(
    corpus: IndexedCorpus,
    queries: Sequence[RetrievalQuery],
    token_budget: int = GATE_TOKEN_BUDGET,
) -> list[QueryOutcome]:
    return await run_queries(_retriever(corpus, token_budget), corpus, queries)


@pytest.fixture(scope="module")
async def python_corpus() -> IndexedCorpus:
    return await build_corpus(PY_CORPUS, PY_TEST_FILES, typescript=False)


@pytest.fixture(scope="module")
async def typescript_corpus() -> IndexedCorpus:
    return await build_corpus(TS_CORPUS, TS_TEST_FILES, typescript=True)


@pytest.fixture(scope="module")
async def outcomes(
    python_corpus: IndexedCorpus, typescript_corpus: IndexedCorpus
) -> list[QueryOutcome]:
    """The gate: packs squeezed to a budget where ranking has to be right."""
    return [
        *await _run(python_corpus, PY_QUERIES),
        *await _run(typescript_corpus, TS_QUERIES),
    ]


@pytest.fixture(scope="module")
async def production_budget_outcomes(
    python_corpus: IndexedCorpus, typescript_corpus: IndexedCorpus
) -> list[QueryOutcome]:
    return [
        *await _run(python_corpus, PY_QUERIES, PRODUCTION_TOKEN_BUDGET),
        *await _run(typescript_corpus, TS_QUERIES, PRODUCTION_TOKEN_BUDGET),
    ]


class TestRetrievalQuality:
    def test_corpus_has_enough_queries(self) -> None:
        assert len(PY_QUERIES) + len(TS_QUERIES) >= QUERY_TARGET

    def test_needed_file_is_retrieved(
        self,
        outcomes: list[QueryOutcome],
        production_budget_outcomes: list[QueryOutcome],
    ) -> None:
        hits = sum(1 for outcome in outcomes if outcome.hit)
        recall = hits / len(outcomes)
        generous = sum(1 for o in production_budget_outcomes if o.hit) / len(
            production_budget_outcomes
        )
        dropped = statistics.mean(
            outcome.pack.stats.dropped_by_budget for outcome in outcomes
        )
        print(
            f"\n[retrieval] needed-file recall {recall:.1%} "
            f"({hits}/{len(outcomes)} queries) at a {GATE_TOKEN_BUDGET}-token "
            f"budget, dropping {dropped:.0f} candidates per pack | "
            f"{generous:.1%} at a {PRODUCTION_TOKEN_BUDGET}-token budget"
        )
        for outcome in outcomes:
            if not outcome.hit:
                print(
                    f"    miss: changing {outcome.query.changed_symbol} did not "
                    f"retrieve {outcome.query.needs} ({outcome.query.why}); "
                    f"pack held {list(outcome.pack.paths)}"
                )
        assert recall >= RECALL_TARGET, (
            f"needed-file recall {recall:.1%} is below the {RECALL_TARGET:.0%} "
            "M2 exit criterion"
        )

    def test_retrieval_latency_is_within_budget(
        self, outcomes: list[QueryOutcome]
    ) -> None:
        timings = sorted(outcome.duration_ms for outcome in outcomes)
        p95 = timings[max(0, round(0.95 * len(timings)) - 1)]
        print(
            f"[retrieval] latency p50 {statistics.median(timings):.1f} ms, "
            f"p95 {p95:.1f} ms, max {timings[-1]:.1f} ms "
            f"(in-memory adapters; a floor, not a production forecast)"
        )
        assert p95 < P95_BUDGET_MS

    def test_every_pack_anchors_on_the_change(
        self, outcomes: list[QueryOutcome]
    ) -> None:
        # A pack without the changed code is not a context pack, whatever else
        # it managed to retrieve.
        for outcome in outcomes:
            assert outcome.pack.by_provenance(Provenance.ANCHOR), (
                f"no anchor for {outcome.query.changed_symbol}"
            )

    def test_packs_stay_within_their_token_budget(
        self, outcomes: list[QueryOutcome]
    ) -> None:
        used = [outcome.pack.stats.tokens_used for outcome in outcomes]
        print(
            f"[retrieval] pack size p50 {statistics.median(used):.0f} tokens, "
            f"max {max(used)} of a {outcomes[0].pack.stats.token_budget} budget"
        )
        for outcome in outcomes:
            stats = outcome.pack.stats
            if stats.tokens_used <= stats.token_budget:
                continue
            # The one documented overrun: anchors are never dropped, so a diff
            # touching a symbol larger than the whole budget still gets its own
            # code. Nothing else may exceed it.
            anchor_tokens = sum(
                item.token_count
                for item in outcome.pack.by_provenance(Provenance.ANCHOR)
            )
            assert stats.tokens_used == anchor_tokens, (
                f"{outcome.query.changed_symbol} overran its budget with "
                "non-anchor context"
            )

    def test_structure_carries_the_pack(self, outcomes: list[QueryOutcome]) -> None:
        """ADR-002 as a measurement, not a claim: most retrieved context should
        arrive through the graph, with vectors supplementing rather than
        driving. If this flips, the retriever has quietly become a vector
        search with extra steps."""
        graph = sum(
            len(outcome.pack.by_provenance(Provenance.GRAPH)) for outcome in outcomes
        )
        semantic = sum(
            len(outcome.pack.by_provenance(Provenance.SEMANTIC))
            for outcome in outcomes
        )
        share = graph / (graph + semantic) if graph + semantic else 0.0
        print(
            f"[retrieval] provenance: {graph} graph items vs {semantic} semantic "
            f"({share:.0%} structural)"
        )
        assert share > 0.5

    def test_hits_are_reached_structurally_not_by_luck(
        self, outcomes: list[QueryOutcome]
    ) -> None:
        """The needed file should mostly arrive on an edge. A hit that only ever
        arrives by cosine similarity is a hit the graph should have found, and
        it will not survive a corpus where similar code is common."""
        structural = 0
        for outcome in outcomes:
            if not outcome.hit:
                continue
            reached = {
                item.path
                for item in outcome.pack.items
                if item.provenance is not Provenance.SEMANTIC
            }
            structural += outcome.query.needs in reached
        hits = sum(1 for outcome in outcomes if outcome.hit)
        print(f"[retrieval] {structural}/{hits} hits reached through the graph")
        assert structural / hits >= 0.8
