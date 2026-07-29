"""Hybrid retrieval: expansion, fusion, budget, and the whole pack (sec. 4.3).

The load-bearing claim of ADR-002 is that *structure* finds the context a review
needs and similarity only supplements it. These tests are written to hold that
claim to account -- most of them would still pass against a pure vector
retriever, except the ones that matter.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.domain.contracts import CodeSpan
from app.domain.indexing.models import Chunk, EdgeKind, Language, SymbolEdge, confidence
from app.domain.retrieval.expansion import ExpansionConfig, expand
from app.domain.retrieval.fusion import (
    DEFAULT_WEIGHTS,
    FusionWeights,
    enforce_budget,
    fuse_score,
)
from app.domain.retrieval.models import ContextItem, Provenance


def _edges(*specs: tuple[EdgeKind, str, str, float]) -> list[SymbolEdge]:
    return [
        SymbolEdge(kind=kind, src_fqn=src, dst_fqn=dst, confidence=score)
        for kind, src, dst, score in specs
    ]


def _loader(edges: Sequence[SymbolEdge]):
    async def load(fqns: Sequence[str]) -> Sequence[SymbolEdge]:
        wanted = set(fqns)
        return [e for e in edges if e.src_fqn in wanted or e.dst_fqn in wanted]

    return load


def _item(
    *,
    fqn: str,
    tokens: int,
    score: float,
    provenance: Provenance = Provenance.GRAPH,
    path: str = "app/m.py",
    line: int = 1,
) -> ContextItem:
    return ContextItem(
        chunk=Chunk(
            content_hash=f"{abs(hash(fqn)):064x}"[:64],
            symbol_fqn=fqn,
            span=CodeSpan(path=path, line_start=line, line_end=line + 2),
            language=Language.PYTHON,
            token_count=tokens,
            content=f"def {fqn.rsplit('.', 1)[-1]}(): ...",
        ),
        provenance=provenance,
        score=score,
        graph_distance=0 if provenance is Provenance.ANCHOR else 1,
        reason="test fixture",
    )


class TestGraphExpansion:
    async def test_reaches_both_callers_and_callees(self) -> None:
        # The caller is the point: it is what passes the argument that makes the
        # change a bug, and it is often not textually similar to the callee.
        edges = _edges(
            (EdgeKind.CALLS, "app.handler.handle", "app.store.save", 1.0),
            (EdgeKind.CALLS, "app.store.save", "app.db.write", 1.0),
        )
        found = await expand(["app.store.save"], _loader(edges))

        by_fqn = {n.fqn: n for n in found}
        assert set(by_fqn) == {"app.handler.handle", "app.db.write"}
        assert by_fqn["app.handler.handle"].reason == "calls app.store.save"
        assert by_fqn["app.db.write"].reason == "called by app.store.save"

    async def test_stops_at_the_configured_depth(self) -> None:
        edges = _edges(
            (EdgeKind.CALLS, "a.one", "a.two", 1.0),
            (EdgeKind.CALLS, "a.two", "a.three", 1.0),
            (EdgeKind.CALLS, "a.three", "a.four", 1.0),
        )
        found = await expand(["a.one"], _loader(edges))
        assert {n.fqn for n in found} == {"a.two", "a.three"}
        assert {n.distance for n in found} == {1, 2}

    async def test_anchors_are_never_returned_as_their_own_neighbours(self) -> None:
        edges = _edges((EdgeKind.CALLS, "a.one", "a.two", 1.0))
        found = await expand(["a.one", "a.two"], _loader(edges))
        assert found == []

    async def test_proximity_decays_with_distance(self) -> None:
        edges = _edges(
            (EdgeKind.CALLS, "a.one", "a.two", 1.0),
            (EdgeKind.CALLS, "a.two", "a.three", 1.0),
        )
        by_fqn = {n.fqn: n for n in await expand(["a.one"], _loader(edges))}
        assert by_fqn["a.two"].proximity > by_fqn["a.three"].proximity

    async def test_low_confidence_edges_are_down_weighted_not_dropped(self) -> None:
        # ARCHITECTURE sec. 4.2: unresolved/heuristic edges are down-weighted
        # during expansion rather than dropped.
        edges = _edges(
            (EdgeKind.CALLS, "a.anchor", "a.certain", confidence.EXACT),
            (EdgeKind.CALLS, "a.anchor", "a.guessed", confidence.HEURISTIC_AMBIGUOUS),
        )
        by_fqn = {n.fqn: n for n in await expand(["a.anchor"], _loader(edges))}
        assert set(by_fqn) == {"a.certain", "a.guessed"}
        assert by_fqn["a.certain"].proximity > by_fqn["a.guessed"].proximity

    async def test_a_confident_two_hop_can_outrank_a_doubtful_one_hop(self) -> None:
        edges = _edges(
            (EdgeKind.IMPORTS, "a.anchor", "a.weak", confidence.UNRESOLVED),
            (EdgeKind.CALLS, "a.anchor", "a.mid", confidence.EXACT),
            (EdgeKind.CALLS, "a.mid", "a.far", confidence.EXACT),
        )
        by_fqn = {n.fqn: n for n in await expand(["a.anchor"], _loader(edges))}
        assert by_fqn["a.far"].proximity > by_fqn["a.weak"].proximity

    async def test_unresolved_edges_have_no_far_side_to_travel_to(self) -> None:
        dangling = SymbolEdge(
            kind=EdgeKind.CALLS,
            src_fqn="a.anchor",
            dst_unresolved_name="mystery",
            confidence=confidence.UNRESOLVED,
        )
        assert await expand(["a.anchor"], _loader([dangling])) == []

    async def test_fan_out_is_capped_per_level_keeping_the_best(self) -> None:
        edges = _edges(
            *((EdgeKind.CALLS, f"caller{i}", "hot.symbol", 0.1 * (i % 10 + 1))
              for i in range(100))
        )
        found = await expand(
            ["hot.symbol"], _loader(edges), ExpansionConfig(max_per_level=5)
        )
        assert len(found) == 5
        assert min(n.proximity for n in found) >= 0.4 / 2

    async def test_edges_are_loaded_once_per_level_not_per_symbol(self) -> None:
        # An N+1 query here is what blows the 800 ms p95 budget in production.
        edges = _edges(*((EdgeKind.CALLS, "a.anchor", f"a.callee{i}", 1.0) for i in range(20)))
        calls: list[int] = []

        async def counting(fqns: Sequence[str]) -> Sequence[SymbolEdge]:
            calls.append(len(fqns))
            wanted = set(fqns)
            return [e for e in edges if e.src_fqn in wanted or e.dst_fqn in wanted]

        await expand(["a.anchor"], counting)
        assert len(calls) == 2, "one round trip per BFS level, whatever the fan-out"

    @pytest.mark.parametrize(("depth", "per_level"), [(0, 1), (1, 0)])
    def test_degenerate_configuration_is_rejected(
        self, depth: int, per_level: int
    ) -> None:
        with pytest.raises(ValueError):
            ExpansionConfig(max_depth=depth, max_per_level=per_level)


class TestFusion:
    def test_structure_outweighs_similarity(self) -> None:
        # ADR-002 chose structure-first; the ranking has to express that.
        structural = fuse_score(graph_proximity=1.0)
        semantic = fuse_score(semantic_score=1.0)
        assert structural > semantic

    def test_signals_corroborate(self) -> None:
        both = fuse_score(graph_proximity=0.5, semantic_score=0.5)
        assert both > fuse_score(graph_proximity=0.5)
        assert both > fuse_score(semantic_score=0.5)

    def test_weights_are_configurable(self) -> None:
        semantic_first = FusionWeights(graph=0.1, semantic=0.9)
        assert fuse_score(semantic_score=1.0, weights=semantic_first) > fuse_score(
            graph_proximity=1.0, weights=semantic_first
        )

    @pytest.mark.parametrize(
        "weights",
        [{"graph": -0.1}, {"graph": 0.0, "semantic": 0.0, "co_change": 0.0}],
    )
    def test_invalid_weights_are_rejected(self, weights: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            FusionWeights(**weights)

    def test_co_change_is_wired_but_unused_by_default(self) -> None:
        # The signal has no source yet; the formula must not pretend otherwise.
        assert DEFAULT_WEIGHTS.co_change > 0.0
        assert fuse_score(graph_proximity=1.0, co_change=0.0) == pytest.approx(
            DEFAULT_WEIGHTS.graph
        )


class TestBudgetEnforcement:
    def test_drops_whole_items_lowest_score_first(self) -> None:
        items = [
            _item(fqn="a.best", tokens=50, score=0.9),
            _item(fqn="a.worst", tokens=50, score=0.1),
        ]
        kept, used, dropped = enforce_budget(items, token_budget=60)
        assert [item.symbol_fqn for item in kept] == ["a.best"]
        assert (used, dropped) == (50, 1)

    def test_never_truncates_a_symbol(self) -> None:
        # Half an implementation is worse than none: the reviewer cannot tell
        # the half they are reading is incomplete.
        items = [_item(fqn="a.big", tokens=500, score=0.9)]
        kept, used, dropped = enforce_budget(items, token_budget=100)
        assert kept == ()
        assert (used, dropped) == (0, 1)

    def test_anchors_survive_the_budget(self) -> None:
        items = [
            _item(fqn="a.anchor", tokens=900, score=1.0, provenance=Provenance.ANCHOR),
            _item(fqn="a.extra", tokens=50, score=0.9),
        ]
        kept, used, dropped = enforce_budget(items, token_budget=100)
        assert [item.symbol_fqn for item in kept] == ["a.anchor"]
        assert used == 900, "an over-budget pack reports the overrun honestly"
        assert dropped == 1

    def test_a_large_item_does_not_evict_smaller_ones_behind_it(self) -> None:
        items = [
            _item(fqn="a.huge", tokens=1000, score=0.99),
            _item(fqn="a.small1", tokens=10, score=0.5),
            _item(fqn="a.small2", tokens=10, score=0.4),
        ]
        kept, used, dropped = enforce_budget(items, token_budget=100)
        assert {item.symbol_fqn for item in kept} == {"a.small1", "a.small2"}
        assert (used, dropped) == (20, 1)

    def test_anchors_are_ordered_first_in_the_pack(self) -> None:
        items = [
            _item(fqn="a.graph", tokens=10, score=0.9),
            _item(fqn="a.anchor", tokens=10, score=0.2, provenance=Provenance.ANCHOR),
        ]
        kept, _, _ = enforce_budget(items, token_budget=1000)
        assert kept[0].provenance is Provenance.ANCHOR

    def test_ordering_is_deterministic_for_equal_scores(self) -> None:
        items = [
            _item(fqn="a.b", tokens=10, score=0.5, path="app/b.py"),
            _item(fqn="a.a", tokens=10, score=0.5, path="app/a.py"),
        ]
        first, _, _ = enforce_budget(items, token_budget=1000)
        second, _, _ = enforce_budget(list(reversed(items)), token_budget=1000)
        assert [i.path for i in first] == [i.path for i in second] == [
            "app/a.py",
            "app/b.py",
        ]


class TestContextItemInvariants:
    def test_an_anchor_cannot_claim_graph_distance(self) -> None:
        with pytest.raises(ValueError, match="own origin"):
            ContextItem(
                chunk=_item(fqn="a.x", tokens=1, score=1.0).chunk,
                provenance=Provenance.ANCHOR,
                score=1.0,
                graph_distance=2,
                reason="wrong",
            )
