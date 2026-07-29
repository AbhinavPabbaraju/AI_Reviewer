"""Graph expansion: the structural half of retrieval (ARCHITECTURE sec. 4.3).

Breadth-first from the diff's anchor symbols, depth <= 2, budgeted. Three
decisions carry the design:

**Depth 2, not "until it stops."** Depth 1 is the direct neighbourhood: what the
changed code calls, who calls it, what it inherits. Depth 2 catches the common
indirection (a handler calls a service that calls the changed store method).
Depth 3 in a normally-connected repository is most of the repository, which is
the same as no retrieval at all.

**Proximity decays, and unresolved edges decay faster.** An edge the resolver
tagged ``UNRESOLVED`` is a guess; following it at full weight would let one bad
guess drag unrelated code into every pack. Proximity is
``confidence / (distance + 1)``, so a confident two-hop neighbour can still
outrank a doubtful one-hop one -- which is the correct ordering, because M1's
confidence tags are calibrated by construction rather than self-reported.

**A per-level fan-out cap.** One utility with four hundred callers must not
evict everything else from the pack before fusion gets a say. The cap keeps the
best-scoring neighbours per level, so expansion cost is bounded by the budget
rather than by how popular the changed symbol happens to be.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from app.domain.indexing.models import EdgeKind, SymbolEdge

__all__ = [
    "DEFAULT_EXPANSION",
    "DEFAULT_KIND_WEIGHTS",
    "EdgeLoader",
    "ExpansionConfig",
    "Neighbour",
    "expand",
]


type EdgeLoader = Callable[[Sequence[str]], Awaitable[Sequence[SymbolEdge]]]
"""Loads every edge touching a set of fqns -- one BFS level, one round trip."""


@dataclass(frozen=True, slots=True)
class ExpansionConfig:
    max_depth: int = 2
    max_per_level: int = 40
    """Neighbours kept per BFS level, best proximity first."""
    kind_weights: Mapping[EdgeKind, float] = field(
        default_factory=lambda: DEFAULT_KIND_WEIGHTS
    )

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.max_per_level < 1:
            raise ValueError("max_per_level must be at least 1")

    def weight_for(self, kind: EdgeKind) -> float:
        return self.kind_weights.get(kind, 0.5)


# Relative worth of each relation *for reviewing a diff*, which is not the same
# as how interesting the relation is in general. CALLS and INHERITS constrain
# the changed code's behaviour; TESTS says what is meant to hold; IMPORTS is
# mostly module bookkeeping and is the weakest signal in the set.
DEFAULT_KIND_WEIGHTS: Mapping[EdgeKind, float] = {
    EdgeKind.CALLS: 1.0,
    EdgeKind.INHERITS: 1.0,
    EdgeKind.TESTS: 0.9,
    EdgeKind.REFERENCES: 0.7,
    EdgeKind.DEFINES: 0.5,
    EdgeKind.IMPORTS: 0.3,
}

DEFAULT_EXPANSION = ExpansionConfig()


@dataclass(frozen=True, slots=True)
class Neighbour:
    """A symbol reached by expansion, with why and how far."""

    fqn: str
    distance: int
    proximity: float
    kind: EdgeKind
    via_fqn: str
    """The anchor-side endpoint this was reached from -- the 'why' in the pack."""
    inbound: bool
    """True when the neighbour *points at* the anchor (a caller of it)."""

    @property
    def reason(self) -> str:
        relation = _INBOUND_PHRASES[self.kind] if self.inbound else _OUTBOUND_PHRASES[self.kind]
        return f"{relation} {self.via_fqn}"


_OUTBOUND_PHRASES: Mapping[EdgeKind, str] = {
    EdgeKind.CALLS: "called by",
    EdgeKind.INHERITS: "base class of",
    EdgeKind.TESTS: "under test by",
    EdgeKind.REFERENCES: "referenced by",
    EdgeKind.DEFINES: "defined in",
    EdgeKind.IMPORTS: "imported by",
}

_INBOUND_PHRASES: Mapping[EdgeKind, str] = {
    EdgeKind.CALLS: "calls",
    EdgeKind.INHERITS: "inherits from",
    EdgeKind.TESTS: "tests",
    EdgeKind.REFERENCES: "references",
    EdgeKind.DEFINES: "defines",
    EdgeKind.IMPORTS: "imports",
}


async def expand(
    anchors: Iterable[str],
    load_edges: EdgeLoader,
    config: ExpansionConfig = DEFAULT_EXPANSION,
) -> list[Neighbour]:
    """BFS out from ``anchors``, returning the neighbours found, nearest first.

    Anchors themselves are never returned: they are already in the pack, and
    re-adding them as their own neighbours would double-count the diff.
    """
    origin = set(anchors)
    if not origin:
        return []

    seen: set[str] = set(origin)
    frontier: list[str] = sorted(origin)
    found: list[Neighbour] = []

    for distance in range(1, config.max_depth + 1):
        edges = await load_edges(frontier)
        level: dict[str, Neighbour] = {}
        frontier_set = set(frontier)
        for edge in edges:
            for fqn, via, inbound in _endpoints(edge, frontier_set):
                if fqn in seen:
                    continue
                proximity = (
                    edge.confidence * config.weight_for(edge.kind) / (distance + 1)
                )
                current = level.get(fqn)
                if current is None or proximity > current.proximity:
                    level[fqn] = Neighbour(
                        fqn=fqn,
                        distance=distance,
                        proximity=min(proximity, 1.0),
                        kind=edge.kind,
                        via_fqn=via,
                        inbound=inbound,
                    )
        if not level:
            break
        best = sorted(
            level.values(), key=lambda n: (-n.proximity, n.fqn)
        )[: config.max_per_level]
        found.extend(best)
        seen.update(neighbour.fqn for neighbour in best)
        frontier = [neighbour.fqn for neighbour in best]

    return found


def _endpoints(
    edge: SymbolEdge, frontier: set[str]
) -> list[tuple[str, str, bool]]:
    """``(neighbour, via, inbound)`` for each way this edge leaves the frontier.

    An unresolved edge has no destination symbol to travel to; it is kept in the
    graph for honesty (sec. 4.2) but there is nothing on the far end to retrieve.
    """
    results: list[tuple[str, str, bool]] = []
    if edge.dst_fqn is None:
        return results
    if edge.src_fqn in frontier:
        results.append((edge.dst_fqn, edge.src_fqn, False))
    if edge.dst_fqn in frontier:
        results.append((edge.src_fqn, edge.dst_fqn, True))
    return results
