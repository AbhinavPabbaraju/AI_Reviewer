"""Context packs: what the reviewer actually gets to read (ARCHITECTURE sec. 4.3).

A context pack is the answer to "what does someone need in front of them to judge
this diff?" -- deliberately not "what is most similar to this diff." The
distinction is ADR-002's whole argument: the caller that passes ``None`` is what
makes the null-deref a bug, and it is often not textually similar to the callee
at all.

Every item records **why** it is in the pack (:class:`Provenance`) and what it
scored on each signal. That is not decoration: the retrieval inspector in M7 is
"what context did this review actually see?", and a pack that cannot explain
itself makes a bad review impossible to diagnose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from app.domain.base import Frozen
from app.domain.contracts import CodeSpan
from app.domain.indexing.models import Chunk

__all__ = [
    "ContextItem",
    "ContextPack",
    "Provenance",
    "RetrievalStats",
]


class Provenance(StrEnum):
    """How an item earned its place in the pack."""

    ANCHOR = "anchor"
    """A symbol the diff actually touches. Never dropped by the budget: a pack
    without the changed code is not a pack."""

    GRAPH = "graph"
    """Reached by symbol-graph expansion -- a caller, callee, base class, or
    test of an anchor. The structural half of ADR-002, and the half that finds
    the constraint the diff violates."""

    SEMANTIC = "semantic"
    """Reached by vector search: similar patterns, docs, config, migrations --
    material with no edge to the diff. The supplement, not the foundation."""


class ContextItem(Frozen):
    """One chunk in the pack, with its provenance and per-signal scores."""

    chunk: Chunk
    provenance: Provenance
    score: float = Field(ge=0.0, description="Fused rank; higher is kept longer.")
    graph_distance: int | None = Field(
        default=None,
        ge=0,
        description="BFS hops from the nearest anchor; None if not graph-reached.",
    )
    graph_proximity: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(
        min_length=1,
        description="Human-readable justification, e.g. 'calls app.store.save'. "
        "Rendered directly by the M7 retrieval inspector.",
    )

    @property
    def path(self) -> str:
        return self.chunk.path

    @property
    def symbol_fqn(self) -> str | None:
        return self.chunk.symbol_fqn

    @property
    def token_count(self) -> int:
        return self.chunk.token_count

    @model_validator(mode="after")
    def _anchors_are_at_distance_zero(self) -> Self:
        if self.provenance is Provenance.ANCHOR and self.graph_distance not in (0, None):
            raise ValueError("an anchor is its own origin: graph_distance must be 0")
        return self


class RetrievalStats(Frozen):
    """Measured facts about one retrieval, for the latency/quality budgets."""

    anchors: int
    graph_candidates: int
    semantic_candidates: int
    dropped_by_budget: int
    tokens_used: int
    token_budget: int
    duration_ms: int

    @property
    def budget_utilization(self) -> float:
        return self.tokens_used / self.token_budget if self.token_budget else 0.0


class ContextPack(Frozen):
    """The ranked, budget-enforced context for one review request."""

    repository_id: str
    hunks: tuple[CodeSpan, ...]
    items: tuple[ContextItem, ...]
    stats: RetrievalStats

    @property
    def paths(self) -> tuple[str, ...]:
        """Distinct paths present, in pack order. The M2 exit criterion is
        phrased over this: does the file a human would need appear at all?"""
        seen: dict[str, None] = {}
        for item in self.items:
            seen.setdefault(item.path, None)
        return tuple(seen)

    def by_provenance(self, provenance: Provenance) -> tuple[ContextItem, ...]:
        return tuple(item for item in self.items if item.provenance is provenance)
