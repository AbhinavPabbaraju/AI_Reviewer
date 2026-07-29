"""Fusion ranking and budget enforcement (ARCHITECTURE sec. 4.3, steps 4-5).

``score = w_g * graph_proximity + w_s * cosine + w_r * co_change_recency``

The weights are stated here as data, with graph proximity weighted above
semantic similarity, because ADR-002 chose structure-first retrieval and the
ranking has to actually express that choice. They are **defaults, not truth**:
sec. 4.3 says weights are tuned against the eval corpus rather than chosen by
taste, and the M2 exit criterion (the file a human needs appears >= 90% of the
time) is what will move them.

Budget enforcement drops whole items, lowest score first. A pack is never
truncated mid-symbol: half an implementation is worse than none, because the
reviewer cannot tell that the half they are reading is incomplete and will
happily report a bug about the missing part.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.retrieval.models import ContextItem, Provenance

__all__ = ["DEFAULT_WEIGHTS", "FusionWeights", "enforce_budget", "fuse_score"]


@dataclass(frozen=True, slots=True)
class FusionWeights:
    graph: float = 0.6
    semantic: float = 0.3
    co_change: float = 0.1
    """Weight for how recently a file co-changed with the diff's files.

    Wired through the formula but always supplied as 0.0 today: co-change needs
    a commit-history index that nothing in M1 produces. It is kept in the
    signature rather than removed so that adding the signal is a change to one
    call site, and so the ranking never silently pretends to use evidence it
    does not have.
    """

    def __post_init__(self) -> None:
        for name, value in (
            ("graph", self.graph),
            ("semantic", self.semantic),
            ("co_change", self.co_change),
        ):
            if value < 0.0:
                raise ValueError(f"{name} weight must not be negative")
        if self.graph + self.semantic + self.co_change <= 0.0:
            raise ValueError("at least one fusion weight must be positive")


DEFAULT_WEIGHTS = FusionWeights()


def fuse_score(
    *,
    graph_proximity: float = 0.0,
    semantic_score: float = 0.0,
    co_change: float = 0.0,
    weights: FusionWeights = DEFAULT_WEIGHTS,
) -> float:
    """The fused rank for one candidate. Signals are independent: an item found
    both structurally and semantically scores on both, which is exactly the
    corroboration a hybrid retriever exists to reward."""
    return (
        weights.graph * graph_proximity
        + weights.semantic * semantic_score
        + weights.co_change * co_change
    )


def enforce_budget(
    items: Sequence[ContextItem], token_budget: int
) -> tuple[tuple[ContextItem, ...], int, int]:
    """Rank by score and keep whole items while they fit.

    Returns ``(kept, tokens_used, dropped)``. Anchors are kept unconditionally
    and charged against the budget: they are the diff itself, and a "context"
    pack that dropped the changed code to fit a caller in would be reviewing the
    wrong thing. If the anchors alone exceed the budget the pack is over budget
    and honest about it, rather than silently half-populated.

    Lower-scoring items are *skipped, not stopped at*: a single huge chunk near
    the top must not evict the ten small ones behind it that would all have fit.
    """
    anchors = [item for item in items if item.provenance is Provenance.ANCHOR]
    rest = sorted(
        (item for item in items if item.provenance is not Provenance.ANCHOR),
        key=lambda item: (-item.score, item.path, item.chunk.span.line_start),
    )

    kept = list(anchors)
    used = sum(item.token_count for item in anchors)
    dropped = 0
    for item in rest:
        if used + item.token_count <= token_budget:
            kept.append(item)
            used += item.token_count
        else:
            dropped += 1

    kept.sort(
        key=lambda item: (
            item.provenance is not Provenance.ANCHOR,
            -item.score,
            item.path,
            item.chunk.span.line_start,
        )
    )
    return tuple(kept), used, dropped
