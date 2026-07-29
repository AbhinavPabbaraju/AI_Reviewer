"""Ports for Stage III: what retrieval needs from the index.

Deliberately shaped as *set-at-a-time* queries, one per BFS level rather than one
per symbol. A depth-2 expansion is then two round trips whatever the fan-out,
which is what keeps p95 retrieval inside its 800 ms budget when a popular
utility has four hundred callers. A neighbours-of-one-symbol port would have
been a friendlier interface and an N+1 query in production.

The vector half of retrieval uses :class:`app.domain.ports.VectorStorePort` from
M0 unchanged -- its ``repository_id``-first signature is what makes an unfiltered
cross-tenant ANN search impossible to write by accident.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.domain.contracts import CodeSpan
from app.domain.indexing.models import Chunk, Symbol, SymbolEdge

__all__ = ["SymbolIndexPort"]


@runtime_checkable
class SymbolIndexPort(Protocol):
    """Read access to one repository's symbol graph and chunks."""

    async def symbols_in_spans(
        self, repository_id: str, spans: Sequence[CodeSpan]
    ) -> Sequence[Symbol]:
        """Every symbol whose body overlaps one of ``spans``.

        Overlap, not containment: a hunk that changes three lines in the middle
        of a function must anchor on that whole function, and a hunk spanning a
        class body anchors on the class *and* the methods it covers.
        """
        ...

    async def edges_touching(
        self, repository_id: str, fqns: Sequence[str]
    ) -> Sequence[SymbolEdge]:
        """Every edge with one endpoint in ``fqns``, in either direction.

        Both directions matter and for different reasons: callees say what the
        changed code depends on, callers say who depends on *it* -- and the
        caller is usually where the bug becomes visible.
        """
        ...

    async def chunks_for_symbols(
        self, repository_id: str, fqns: Sequence[str]
    ) -> Sequence[Chunk]:
        """The chunk bodies for those symbols, for packing into the context."""
        ...

    async def chunks_by_hash(
        self, repository_id: str, content_hashes: Sequence[str]
    ) -> Sequence[Chunk]:
        """Chunks by content hash, for turning ANN hits into pack items.

        The vector store answers with ids and scores; the bodies, spans and
        languages come from here rather than being reconstructed from the match,
        so a semantic item is the same trustworthy ``Chunk`` a graph item is.
        """
        ...
