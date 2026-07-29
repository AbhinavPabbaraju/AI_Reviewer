"""In-memory :class:`SymbolIndexPort` and :class:`VectorStorePort` over a snapshot.

Complete implementations, not stubs -- the same posture as ``store/memory.py``.
They are what the M2 retrieval-quality gate runs against, and what the M6 eval
harness will run the whole pipeline against, so they have to behave like the
Postgres/pgvector adapters will:

* every query is scoped by ``repository_id``, and a mismatched id returns
  nothing rather than leaking another tenant's code;
* ANN search ranks by cosine over normalized vectors, the same metric as the
  DDL's ``hnsw (embedding vector_cosine_ops)`` index;
* symbol lookup by span uses overlap, so a three-line hunk anchors on the whole
  enclosing function.

The linear scan here is O(chunks); pgvector's HNSW is not. That difference is
the reason the port exists, and the reason retrieval latency is measured against
the real adapter rather than this one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from uuid import UUID

from app.domain.contracts import CodeSpan
from app.domain.indexing.models import Chunk, Embedding, Symbol, SymbolEdge
from app.domain.indexing.ports import SnapshotWrite
from app.domain.ports import ChunkMatch

__all__ = ["InMemorySymbolIndex", "InMemoryVectorStore"]


class InMemorySymbolIndex:
    """Serves one repository's symbol graph and chunks from a snapshot."""

    def __init__(self, snapshot: SnapshotWrite) -> None:
        self._repository_id = str(snapshot.repository_id)
        self._symbols: dict[str, Symbol] = {s.fqn: s for s in snapshot.symbols}
        self._edges: tuple[SymbolEdge, ...] = snapshot.edges
        self._chunks_by_hash: dict[str, Chunk] = {
            chunk.content_hash: chunk for chunk in snapshot.chunks
        }
        self._chunks_by_fqn: dict[str, list[Chunk]] = {}
        for chunk in snapshot.chunks:
            if chunk.symbol_fqn is not None:
                self._chunks_by_fqn.setdefault(chunk.symbol_fqn, []).append(chunk)

        # Edges are indexed by both endpoints: expansion walks callers as well
        # as callees, and scanning the edge list per BFS level would make the
        # fake behave nothing like the indexed table it stands in for.
        self._by_endpoint: dict[str, list[SymbolEdge]] = {}
        for edge in snapshot.edges:
            self._by_endpoint.setdefault(edge.src_fqn, []).append(edge)
            if edge.dst_fqn is not None:
                self._by_endpoint.setdefault(edge.dst_fqn, []).append(edge)

    # -- SymbolIndexPort -------------------------------------------------- #

    async def symbols_in_spans(
        self, repository_id: str, spans: Sequence[CodeSpan]
    ) -> Sequence[Symbol]:
        if not self._owns(repository_id):
            return []
        found: dict[str, Symbol] = {}
        for span in spans:
            for symbol in self._symbols.values():
                if symbol.span.path != span.path:
                    continue
                if _overlaps(symbol.span, span):
                    found[symbol.fqn] = symbol
        return sorted(found.values(), key=lambda s: (s.span.path, s.span.line_start))

    async def edges_touching(
        self, repository_id: str, fqns: Sequence[str]
    ) -> Sequence[SymbolEdge]:
        if not self._owns(repository_id):
            return []
        seen: dict[int, SymbolEdge] = {}
        for fqn in fqns:
            for edge in self._by_endpoint.get(fqn, ()):
                seen[id(edge)] = edge
        return tuple(seen.values())

    async def chunks_for_symbols(
        self, repository_id: str, fqns: Sequence[str]
    ) -> Sequence[Chunk]:
        if not self._owns(repository_id):
            return []
        return [
            chunk for fqn in fqns for chunk in self._chunks_by_fqn.get(fqn, ())
        ]

    async def chunks_by_hash(
        self, repository_id: str, content_hashes: Sequence[str]
    ) -> Sequence[Chunk]:
        if not self._owns(repository_id):
            return []
        return [
            chunk
            for content_hash in content_hashes
            if (chunk := self._chunks_by_hash.get(content_hash)) is not None
        ]

    def _owns(self, repository_id: str) -> bool:
        return repository_id == self._repository_id


class InMemoryVectorStore:
    """Cosine ANN over a snapshot's embeddings, filtered by repository."""

    def __init__(self, snapshot: SnapshotWrite) -> None:
        self._repository_id = str(snapshot.repository_id)
        self._chunks: dict[str, Chunk] = {
            chunk.content_hash: chunk for chunk in snapshot.chunks
        }
        self._vectors: Mapping[str, Embedding] = snapshot.embeddings

    async def search(
        self,
        repository_id: str,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        exclude_paths: Sequence[str] = (),
    ) -> Sequence[ChunkMatch]:
        if repository_id != self._repository_id:
            return []
        excluded = set(exclude_paths)
        scored: list[tuple[float, Chunk]] = []
        for content_hash, vector in self._vectors.items():
            chunk = self._chunks.get(content_hash)
            if chunk is None or chunk.path in excluded:
                continue
            scored.append((_cosine(query_vector, vector), chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].content_hash))
        return [
            ChunkMatch(
                chunk_id=chunk.content_hash,
                path=chunk.path,
                line_start=chunk.span.line_start,
                line_end=chunk.span.line_end,
                content=chunk.content,
                score=score,
                symbol_fqn=chunk.symbol_fqn,
            )
            for score, chunk in scored[:limit]
        ]


def _overlaps(left: CodeSpan, right: CodeSpan) -> bool:
    return left.line_start <= right.line_end and right.line_start <= left.line_end


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def repository_key(repository_id: UUID) -> str:
    """The string form of a repository id used across the retrieval ports."""
    return str(repository_id)
