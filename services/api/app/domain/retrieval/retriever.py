"""Stage III orchestration: diff hunks in, context pack out (sec. 4.3).

The order is the architecture's argument made executable:

1. **Anchor** on the symbols the diff actually touches.
2. **Expand** the symbol graph around them (structure first).
3. **Supplement** with vector search, and only with material the graph could not
   reach -- a semantically similar copy of a symbol already in the pack is not
   context, it is duplication.
4. **Fuse** the signals into one ranking.
5. **Enforce** the token budget by dropping whole symbols.

Pure domain: every external thing it needs (the symbol index, the vector store,
the embedder) arrives as a port, which is what lets the M6 harness run this
against fakes and get identical results every time.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.contracts import CodeSpan
from app.domain.indexing.models import Chunk, SymbolEdge
from app.domain.ports import EmbeddingPort, VectorStorePort
from app.domain.retrieval.expansion import (
    DEFAULT_EXPANSION,
    EdgeLoader,
    ExpansionConfig,
    Neighbour,
    expand,
)
from app.domain.retrieval.fusion import DEFAULT_WEIGHTS, FusionWeights, enforce_budget, fuse_score
from app.domain.retrieval.models import (
    ContextItem,
    ContextPack,
    Provenance,
    RetrievalStats,
)
from app.domain.retrieval.ports import SymbolIndexPort

__all__ = ["ContextRetriever", "RetrievalConfig"]


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    token_budget: int = 12_000
    """Ceiling for the pack. The review prompt has to fit around it, so this is a
    retrieval budget, not a context-window budget."""
    semantic_limit: int = 20
    expansion: ExpansionConfig = DEFAULT_EXPANSION
    weights: FusionWeights = DEFAULT_WEIGHTS

    def __post_init__(self) -> None:
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.semantic_limit < 0:
            raise ValueError("semantic_limit must not be negative")


class ContextRetriever:
    """Builds context packs. One instance is reusable across requests."""

    def __init__(
        self,
        *,
        index: SymbolIndexPort,
        vectors: VectorStorePort | None = None,
        embeddings: EmbeddingPort | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._index = index
        self._vectors = vectors
        self._embeddings = embeddings
        self._config = config or RetrievalConfig()

    async def retrieve(
        self, *, repository_id: str, hunks: Sequence[CodeSpan]
    ) -> ContextPack:
        started = time.perf_counter()
        config = self._config

        anchor_symbols = await self._index.symbols_in_spans(repository_id, hunks)
        anchor_fqns = [symbol.fqn for symbol in anchor_symbols]
        anchor_chunks = await self._index.chunks_for_symbols(
            repository_id, anchor_fqns
        )
        anchor_items = [
            ContextItem(
                chunk=chunk,
                provenance=Provenance.ANCHOR,
                score=1.0,
                graph_distance=0,
                graph_proximity=1.0,
                reason="changed by this diff",
            )
            for chunk in anchor_chunks
        ]

        neighbours = await expand(
            anchor_fqns, self._edge_loader(repository_id), config.expansion
        )
        graph_items = await self._graph_items(repository_id, neighbours)

        covered_paths = {item.path for item in (*anchor_items, *graph_items)}
        covered_hashes = {
            item.chunk.content_hash for item in (*anchor_items, *graph_items)
        }
        semantic_items = await self._semantic_items(
            repository_id, anchor_chunks, covered_paths, covered_hashes
        )

        candidates = [*anchor_items, *graph_items, *semantic_items]
        kept, tokens_used, dropped = enforce_budget(candidates, config.token_budget)

        return ContextPack(
            repository_id=repository_id,
            hunks=tuple(hunks),
            items=kept,
            stats=RetrievalStats(
                anchors=len(anchor_items),
                graph_candidates=len(graph_items),
                semantic_candidates=len(semantic_items),
                dropped_by_budget=dropped,
                tokens_used=tokens_used,
                token_budget=config.token_budget,
                duration_ms=round((time.perf_counter() - started) * 1000),
            ),
        )

    # -- stages ----------------------------------------------------------- #

    def _edge_loader(self, repository_id: str) -> EdgeLoader:
        async def load(fqns: Sequence[str]) -> Sequence[SymbolEdge]:
            return await self._index.edges_touching(repository_id, fqns)

        return load

    async def _graph_items(
        self, repository_id: str, neighbours: Sequence[Neighbour]
    ) -> list[ContextItem]:
        if not neighbours:
            return []
        by_fqn: Mapping[str, Neighbour] = {n.fqn: n for n in neighbours}
        chunks = await self._index.chunks_for_symbols(repository_id, list(by_fqn))
        items: list[ContextItem] = []
        for chunk in chunks:
            neighbour = by_fqn.get(chunk.symbol_fqn or "")
            if neighbour is None:
                continue
            items.append(
                ContextItem(
                    chunk=chunk,
                    provenance=Provenance.GRAPH,
                    score=fuse_score(
                        graph_proximity=neighbour.proximity,
                        weights=self._config.weights,
                    ),
                    graph_distance=neighbour.distance,
                    graph_proximity=neighbour.proximity,
                    reason=neighbour.reason,
                )
            )
        return items

    async def _semantic_items(
        self,
        repository_id: str,
        anchor_chunks: Sequence[Chunk],
        covered_paths: set[str],
        covered_hashes: set[str],
    ) -> list[ContextItem]:
        """Vector search for what the graph could not reach.

        Anchor *paths* are excluded at the store, and anything the graph already
        pulled in is dropped here: the supplement's job is coverage the structure
        missed, and paying budget for a second copy of a symbol already in the
        pack is the failure mode of pure-vector retrieval.
        """
        config = self._config
        if (
            self._vectors is None
            or self._embeddings is None
            or config.semantic_limit == 0
            or not anchor_chunks
        ):
            return []

        query = "\n\n".join(chunk.content for chunk in anchor_chunks)
        [vector] = await self._embeddings.embed([query])
        matches = await self._vectors.search(
            repository_id,
            vector,
            limit=config.semantic_limit,
            exclude_paths=sorted({chunk.path for chunk in anchor_chunks}),
        )

        wanted = {
            match.chunk_id: match
            for match in matches
            if match.chunk_id not in covered_hashes and match.path not in covered_paths
        }
        if not wanted:
            return []

        chunks = await self._index.chunks_by_hash(repository_id, list(wanted))
        items: list[ContextItem] = []
        for chunk in chunks:
            match = wanted[chunk.content_hash]
            similarity = min(max(match.score, 0.0), 1.0)
            items.append(
                ContextItem(
                    chunk=chunk,
                    provenance=Provenance.SEMANTIC,
                    score=fuse_score(
                        semantic_score=similarity, weights=config.weights
                    ),
                    semantic_score=similarity,
                    reason=f"similar to the changed code (cosine {similarity:.2f})",
                )
            )
        return items
