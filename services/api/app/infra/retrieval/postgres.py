"""Postgres/pgvector :class:`SymbolIndexPort` and :class:`VectorStorePort`.

The production counterpart to ``retrieval/memory.py``, and the reason the
retrieval ports are shaped the way they are. Three properties carry over from
the in-memory adapter deliberately, because the M6 harness will run the pipeline
against one and production against the other:

* **Every query is scoped twice** -- by repository *and* by snapshot. Repository
  scoping is the tenancy boundary (an unfiltered ANN search across tenants is a
  data leak, which is why ``repository_id`` is the first positional argument of
  ``VectorStorePort.search``). Snapshot scoping is the correctness boundary: a
  repository accumulates a symbol row per commit indexed, and a pack built from
  the union of every commit would be reviewing code that no longer exists.
* **Result ordering matches the in-memory adapter**, via ``array_position``.
  Fusion breaks score ties by input order, so an adapter that returned the same
  rows in a different order would build a subtly different pack at the budget
  boundary -- and the retrieval-quality gate would measure a different system
  than the one that ships.
* **One query per BFS level**, never one per symbol. ``edges_touching`` takes
  the whole level's fqns and resolves them to symbol ids, walks both edge
  directions, and maps back to fqns in a single statement.

Unlike the in-memory adapter's O(chunks) scan, ANN here is an HNSW index scan
(``vector_cosine_ops``, per 0001). Cosine *distance* is what the operator
returns; the port is specified in similarity, so the query converts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self
from uuid import UUID

from app.domain.contracts import CodeSpan
from app.domain.indexing.models import (
    Chunk,
    EdgeKind,
    Language,
    Symbol,
    SymbolEdge,
    SymbolKind,
)
from app.domain.ports import ChunkMatch

__all__ = [
    "PostgresSymbolIndex",
    "PostgresVectorStore",
    "ServingSnapshot",
    "latest_ready_snapshot",
]


@dataclass(frozen=True, slots=True)
class ServingSnapshot:
    """The snapshot a read is being served from.

    Carries ``embedding_model`` because a query vector is only comparable to
    stored vectors produced by the same model. Anything embedding a query has to
    be able to check that, and the snapshot is the only place that records what
    actually embedded the corpus.
    """

    id: UUID
    commit_sha: str
    embedding_model: str


async def latest_ready_snapshot(
    pool: Any, repository_id: UUID
) -> ServingSnapshot | None:
    """The snapshot retrieval should serve for a repository.

    Separate from :class:`app.infra.store.postgres.PostgresIndexStore` because
    the read side must not depend on the write side; both express the same
    "newest ready snapshot" rule against the partial index in 0001.
    """
    row = await pool.fetchrow(
        """
        SELECT id, commit_sha, embedding_model FROM index_snapshots
        WHERE repository_id = $1 AND status = 'ready'
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        repository_id,
    )
    if row is None:
        return None
    return ServingSnapshot(
        id=UUID(str(row["id"])),
        commit_sha=row["commit_sha"],
        embedding_model=row["embedding_model"],
    )


class _SnapshotScoped:
    """Shared repository/snapshot binding and the tenancy check."""

    def __init__(self, pool: Any, repository_id: UUID, snapshot_id: UUID) -> None:
        self._pool = pool
        self._repository_id = repository_id
        self._snapshot_id = snapshot_id

    @property
    def snapshot_id(self) -> UUID:
        return self._snapshot_id

    def _owns(self, repository_id: str) -> bool:
        """A mismatched id returns nothing rather than raising, matching the
        in-memory adapter: the caller asked another tenant's index a question,
        and the only safe answer is silence."""
        return repository_id == str(self._repository_id)


class PostgresSymbolIndex(_SnapshotScoped):
    """Serves one snapshot's symbol graph and chunks."""

    @classmethod
    async def for_latest(cls, pool: Any, repository_id: UUID) -> Self | None:
        """Bind to the repository's newest ready snapshot, or ``None`` if it has
        never been indexed -- which is a real state (a fresh installation), not
        an error, and the caller has to decide what to do about it."""
        serving = await latest_ready_snapshot(pool, repository_id)
        if serving is None:
            return None
        return cls(pool, repository_id, serving.id)

    # -- SymbolIndexPort --------------------------------------------------- #

    async def symbols_in_spans(
        self, repository_id: str, spans: Sequence[CodeSpan]
    ) -> Sequence[Symbol]:
        if not self._owns(repository_id) or not spans:
            return []
        # Overlap, not containment: a three-line hunk in the middle of a function
        # must anchor on the whole function. `unnest` turns the span list into a
        # joinable relation so N hunks stay one round trip.
        rows = await self._pool.fetch(
            """
            SELECT DISTINCT s.fqn, s.name, s.kind, s.line_start, s.line_end,
                   s.signature, s.docstring, s.is_exported, s.parent_fqn,
                   f.path, f.language
            FROM symbols s
            JOIN files f ON f.id = s.file_id
            JOIN unnest($2::text[], $3::int[], $4::int[])
                 AS hunk(path, line_start, line_end)
              ON hunk.path = f.path
             AND s.line_start <= hunk.line_end
             AND hunk.line_start <= s.line_end
            WHERE s.snapshot_id = $1
            ORDER BY f.path, s.line_start
            """,
            self._snapshot_id,
            [span.path for span in spans],
            [span.line_start for span in spans],
            [span.line_end for span in spans],
        )
        return [_to_symbol(row) for row in rows]

    async def edges_touching(
        self, repository_id: str, fqns: Sequence[str]
    ) -> Sequence[SymbolEdge]:
        if not self._owns(repository_id) or not fqns:
            return []
        # Both directions in one statement: callees say what the changed code
        # depends on, callers say who depends on it, and the caller is usually
        # where the bug becomes visible.
        rows = await self._pool.fetch(
            """
            WITH wanted AS (
                SELECT id FROM symbols
                WHERE snapshot_id = $1 AND fqn = ANY($2::text[])
            )
            SELECT e.kind, src.fqn AS src_fqn, dst.fqn AS dst_fqn,
                   e.dst_unresolved_name, e.confidence
            FROM symbol_edges e
            JOIN symbols src ON src.id = e.src_symbol_id
            LEFT JOIN symbols dst ON dst.id = e.dst_symbol_id
            WHERE src.snapshot_id = $1
              AND (e.src_symbol_id IN (SELECT id FROM wanted)
                   OR e.dst_symbol_id IN (SELECT id FROM wanted))
            ORDER BY e.id
            """,
            self._snapshot_id,
            list(fqns),
        )
        return [_to_edge(row) for row in rows]

    async def chunks_for_symbols(
        self, repository_id: str, fqns: Sequence[str]
    ) -> Sequence[Chunk]:
        if not self._owns(repository_id) or not fqns:
            return []
        rows = await self._pool.fetch(
            f"""
            SELECT {_CHUNK_COLUMNS}
            FROM chunks c
            JOIN snapshot_chunks sc ON sc.chunk_id = c.id AND sc.snapshot_id = $1
            WHERE c.repository_id = $2 AND c.symbol_fqn = ANY($3::text[])
            ORDER BY array_position($3::text[], c.symbol_fqn), c.line_start
            """,
            self._snapshot_id,
            self._repository_id,
            list(fqns),
        )
        return [_to_chunk(row) for row in rows]

    async def chunks_by_hash(
        self, repository_id: str, content_hashes: Sequence[str]
    ) -> Sequence[Chunk]:
        if not self._owns(repository_id) or not content_hashes:
            return []
        rows = await self._pool.fetch(
            f"""
            SELECT {_CHUNK_COLUMNS}
            FROM chunks c
            JOIN snapshot_chunks sc ON sc.chunk_id = c.id AND sc.snapshot_id = $1
            WHERE c.repository_id = $2 AND c.content_hash = ANY($3::text[])
            ORDER BY array_position($3::text[], c.content_hash)
            """,
            self._snapshot_id,
            self._repository_id,
            list(content_hashes),
        )
        return [_to_chunk(row) for row in rows]


class PostgresVectorStore(_SnapshotScoped):
    """Cosine ANN over one snapshot's chunk embeddings (HNSW, per 0001)."""

    @classmethod
    async def for_latest(cls, pool: Any, repository_id: UUID) -> Self | None:
        serving = await latest_ready_snapshot(pool, repository_id)
        if serving is None:
            return None
        return cls(pool, repository_id, serving.id)

    async def search(
        self,
        repository_id: str,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        exclude_paths: Sequence[str] = (),
    ) -> Sequence[ChunkMatch]:
        if not self._owns(repository_id) or limit <= 0:
            return []
        # `1 - (a <=> b)` converts pgvector's cosine *distance* into the cosine
        # *similarity* the port is specified in. Chunks with no vector are
        # excluded rather than scored zero -- an unembedded chunk is missing
        # data, not a bad match, and the graph half of retrieval still reaches it.
        rows = await self._pool.fetch(
            f"""
            SELECT {_CHUNK_COLUMNS},
                   1 - (c.embedding <=> $3) AS score
            FROM chunks c
            JOIN snapshot_chunks sc ON sc.chunk_id = c.id AND sc.snapshot_id = $1
            WHERE c.repository_id = $2
              AND c.embedding IS NOT NULL
              AND NOT (c.path = ANY($4::text[]))
            ORDER BY c.embedding <=> $3, c.content_hash
            LIMIT $5
            """,
            self._snapshot_id,
            self._repository_id,
            list(query_vector),
            list(exclude_paths),
            limit,
        )
        return [
            ChunkMatch(
                chunk_id=row["content_hash"],
                path=row["path"],
                line_start=row["line_start"],
                line_end=row["line_end"],
                content=row["content"],
                score=float(row["score"]),
                symbol_fqn=row["symbol_fqn"],
            )
            for row in rows
        ]


# -- row -> domain --------------------------------------------------------- #

_CHUNK_COLUMNS = (
    "c.content_hash, c.symbol_fqn, c.path, c.language, "
    "c.line_start, c.line_end, c.token_count, c.content"
)


def _to_symbol(row: Any) -> Symbol:
    return Symbol(
        fqn=row["fqn"],
        name=row["name"],
        kind=SymbolKind(row["kind"]),
        span=CodeSpan(
            path=row["path"],
            line_start=row["line_start"],
            line_end=row["line_end"],
        ),
        language=Language(row["language"]),
        signature=row["signature"],
        docstring=row["docstring"],
        is_exported=row["is_exported"],
        parent_fqn=row["parent_fqn"],
    )


def _to_edge(row: Any) -> SymbolEdge:
    return SymbolEdge(
        kind=EdgeKind(row["kind"]),
        src_fqn=row["src_fqn"],
        dst_fqn=row["dst_fqn"],
        dst_unresolved_name=row["dst_unresolved_name"],
        confidence=row["confidence"],
    )


def _to_chunk(row: Any) -> Chunk:
    return Chunk(
        content_hash=row["content_hash"],
        symbol_fqn=row["symbol_fqn"],
        span=CodeSpan(
            path=row["path"],
            line_start=row["line_start"],
            line_end=row["line_end"],
        ),
        language=Language(row["language"]),
        token_count=row["token_count"],
        content=row["content"],
    )
