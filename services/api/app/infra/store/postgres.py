"""Postgres :class:`IndexStorePort` -- the index store of record.

The counterpart to ``store/memory.py``, and held to the same standard: the
indexer must not be able to tell which one it is talking to. ``tests/infra/
test_store_parity.py`` runs both through the same scenarios for exactly that
reason.

Three things drive the shape of the code:

**Bulk loads go through COPY, not executemany.** A cold index of the 5,050-file
gate repository writes ~5k files, ~30k symbols, ~30k edges and ~29k chunks. That
is 90,000 rows, and 90,000 round trips would put the store on the critical path
of an exit criterion it has no business dominating. Each entity is COPYed into a
temp table and then moved with one ``INSERT ... SELECT``, which also gives the
joins (path -> file id, fqn -> symbol id) somewhere to happen in the server
rather than in Python.

**A snapshot write is one transaction.** ``0001_init.sql`` makes (repository,
commit, embedding model, parser version) unique, so re-indexing a commit is a
*replacement*, not a second row. Everything below drops the prior snapshot and
writes the new one atomically -- a half-written symbol graph would be worse than
no graph, because retrieval cannot tell the difference.

**Replacement must not destroy paid-for embeddings.** Chunks are deduplicated
per repository and survive the snapshot that first introduced them (0002); the
chunk upsert therefore never overwrites a stored vector with NULL. This is the
persistence half of the incrementality story -- the parse cache saves the
parsing, and this saves the embedding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, Final
from uuid import UUID

from app.domain.indexing.models import Chunk, Embedding, SourceFile, Symbol, SymbolEdge
from app.domain.indexing.ports import SnapshotRef, SnapshotWrite

__all__ = ["PostgresIndexStore"]

_READY: Final = "ready"


class PostgresIndexStore:
    """Implements :class:`app.domain.indexing.ports.IndexStorePort`."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # -- IndexStorePort --------------------------------------------------- #

    async def latest_snapshot(self, repository_id: UUID) -> SnapshotRef | None:
        row = await self._pool.fetchrow(
            """
            SELECT id, commit_sha FROM index_snapshots
            WHERE repository_id = $1 AND status = 'ready'
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            repository_id,
        )
        if row is None:
            return None
        return SnapshotRef(id=row["id"], commit_sha=row["commit_sha"])

    async def snapshot_file_blobs(self, snapshot_id: UUID) -> Mapping[str, str]:
        rows = await self._pool.fetch(
            "SELECT path, blob_sha FROM files WHERE snapshot_id = $1", snapshot_id
        )
        return {row["path"]: row["blob_sha"] for row in rows}

    async def known_chunk_hashes(self, repository_id: UUID) -> AbstractSet[str]:
        rows = await self._pool.fetch(
            "SELECT content_hash FROM chunks WHERE repository_id = $1", repository_id
        )
        return frozenset(row["content_hash"] for row in rows)

    async def save(self, snapshot: SnapshotWrite) -> None:
        async with self._pool.acquire() as connection, connection.transaction():
            stored = await self._stored_chunks(connection, snapshot)
            await self._write_snapshot_row(connection, snapshot, len(stored))
            await self._write_files(connection, snapshot)
            await self._write_symbols(connection, snapshot)
            await self._write_edges(connection, snapshot)
            await self._write_chunks(connection, snapshot, stored)

    # -- write steps ------------------------------------------------------ #

    async def _stored_chunks(
        self, connection: Any, snapshot: SnapshotWrite
    ) -> Mapping[str, bool]:
        """``content_hash -> already has an embedding``, for the chunks this
        snapshot carries that the repository already stores.

        Serves two purposes at once, which is why it runs before anything is
        written. Its size is ``chunks_reused`` -- measured against the store's
        state *before* the write, because a writer reporting its own reuse rate
        proves nothing, and the DDL claims that column proves incrementality
        works. Its contents decide which chunk rows have to be written at all.
        """
        if not snapshot.chunks:
            return {}
        rows = await connection.fetch(
            """
            SELECT content_hash, embedding IS NOT NULL AS embedded
            FROM chunks
            WHERE repository_id = $1 AND content_hash = ANY($2::char(64)[])
            """,
            snapshot.repository_id,
            [chunk.content_hash for chunk in snapshot.chunks],
        )
        return {row["content_hash"]: row["embedded"] for row in rows}

    async def _write_snapshot_row(
        self, connection: Any, snapshot: SnapshotWrite, reused: int
    ) -> None:
        # Re-indexing a commit replaces its snapshot. The UNIQUE constraint in
        # 0001 says there is one snapshot per (repo, commit, model, parser), and
        # the honest reading of a repeat is "that run is being done again" --
        # parsing is deterministic, so the replacement is byte-identical anyway.
        # Chunks are not collateral damage: 0002 detached them from `files`.
        await connection.execute(
            """
            DELETE FROM index_snapshots
            WHERE repository_id = $1 AND commit_sha = $2
              AND embedding_model = $3 AND parser_version = $4
            """,
            snapshot.repository_id,
            snapshot.commit_sha,
            snapshot.embedding_model,
            snapshot.parser_version,
        )
        await connection.execute(
            """
            INSERT INTO index_snapshots (
                id, repository_id, commit_sha, parent_snapshot_id, status,
                files_indexed, chunks_embedded, chunks_reused,
                embedding_model, parser_version
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            snapshot.id,
            snapshot.repository_id,
            snapshot.commit_sha,
            snapshot.parent_snapshot_id,
            _READY,
            len(snapshot.files),
            len(snapshot.embeddings),
            reused,
            snapshot.embedding_model,
            snapshot.parser_version,
        )

    async def _write_files(self, connection: Any, snapshot: SnapshotWrite) -> None:
        if not snapshot.files:
            return
        await _copy_into_temp(
            connection,
            "tmp_files",
            "path TEXT, blob_sha TEXT, language TEXT, size_bytes INT,"
            " line_count INT, is_test BOOL, is_generated BOOL",
            (_file_record(file) for file in snapshot.files),
        )
        await connection.execute(
            """
            INSERT INTO files (
                repository_id, snapshot_id, path, blob_sha, language,
                size_bytes, line_count, is_test, is_generated
            )
            SELECT $1, $2, path, blob_sha, language,
                   size_bytes, line_count, is_test, is_generated
            FROM tmp_files
            """,
            snapshot.repository_id,
            snapshot.id,
        )

    async def _write_symbols(self, connection: Any, snapshot: SnapshotWrite) -> None:
        if not snapshot.symbols:
            return
        await _copy_into_temp(
            connection,
            "tmp_symbols",
            "fqn TEXT, name TEXT, kind TEXT, path TEXT, line_start INT,"
            " line_end INT, signature TEXT, docstring TEXT, is_exported BOOL,"
            " parent_fqn TEXT",
            (_symbol_record(symbol) for symbol in snapshot.symbols),
        )
        # The join to `files` is what turns a symbol's path into a file id
        # without a second round trip per symbol. It is an inner join, and a
        # symbol whose file is not in this snapshot is a bug in the caller, so
        # the row count is checked rather than assumed.
        written = _rows_affected(
            await connection.execute(
                """
                INSERT INTO symbols (
                    repository_id, snapshot_id, file_id, fqn, name, kind,
                    line_start, line_end, signature, docstring, is_exported,
                    parent_fqn
                )
                SELECT $1, $2, f.id, t.fqn, t.name, t.kind::symbol_kind,
                       t.line_start, t.line_end, t.signature, t.docstring,
                       t.is_exported, t.parent_fqn
                FROM tmp_symbols t
                JOIN files f ON f.snapshot_id = $2 AND f.path = t.path
                """,
                snapshot.repository_id,
                snapshot.id,
            )
        )
        _require_all_written("symbols", written, len(snapshot.symbols))

    async def _write_edges(self, connection: Any, snapshot: SnapshotWrite) -> None:
        if not snapshot.edges:
            return
        await _copy_into_temp(
            connection,
            "tmp_edges",
            "src_fqn TEXT, dst_fqn TEXT, dst_unresolved_name TEXT,"
            " kind TEXT, confidence REAL",
            (_edge_record(edge) for edge in snapshot.edges),
        )
        # An edge whose resolved `dst_fqn` names no symbol in this snapshot would
        # arrive here with both targets NULL and be rejected by the DDL's
        # `edge_has_a_target` CHECK -- which is the correct outcome, and the
        # reason that constraint is worth having. An unresolved edge keeps its
        # textual name and is unaffected (ARCHITECTURE sec. 4.2).
        written = _rows_affected(
            await connection.execute(
                """
                INSERT INTO symbol_edges (
                    repository_id, src_symbol_id, dst_symbol_id,
                    dst_unresolved_name, kind, confidence
                )
                SELECT $1, src.id, dst.id, t.dst_unresolved_name,
                       t.kind::edge_kind, t.confidence
                FROM tmp_edges t
                JOIN symbols src ON src.snapshot_id = $2 AND src.fqn = t.src_fqn
                LEFT JOIN symbols dst
                       ON dst.snapshot_id = $2 AND dst.fqn = t.dst_fqn
                """,
                snapshot.repository_id,
                snapshot.id,
            )
        )
        _require_all_written("edges", written, len(snapshot.edges))

    async def _write_chunks(
        self, connection: Any, snapshot: SnapshotWrite, stored: Mapping[str, bool]
    ) -> None:
        """Write the chunk rows this snapshot actually changes, then membership.

        A chunk is immutable: its content hash covers the repository, path, symbol
        and normalized body, so a hash that is already stored has nothing new to
        say. Rewriting all of them anyway is what a straightforward upsert does,
        and on a one-file push it costs a full rewrite of ~29,000 rows -- with the
        bodies -- to change one. That is the difference between the incremental
        gate passing and failing, so the write set is narrowed to:

        * chunks the repository has never seen, and
        * chunks stored without a vector, for which this run supplies one (an
          embedder added after an unembedded index, or a run that failed between
          embedding and storing).

        Everything else needs only a membership row, which is two ids.
        """
        if not snapshot.chunks:
            return
        unique = _unique_by_hash(snapshot.chunks)
        pending = [
            chunk
            for chunk in unique
            if chunk.content_hash not in stored
            or (
                not stored[chunk.content_hash]
                and snapshot.embeddings.get(chunk.content_hash) is not None
            )
        ]
        if pending:
            await _copy_into_temp(
                connection,
                "tmp_chunks",
                "content_hash TEXT, symbol_fqn TEXT, path TEXT, language TEXT,"
                " line_start INT, line_end INT, token_count INT, content TEXT,"
                " embedding vector",
                (
                    _chunk_record(chunk, snapshot.embeddings.get(chunk.content_hash))
                    for chunk in pending
                ),
            )
            await connection.execute(
                """
                INSERT INTO chunks (
                    repository_id, file_id, symbol_id, content_hash, symbol_fqn,
                    path, language, line_start, line_end, token_count, content,
                    embedding
                )
                SELECT $1, f.id, s.id, t.content_hash, t.symbol_fqn,
                       t.path, t.language, t.line_start, t.line_end,
                       t.token_count, t.content, t.embedding
                FROM tmp_chunks t
                LEFT JOIN files f ON f.snapshot_id = $2 AND f.path = t.path
                LEFT JOIN symbols s
                       ON s.snapshot_id = $2 AND s.fqn = t.symbol_fqn
                ON CONFLICT (repository_id, content_hash) DO UPDATE SET
                    file_id   = EXCLUDED.file_id,
                    symbol_id = EXCLUDED.symbol_id,
                    -- Never overwrite a stored vector with NULL. A run configured
                    -- without an embedder still writes every *new* chunk it sees,
                    -- and losing an already-embedded chunk's vector because of it
                    -- would silently un-do the expensive half of indexing.
                    embedding = COALESCE(EXCLUDED.embedding, chunks.embedding)
                """,
                snapshot.repository_id,
                snapshot.id,
            )

        # Membership is what scopes retrieval to one commit; chunks themselves
        # are repository-scoped and shared between snapshots (0002). This is the
        # only part every chunk pays on every index, and it is two ids wide.
        await _copy_into_temp(
            connection,
            "tmp_chunk_hashes",
            "content_hash TEXT",
            ((chunk.content_hash,) for chunk in unique),
        )
        await connection.execute(
            """
            INSERT INTO snapshot_chunks (snapshot_id, chunk_id)
            SELECT $2, c.id
            FROM tmp_chunk_hashes t
            JOIN chunks c
              ON c.repository_id = $1 AND c.content_hash = t.content_hash
            ON CONFLICT DO NOTHING
            """,
            snapshot.repository_id,
            snapshot.id,
        )


# -- record shaping ------------------------------------------------------- #


def _file_record(file: SourceFile) -> tuple[Any, ...]:
    return (
        file.path,
        file.blob_sha,
        file.language.value,
        file.size_bytes,
        file.line_count,
        file.is_test,
        file.is_generated,
    )


def _symbol_record(symbol: Symbol) -> tuple[Any, ...]:
    return (
        symbol.fqn,
        symbol.name,
        symbol.kind.value,
        symbol.span.path,
        symbol.span.line_start,
        symbol.span.line_end,
        symbol.signature,
        symbol.docstring,
        symbol.is_exported,
        symbol.parent_fqn,
    )


def _edge_record(edge: SymbolEdge) -> tuple[Any, ...]:
    return (
        edge.src_fqn,
        edge.dst_fqn,
        edge.dst_unresolved_name,
        edge.kind.value,
        edge.confidence,
    )


def _chunk_record(chunk: Chunk, embedding: Embedding | None) -> tuple[Any, ...]:
    return (
        chunk.content_hash,
        chunk.symbol_fqn,
        chunk.span.path,
        chunk.language.value,
        chunk.span.line_start,
        chunk.span.line_end,
        chunk.token_count,
        chunk.content,
        embedding,
    )


def _unique_by_hash(chunks: Sequence[Chunk]) -> list[Chunk]:
    """One row per content hash.

    A snapshot legitimately carries the same chunk twice -- two identical
    ``__init__`` bodies in different files hash differently (the path is in the
    hash), but a file listed twice by a cold-cache fallback does not. Letting the
    duplicate reach ``ON CONFLICT`` would raise "cannot affect row a second
    time", so it is collapsed before COPY rather than papered over after.
    """
    seen: dict[str, Chunk] = {}
    for chunk in chunks:
        seen.setdefault(chunk.content_hash, chunk)
    return list(seen.values())


# -- COPY plumbing --------------------------------------------------------- #


async def _copy_into_temp(
    connection: Any,
    table: str,
    columns_ddl: str,
    records: Iterable[tuple[Any, ...]],
) -> None:
    """Create a per-transaction temp table and COPY ``records`` into it.

    ``ON COMMIT DROP`` rather than an explicit cleanup: the temp table's lifetime
    is exactly the snapshot write's transaction, and tying it to the commit means
    a rolled-back write leaves nothing behind to collide with the retry.
    """
    await connection.execute(
        f"CREATE TEMP TABLE {table} ({columns_ddl}) ON COMMIT DROP"
    )
    await connection.copy_records_to_table(table, records=records)


def _rows_affected(status: str) -> int:
    """asyncpg returns the raw command tag ("INSERT 0 4212")."""
    return int(status.rsplit(" ", 1)[-1])


def _require_all_written(entity: str, written: int, expected: int) -> None:
    if written != expected:
        raise ValueError(
            f"persisted {written} of {expected} {entity}: the join that resolves "
            f"them dropped {expected - written} rows. A silently smaller graph is "
            f"a retrieval bug that surfaces as a missing review comment."
        )
