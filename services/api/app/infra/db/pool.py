"""Connection pool and the pgvector wire codec.

Two things live here that every Postgres adapter needs and neither should own:
a pool whose connections all agree on how to talk about vectors, and the
``vector`` type codec itself.

The codec is registered in binary form on purpose. pgvector accepts vectors as
text (``'[0.1,0.2,...]'``), which is what most examples show, but a 1536-dim
float4 vector is ~18 KB of text against 6 KB of binary, and a cold index writes
one per chunk. More importantly, ``COPY`` -- the only way to load 30,000 chunks
without 30,000 round trips -- is a binary protocol, so a text codec would have
forced the bulk path onto ``executemany``. The format is public and stable:
``int16 dim, int16 unused, dim x float4``, all big-endian.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final, cast

import asyncpg

if TYPE_CHECKING:
    from asyncpg import Connection, Pool

__all__ = ["close_pool", "create_pool", "register_codecs"]

_VECTOR_HEADER: Final = struct.Struct(">HH")

# Postgres defaults to 8 MB of work memory for an HNSW scan; a 1536-dim index
# wants more before it will keep the whole candidate list in memory. Set per
# connection rather than per query so the ANN path has no per-call ceremony.
_HNSW_EF_SEARCH: Final = 100


def _encode_vector(value: Sequence[float]) -> bytes:
    return _VECTOR_HEADER.pack(len(value), 0) + struct.pack(
        f">{len(value)}f", *value
    )


def _decode_vector(data: bytes) -> tuple[float, ...]:
    dimensions, _ = _VECTOR_HEADER.unpack_from(data, 0)
    return cast(
        tuple[float, ...],
        struct.unpack_from(f">{dimensions}f", data, _VECTOR_HEADER.size),
    )


async def register_codecs(connection: Connection[Any]) -> None:
    """Teach one connection the ``vector`` type. Idempotent per connection.

    Tolerant of the extension being absent so that a pool can be opened against
    a database whose migrations have not run yet -- the migration runner needs a
    connection before ``CREATE EXTENSION vector`` has ever executed, and failing
    there would make bootstrapping a chicken-and-egg problem.
    """
    try:
        await connection.set_type_codec(
            "vector",
            schema="public",
            encoder=_encode_vector,
            decoder=_decode_vector,
            format="binary",
        )
    except ValueError:  # extension not installed yet
        return
    await connection.execute(f"SET hnsw.ef_search = {_HNSW_EF_SEARCH}")


async def create_pool(
    dsn: str, *, min_size: int = 1, max_size: int = 10
) -> Pool[Any]:
    """Open a pool whose every connection speaks ``vector``."""
    pool = await asyncpg.create_pool(
        dsn, min_size=min_size, max_size=max_size, init=register_codecs
    )
    if pool is None:  # pragma: no cover - asyncpg only returns None on bad args
        raise RuntimeError(f"could not open a connection pool to {dsn!r}")
    return cast("Pool[Any]", pool)


async def close_pool(pool: Pool[Any]) -> None:
    await pool.close()
