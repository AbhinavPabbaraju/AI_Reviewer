"""Fixtures for the Postgres/pgvector adapter tests.

These tests need a real database -- pgvector's ANN operator, native enums, the
CHECK constraints that mirror the domain validators, and ``COPY`` are all things
a fake would have to reimplement, at which point it would be testing the fake.
So they skip cleanly when ``ARGUS_TEST_DATABASE_URL`` is unset, and the default
``pytest`` run stays offline and fast.

    ARGUS_TEST_DATABASE_URL=postgresql://argus@/argus_test .venv/bin/python -m pytest

The database is **truncated between tests**, not recreated: migrations run once
per session, and each test starts from an empty schema. Tenancy rows come from
``pg_repository`` because the index store deliberately does not create them --
``index_snapshots.repository_id`` is a foreign key, and a store that invented
its own tenants to satisfy it would be routing around the constraint that keeps
one installation's code out of another's index.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any, Final
from uuid import UUID, uuid4

import pytest

from app.infra.db.migrate import apply_migrations
from app.infra.db.pool import close_pool, create_pool

DSN_ENV: Final = "ARGUS_TEST_DATABASE_URL"

_SKIP_REASON: Final = (
    f"set {DSN_ENV} to a Postgres 16+ database with pgvector to run the "
    "Postgres adapter tests"
)

# Order is irrelevant under CASCADE, but listing every table means a new table
# that nobody adds here fails loudly as cross-test leakage rather than quietly
# accumulating rows.
_TABLES: Final = (
    "installations",
    "repositories",
    "index_snapshots",
    "files",
    "symbols",
    "symbol_edges",
    "chunks",
    "snapshot_chunks",
)


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """The DSN, with migrations applied once. Skips the suite when unset.

    Migrations run through ``asyncio.run`` in a *sync* fixture on purpose: it
    keeps this fixture free of any event-loop scoping relationship with the
    function-scoped pools below.
    """
    dsn = os.environ.get(DSN_ENV)
    if not dsn:
        pytest.skip(_SKIP_REASON)
    asyncio.run(apply_migrations(dsn))
    return dsn


@pytest.fixture
async def pg_pool(pg_dsn: str) -> AsyncIterator[Any]:
    pool = await create_pool(pg_dsn, min_size=1, max_size=4)
    try:
        await pool.execute(
            f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE"
        )
        yield pool
    finally:
        await close_pool(pool)


@pytest.fixture
async def pg_repository(pg_pool: Any) -> UUID:
    """An installation and a repository to hang snapshots off."""
    return await create_repository(pg_pool)


async def create_repository(pool: Any, repository_id: UUID | None = None) -> UUID:
    """Insert the tenancy rows a snapshot's foreign keys require."""
    repository_id = repository_id or uuid4()
    installation_id = uuid4()
    github_id = abs(hash(installation_id)) % 1_000_000_000
    await pool.execute(
        """
        INSERT INTO installations (
            id, github_installation_id, account_login, account_type
        ) VALUES ($1, $2, 'argus-test', 'Organization')
        """,
        installation_id,
        github_id,
    )
    await pool.execute(
        """
        INSERT INTO repositories (
            id, installation_id, github_repo_id, owner, name
        ) VALUES ($1, $2, $3, 'argus', $4)
        """,
        repository_id,
        installation_id,
        github_id,
        f"fixture-{github_id}",
    )
    return repository_id
