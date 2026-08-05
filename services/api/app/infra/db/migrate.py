"""Apply ``db/migrations/*.sql`` in order, once each.

Deliberately about forty lines. ROADMAP puts "Alembic in CI" in M8 and this does
not pre-empt it: the migrations are already plain forward-only SQL files
(ARCHITECTURE sec. 10 -- the interesting parts are constraints and index
strategy, which are unreadable through a migration DSL), so what was missing was
never a migration *framework*, only something to run them and remember which
ones ran. Alembic can adopt the same files and the same ledger table later.

Each file is applied inside its own transaction *together with* its ledger row,
so a crash can never leave a schema change that the ledger has forgotten (which
would make the next run replay a migration that already succeeded).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import asyncpg

__all__ = ["MIGRATIONS_DIR", "apply_migrations", "migration_files"]

# app/infra/db/migrate.py -> services/api/app/infra/db -> repository root.
MIGRATIONS_DIR: Final = Path(__file__).resolve().parents[5] / "db" / "migrations"

_LEDGER: Final = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


# The migration files are readable as standalone scripts (`psql -f` should work
# on any one of them), so each wraps itself in BEGIN/COMMIT. The runner takes the
# transaction over instead, because the ledger write has to commit with the DDL.
_OWN_TRANSACTION = re.compile(r"^\s*(BEGIN|COMMIT)\s*;\s*$", re.IGNORECASE | re.MULTILINE)


def migration_files(directory: Path | None = None) -> list[Path]:
    """Every migration, in lexicographic (== chronological) order."""
    return sorted((directory or MIGRATIONS_DIR).glob("*.sql"))


async def apply_migrations(dsn: str, *, directory: Path | None = None) -> list[str]:
    """Bring the database at ``dsn`` up to date. Returns the versions applied.

    Takes a DSN rather than a pool because it runs before the pool exists: the
    ``vector`` codec cannot be registered until ``CREATE EXTENSION vector`` in
    0001 has run.
    """
    connection: Any = await asyncpg.connect(dsn)
    try:
        await connection.execute(_LEDGER)
        done: set[str] = {
            row["version"] for row in await connection.fetch(
                "SELECT version FROM schema_migrations"
            )
        }
        applied: list[str] = []
        for path in migration_files(directory):
            version = path.stem
            if version in done:
                continue
            sql = _OWN_TRANSACTION.sub("", path.read_text(encoding="utf-8"))
            async with connection.transaction():
                await connection.execute(sql)
                await connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)", version
                )
            applied.append(version)
        return applied
    finally:
        await connection.close()
