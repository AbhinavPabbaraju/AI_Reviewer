"""Process configuration, read from the environment.

Kept deliberately small. Everything here is something a deployment must be able
to change without a code edit, and nothing here is a tuning knob that belongs in
the domain -- the retrieval budget, the expansion depth and the fusion weights
live in ``RetrievalConfig`` and ``ExpansionConfig``, where they are typed,
defaulted, and covered by the gates.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Environment-driven settings, prefixed ``ARGUS_``."""

    model_config = SettingsConfigDict(
        env_prefix="ARGUS_", env_file=".env", extra="ignore"
    )

    database_url: str = Field(
        default="postgresql://argus@localhost:5432/argus",
        description="Postgres DSN. pgvector must be installed in this database.",
    )
    database_pool_min: int = Field(default=1, ge=0)
    database_pool_max: int = Field(default=10, ge=1)

    embedding_dimensions: int = Field(
        default=1536,
        ge=1,
        description="Must match `chunks.embedding vector(N)` in the DDL; the "
        "column is dimensioned because HNSW cannot index a column of unknown "
        "width, so this is not independently tunable.",
    )

    search_max_limit: int = Field(
        default=50,
        ge=1,
        description="Ceiling on `POST /search` results per request. An ANN scan "
        "is cheap but the response carries chunk bodies, so the cap is on the "
        "payload rather than on the index.",
    )


def get_settings() -> Settings:
    """Settings for this process.

    Not cached: FastAPI resolves it once at startup through the dependency
    system, and a module-level cache would make it awkward to override in tests
    for no benefit anywhere else.
    """
    return Settings()
