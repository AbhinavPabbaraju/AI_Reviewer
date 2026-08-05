"""The FastAPI application.

Thin by construction (ARCHITECTURE sec. 3): this process verifies input, reads
Postgres, and will enqueue work. It never calls an LLM inline and never indexes
inline -- both are the worker's job, and an API that did either could not hold
its < 200 ms webhook-ack budget.

The connection pool is opened in the lifespan rather than at import time so that
importing this module (which the tests and the OpenAPI type generator both do)
does not require a database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import build_default_embedder
from app.api.routers import search
from app.config import Settings, get_settings
from app.infra.db.pool import close_pool, create_pool

__all__ = ["create_app"]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Takes settings so tests can point it at a scratch
    database without reaching through the environment."""
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.embedder = build_default_embedder(resolved)
        app.state.pool = await create_pool(
            resolved.database_url,
            min_size=resolved.database_pool_min,
            max_size=resolved.database_pool_max,
        )
        try:
            yield
        finally:
            await close_pool(app.state.pool)

    app = FastAPI(
        title="Argus",
        version="0.1.0",
        summary="An AI pull-request reviewer optimized for precision, not coverage.",
        lifespan=lifespan,
    )
    app.include_router(search.router)

    @app.get("/health", tags=["ops"], summary="Liveness and database reachability")
    async def health() -> dict[str, str]:
        # Checks the pool, not just the process: an API that answers "ok" while
        # unable to reach Postgres tells a load balancer to keep sending it
        # traffic it cannot serve.
        await app.state.pool.fetchval("SELECT 1")
        return {"status": "ok"}

    return app


app = create_app
"""ASGI factory. Run with ``uvicorn app.api.main:app --factory``."""
