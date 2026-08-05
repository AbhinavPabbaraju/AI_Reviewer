"""Dependency providers for the API.

The API is *thin* (ARCHITECTURE sec. 3): it verifies, reads, enqueues, and
returns. Everything it does with the database goes through the same adapters the
worker uses, so there is exactly one implementation of "search this repository"
in the system and the endpoint is a transport wrapper around it.

The connection pool lives on ``app.state`` and is opened once in the lifespan
rather than per request -- opening a pool per request would be a new TCP
connection, a new TLS handshake and a fresh ``vector`` codec registration on
every call.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from app.config import Settings
from app.domain.ports import EmbeddingPort
from app.infra.embedding.deterministic import DeterministicEmbedder

__all__ = ["EmbedderDep", "PoolDep", "SettingsDep"]


def get_pool(request: Request) -> Any:
    return request.app.state.pool


def get_settings_from_state(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_embedder(request: Request) -> EmbeddingPort:
    """The embedding provider for query vectors.

    Currently the deterministic offline embedder, because it is the only
    implementation of ``EmbeddingPort`` that exists -- a network provider is
    still outstanding M2 work. That is not a placeholder that silently degrades
    search: the endpoint refuses to answer when the model that embedded the
    corpus is not the model embedding the query, so swapping this for a network
    adapter changes which snapshots are searchable rather than quietly returning
    nonsense against vectors from a different space.
    """
    embedder: EmbeddingPort = request.app.state.embedder
    return embedder


def build_default_embedder(settings: Settings) -> EmbeddingPort:
    return DeterministicEmbedder(dimensions=settings.embedding_dimensions)


PoolDep = Annotated[Any, Depends(get_pool)]
SettingsDep = Annotated[Settings, Depends(get_settings_from_state)]
EmbedderDep = Annotated[EmbeddingPort, Depends(get_embedder)]
