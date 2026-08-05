"""``POST /search`` end to end, against a real database.

Driven through the ASGI app with a real HTTP client rather than by calling the
handler function: the things most likely to be wrong in an endpoint are request
validation, status codes and serialization, and none of those run when you call
the handler directly.

The two error paths get as much attention as the happy one, because both are
cases where the tempting behaviour is to return something plausible. An
unindexed repository must not look like "no matches", and a snapshot embedded by
a different model must not be answered with cosine similarities computed across
two incompatible vector spaces.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.config import Settings
from app.infra.store.postgres import PostgresIndexStore
from tests.eval.corpus.indexed import (
    EMBEDDING_DIMENSIONS,
    IndexedCorpus,
    build_corpus,
)
from tests.eval.corpus.python_corpus import PY_CORPUS, PY_TEST_FILES
from tests.pg import create_repository


@pytest.fixture(scope="module")
async def corpus() -> IndexedCorpus:
    return await build_corpus(PY_CORPUS, PY_TEST_FILES, typescript=False)


@pytest.fixture
async def client(pg_dsn: str, pg_pool: Any) -> AsyncIterator[AsyncClient]:
    """The real app over ASGI, sharing the test database.

    ``LifespanManager`` runs startup and shutdown so the app opens its own pool
    exactly as it would in production -- the ``pg_pool`` fixture is here for the
    test's own writes and for its truncation.
    """
    app = create_app(
        Settings(
            database_url=pg_dsn,
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            search_max_limit=25,
        )
    )
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://argus.test"
        ) as http:
            yield http


@pytest.fixture
async def indexed(pg_pool: Any, corpus: IndexedCorpus) -> UUID:
    await create_repository(pg_pool, corpus.repository_id)
    await PostgresIndexStore(pg_pool).save(corpus.snapshot)
    return corpus.repository_id


class TestSearch:
    async def test_returns_ranked_hits_from_the_current_snapshot(
        self, client: AsyncClient, indexed: UUID, corpus: IndexedCorpus
    ) -> None:
        response = await client.post(
            "/search",
            json={
                "repository_id": str(indexed),
                "query": "validate an entity before saving it",
                "limit": 5,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["repository_id"] == str(indexed)
        assert body["snapshot_id"] == str(corpus.snapshot.id)
        assert body["commit_sha"] == corpus.snapshot.commit_sha
        assert 0 < len(body["hits"]) <= 5

        scores = [hit["score"] for hit in body["hits"]]
        assert scores == sorted(scores, reverse=True)
        for hit in body["hits"]:
            assert hit["content"]
            assert hit["line_end"] >= hit["line_start"] >= 1
            assert not hit["path"].startswith("/")

    async def test_excluded_paths_are_omitted(
        self, client: AsyncClient, indexed: UUID
    ) -> None:
        response = await client.post(
            "/search",
            json={
                "repository_id": str(indexed),
                "query": "entity",
                "limit": 20,
                "exclude_paths": ["shop/models.py"],
            },
        )
        assert response.status_code == 200, response.text
        paths = {hit["path"] for hit in response.json()["hits"]}
        assert paths
        assert "shop/models.py" not in paths

    async def test_limit_is_capped_by_configuration(
        self, client: AsyncClient, indexed: UUID
    ) -> None:
        """The cap is on payload size -- the response carries chunk bodies --
        so an over-large request is clamped rather than rejected."""
        response = await client.post(
            "/search",
            json={"repository_id": str(indexed), "query": "order", "limit": 500},
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["hits"]) <= 25

    async def test_unindexed_repository_is_a_404_not_an_empty_result(
        self, client: AsyncClient, pg_pool: Any
    ) -> None:
        never_indexed = await create_repository(pg_pool)
        response = await client.post(
            "/search",
            json={"repository_id": str(never_indexed), "query": "anything"},
        )
        assert response.status_code == 404
        assert "no ready index snapshot" in response.json()["detail"]

    async def test_unknown_repository_is_also_a_404(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/search", json={"repository_id": str(uuid4()), "query": "anything"}
        )
        assert response.status_code == 404

    async def test_model_mismatch_is_refused_rather_than_answered(
        self, client: AsyncClient, pg_pool: Any, corpus: IndexedCorpus
    ) -> None:
        """The failure this guard exists for is silent.

        Cosine similarity between vectors from two different models returns a
        number, not an error, so a model swap without a re-index degrades search
        into plausible nonsense. The snapshot records what embedded it; the
        endpoint compares and refuses.
        """
        await create_repository(pg_pool, corpus.repository_id)
        await PostgresIndexStore(pg_pool).save(
            corpus.snapshot.model_copy(
                update={"embedding_model": "text-embedding-3-small"}
            )
        )
        response = await client.post(
            "/search",
            json={"repository_id": str(corpus.repository_id), "query": "entity"},
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "text-embedding-3-small" in detail
        assert "re-index" in detail


class TestRequestValidation:
    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"query": "x"}, id="missing-repository"),
            pytest.param(
                {"repository_id": "not-a-uuid", "query": "x"}, id="bad-uuid"
            ),
            pytest.param(
                {"repository_id": str(uuid4()), "query": ""}, id="empty-query"
            ),
            pytest.param(
                {"repository_id": str(uuid4()), "query": "x", "limit": 0},
                id="zero-limit",
            ),
            pytest.param(
                {"repository_id": str(uuid4()), "query": "x", "typo": 1},
                id="unknown-field",
            ),
        ],
    )
    async def test_bad_requests_are_rejected(
        self, client: AsyncClient, body: dict[str, object]
    ) -> None:
        """``extra="forbid"`` makes the unknown-field case a 422 too: the
        frontend's types are generated from this schema, so a field the server
        does not know is drift worth failing on."""
        response = await client.post("/search", json=body)
        assert response.status_code == 422


class TestHealth:
    async def test_health_checks_the_database_not_just_the_process(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestOpenAPI:
    async def test_schema_documents_the_endpoint(
        self, client: AsyncClient
    ) -> None:
        """The frontend's TypeScript types are generated from this document, so
        it is part of the contract rather than a by-product."""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "/search" in schema["paths"]
        operation = schema["paths"]["/search"]["post"]
        assert {"200", "404", "409", "422"} <= set(operation["responses"])
