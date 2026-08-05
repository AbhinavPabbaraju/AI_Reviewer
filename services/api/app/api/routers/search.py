"""``POST /search`` -- semantic code search, and the retrieval debugging tool.

ROADMAP M2 asks for this endpoint as "semantic code search (also the manual
debugging tool for retrieval)". It is worth being precise about which half of
retrieval it exposes, because the architecture's central claim is that the other
half matters more.

Retrieval is structure-first (ADR-002): a context pack is anchored on the
symbols a diff *changed*, expanded along the symbol graph, and only then
supplemented by vector search. A free-text query has no anchor symbol, so no
graph expansion is possible from it -- there is nothing to be two hops away
*from*. What a text query can drive is the vector half, and that is exactly the
half you cannot otherwise inspect: the graph is visible in the resolver's own
gates, while "are these embeddings any good on this repository?" is invisible
until something asks the question.

So this endpoint is honest about being the ANN inspector rather than a preview
of what a review would retrieve. Building a full context pack is a different
request shape (it needs hunks, not prose) and belongs with M7's retrieval
inspector, which renders the ``reason`` each pack item already carries.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, status

from app.api.deps import EmbedderDep, PoolDep, SettingsDep
from app.api.schemas import SearchHit, SearchRequest, SearchResponse
from app.infra.retrieval.postgres import PostgresVectorStore, latest_ready_snapshot

__all__ = ["router"]

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic code search over a repository's current snapshot",
    responses={
        404: {"description": "The repository has no ready index snapshot."},
        409: {"description": "The corpus was embedded by a different model."},
    },
)
async def search(
    request: SearchRequest,
    pool: PoolDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
) -> SearchResponse:
    started = time.perf_counter()

    serving = await latest_ready_snapshot(pool, request.repository_id)
    if serving is None:
        # A repository that has never finished an index is a real state -- a
        # fresh installation, or a first index still running -- so it is a 404
        # about the *snapshot*, not a 500 and not an empty result set that would
        # read as "no matches".
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"repository {request.repository_id} has no ready index "
                "snapshot yet"
            ),
        )

    if serving.embedding_model != embedder.model:
        # Vectors from two models are not comparable, and cosine similarity
        # between them is a number rather than an error -- which is the
        # dangerous part. Refusing is the only way this surfaces at all.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"snapshot was embedded with {serving.embedding_model!r} but "
                f"queries are embedded with {embedder.model!r}; re-index the "
                "repository before searching it with this model"
            ),
        )

    limit = min(request.limit, settings.search_max_limit)
    [vector] = await embedder.embed([request.query])
    vectors = PostgresVectorStore(pool, request.repository_id, serving.id)
    matches = await vectors.search(
        str(request.repository_id),
        vector,
        limit=limit,
        exclude_paths=request.exclude_paths,
    )

    return SearchResponse(
        repository_id=request.repository_id,
        snapshot_id=serving.id,
        commit_sha=serving.commit_sha,
        hits=tuple(
            SearchHit(
                chunk_id=match.chunk_id,
                path=match.path,
                line_start=match.line_start,
                line_end=match.line_end,
                symbol_fqn=match.symbol_fqn,
                score=match.score,
                content=match.content,
            )
            for match in matches
        ),
        took_ms=round((time.perf_counter() - started) * 1000),
    )
