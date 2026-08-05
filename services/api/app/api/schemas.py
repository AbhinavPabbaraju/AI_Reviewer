"""Request and response models for the HTTP API.

Separate from ``domain/contracts.py`` on purpose. The contracts are the *output*
of the pipeline and are shared with the worker and the eval harness; these are
the shapes of one transport, and they carry transport concerns (limits,
pagination, the snapshot a result was served from) that the domain has no
opinion about. Keeping them apart is what stops HTTP details leaking into the
object the verification gate operates on.

They are still ``Frozen``/``extra="forbid"`` like everything else, so an unknown
key in a request body is a 422 rather than a silently ignored typo -- and, since
the frontend's TypeScript types are generated from the OpenAPI schema these
produce, a field renamed here cannot drift out of sync with the client.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.domain.base import Frozen

__all__ = ["SearchHit", "SearchRequest", "SearchResponse"]


class SearchRequest(Frozen):
    """A semantic code search over one repository's current snapshot."""

    repository_id: UUID
    query: str = Field(
        min_length=1,
        max_length=4_000,
        description="Natural language or code. Embedded with the same model "
        "that embedded the corpus -- a query embedded by a different model is "
        "not comparable to the stored vectors, however similar the text looks.",
    )
    limit: int = Field(default=20, ge=1)
    exclude_paths: tuple[str, ...] = Field(
        default=(),
        description="Paths to omit. The retriever uses this to keep the "
        "semantic supplement from returning the file the diff already anchors "
        "on; a caller debugging retrieval can use it the same way.",
    )


class SearchHit(Frozen):
    """One retrieved chunk."""

    chunk_id: str
    path: str
    line_start: int
    line_end: int
    symbol_fqn: str | None
    score: float = Field(
        description="Cosine similarity in [-1, 1]. An *uncalibrated* ranking "
        "signal, like `Finding.confidence` before M6 fits the calibration map: "
        "comparable between hits of one query, not between queries."
    )
    content: str


class SearchResponse(Frozen):
    """Results, plus which snapshot answered.

    ``snapshot_id`` and ``commit_sha`` are not decoration: retrieval serves the
    newest *ready* snapshot, so a search run seconds after a push may legitimately
    answer from the previous commit. A debugging tool that hid which commit it
    searched would send people hunting for bugs in the retriever that are really
    just indexing lag.
    """

    repository_id: UUID
    snapshot_id: UUID
    commit_sha: str
    hits: tuple[SearchHit, ...]
    took_ms: int
