"""Ports: the boundary between the domain and the outside world.

The domain declares what it needs; ``infra`` supplies it. This is the only
piece of Clean Architecture ceremony that earns its keep here, and it exists
for one concrete reason: the M6 evaluation harness runs the entire review
pipeline against a fake ``GitHubPort`` and a recorded ``LLMPort``, so that eval
runs are deterministic and free. Without the boundary, the harness would have
to monkeypatch, and a harness built on monkeypatching stops being trusted about
three weeks in.

Structural ``Protocol`` rather than ABC inheritance: adapters do not import the
domain to satisfy it, which keeps the dependency arrow pointing one way even
for third-party wrappers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.domain.contracts import Finding

__all__ = [
    "AnalyzerPort",
    "AnalyzerRequest",
    "ChunkMatch",
    "EmbeddingPort",
    "GitHubPort",
    "LLMPort",
    "LLMResponse",
    "PullRequestDiff",
    "VectorStorePort",
]


@dataclass(frozen=True, slots=True)
class PullRequestDiff:
    """A PR as the pipeline needs it: unified diff plus the metadata required
    to anchor comments. Deliberately not the raw GitHub payload -- the domain
    should not know what GitHub's JSON looks like."""

    base_sha: str
    head_sha: str
    unified_diff: str
    changed_paths: Sequence[str]
    commit_messages: Sequence[str]


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    chunk_id: str
    path: str
    line_start: int
    line_end: int
    content: str
    score: float
    symbol_fqn: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Token counts and cost ride along with the payload because per-run cost is
    an SLO (<= $0.15 median), and a budget you only measure at the provider
    dashboard is a budget you do not enforce."""

    content: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str


@dataclass(frozen=True, slots=True)
class AnalyzerRequest:
    workspace_path: str
    changed_paths: Sequence[str]
    language: str
    timeout_seconds: int = 60


@runtime_checkable
class GitHubPort(Protocol):
    """Everything the pipeline needs from GitHub. Small on purpose: each method
    here is one the fake adapter must implement."""

    async def fetch_diff(self, repo: str, pr_number: int) -> PullRequestDiff: ...

    async def fetch_file(self, repo: str, path: str, sha: str) -> str | None:
        """Returns None when the path does not exist at ``sha``.

        This is the primitive behind the FILE_EXISTS verification gate, which is
        why it returns None rather than raising: a missing file is an expected
        outcome during verification, not an error.
        """
        ...

    async def post_review(
        self, repo: str, pr_number: int, findings: Sequence[Finding], summary: str
    ) -> None: ...


@runtime_checkable
class EmbeddingPort(Protocol):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...


@runtime_checkable
class VectorStorePort(Protocol):
    async def search(
        self,
        repository_id: str,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        exclude_paths: Sequence[str] = (),
    ) -> Sequence[ChunkMatch]:
        """ANN search. ``repository_id`` is mandatory and first: an unfiltered
        vector search across tenants is a data-leak bug, so the signature makes
        it impossible to forget."""
        ...


@runtime_checkable
class LLMPort(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


@runtime_checkable
class AnalyzerPort(Protocol):
    """A sandboxed static analyzer. Implementations must enforce the sandbox
    themselves (non-root, read-only FS, no network, memory and wall-clock caps)
    -- cloned repositories are untrusted input and analyzers execute against
    them."""

    @property
    def name(self) -> str: ...

    def supports(self, language: str) -> bool: ...

    async def run(self, request: AnalyzerRequest) -> Sequence[Finding]: ...
