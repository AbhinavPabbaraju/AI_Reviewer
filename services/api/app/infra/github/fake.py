"""An in-memory :class:`GitHubPort`. Not a mock -- a working implementation.

ARCHITECTURE sec. 3 names this as the reason the port boundary exists at all:
the M6 evaluation harness runs the entire review pipeline against a fake GitHub
and a recorded LLM so eval runs are deterministic and free. It arrives here in
M3 because the verification gate needs the same thing a milestone earlier --
``FILE_EXISTS`` and ``LINE_IN_RANGE`` are questions about a real tree, and
testing them against a mock that returns whatever the test says would be testing
the mock.

It behaves like GitHub in the two ways that matter to the gate: a file absent at
a SHA returns ``None`` rather than raising, and content is stored per SHA, so a
path that exists at base and not at head is representable. That case is exactly
what a model gets wrong when it reviews a PR that deleted a file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.contracts import Finding
from app.domain.ports import PullRequestDiff

__all__ = ["FakeGitHub", "PostedReview"]


@dataclass(frozen=True, slots=True)
class PostedReview:
    """What ``post_review`` was called with. Tests assert on these rather than
    on internal state, so they check the boundary the user actually sees."""

    repo: str
    pr_number: int
    findings: tuple[Finding, ...]
    summary: str


class FakeGitHub:
    """Implements :class:`app.domain.ports.GitHubPort` from in-memory trees."""

    def __init__(
        self,
        *,
        trees: Mapping[str, Mapping[str, str]] | None = None,
        diffs: Mapping[tuple[str, int], PullRequestDiff] | None = None,
    ) -> None:
        # sha -> path -> content
        self._trees: dict[str, dict[str, str]] = {
            sha: dict(files) for sha, files in (trees or {}).items()
        }
        self._diffs: dict[tuple[str, int], PullRequestDiff] = dict(diffs or {})
        self.posted: list[PostedReview] = []

    # -- authoring ---------------------------------------------------------- #

    def add_tree(self, sha: str, files: Mapping[str, str]) -> None:
        self._trees.setdefault(sha, {}).update(files)

    def add_diff(self, repo: str, pr_number: int, diff: PullRequestDiff) -> None:
        self._diffs[(repo, pr_number)] = diff

    # -- GitHubPort --------------------------------------------------------- #

    async def fetch_diff(self, repo: str, pr_number: int) -> PullRequestDiff:
        diff = self._diffs.get((repo, pr_number))
        if diff is None:
            raise KeyError(f"no diff registered for {repo}#{pr_number}")
        return diff

    async def fetch_file(self, repo: str, path: str, sha: str) -> str | None:
        return self._trees.get(sha, {}).get(path)

    async def post_review(
        self, repo: str, pr_number: int, findings: Sequence[Finding], summary: str
    ) -> None:
        self.posted.append(
            PostedReview(
                repo=repo,
                pr_number=pr_number,
                findings=tuple(findings),
                summary=summary,
            )
        )
