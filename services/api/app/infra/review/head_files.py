"""``HeadFilePort`` bindings: the gate's window onto the tree at one SHA.

The port is bound to a single (repo, sha) precisely so that no caller can verify
a finding against the wrong commit -- a mistake that would produce comments
anchored to lines that moved, which is indistinguishable from a hallucination in
the output and much harder to diagnose.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.domain.ports import GitHubPort

__all__ = ["GitHubHeadFiles", "MappingHeadFiles"]


class GitHubHeadFiles:
    """Reads head-SHA files through a :class:`GitHubPort`."""

    def __init__(self, github: GitHubPort, *, repo: str, head_sha: str) -> None:
        self._github = github
        self._repo = repo
        self._head_sha = head_sha

    async def read(self, path: str) -> str | None:
        return await self._github.fetch_file(self._repo, path, self._head_sha)


class MappingHeadFiles:
    """Reads from an in-memory ``path -> content`` map.

    For tests that are about the gate's logic rather than about transport. A
    path absent from the map is absent at head, which is the whole point.
    """

    def __init__(self, files: Mapping[str, str]) -> None:
        self._files = dict(files)

    async def read(self, path: str) -> str | None:
        return self._files.get(path)
