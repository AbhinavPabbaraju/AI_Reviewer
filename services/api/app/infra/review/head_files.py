"""``HeadFilePort`` bindings: the gate's window onto the tree at one SHA.

The port is bound to a single (repo, sha) precisely so that no caller can verify
a finding against the wrong commit -- a mistake that would produce comments
anchored to lines that moved, which is indistinguishable from a hallucination in
the output and much harder to diagnose.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from app.domain.ports import GitHubPort

__all__ = ["GitHubHeadFiles", "MappingHeadFiles", "WorkingTreeHeadFiles"]


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


class WorkingTreeHeadFiles:
    """Reads a checkout on disk -- head is whatever is there right now.

    Pairs with ``WorkingTreeSource``: when the review is of uncommitted work,
    "the tree at head" is the filesystem, and verifying against ``HEAD`` instead
    would reject correct findings for citing lines the commit does not have yet.

    Paths are confined to the root even though ``CodeSpan`` has already rejected
    absolute paths and ``..`` segments at the type boundary. This is the point
    where a path derived from model output becomes a filesystem read, and one
    check at the boundary is cheaper than being sure about every path upstream
    of it forever.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    async def read(self, path: str) -> str | None:
        candidate = (self._root / path).resolve()
        if not candidate.is_relative_to(self._root):
            return None
        try:
            return candidate.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Absent, a directory, or unreadable. All mean the same thing to the
            # FILE_EXISTS gate: there is no file here to anchor a comment on.
            return None
