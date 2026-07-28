"""A :class:`SourceProviderPort` backed by git subprocesses.

Cloned code is untrusted input (ARCHITECTURE sec. 8), so every safety lever git
offers is pulled:

* ``--bare`` + ``--filter=blob:none`` -- a blobless bare clone. Trees are fetched;
  blobs are pulled lazily only for files that survive filtering, which is the
  mechanism behind the index-freshness budget.
* ``core.hooksPath=/dev/null`` -- a cloned repo's hooks never run.
* ``--no-tags`` and no submodule recursion -- less surface, no arbitrary fetch.
* ``protocol.ext.allow=never`` -- disables the ``ext::`` transport, a known RCE
  vector when cloning untrusted URLs.

Nothing here executes code *from* the tree; git only ever reads it.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path

from app.domain.indexing.ports import FileEntry

__all__ = ["GitCommandError", "GitSourceProvider"]


class GitCommandError(RuntimeError):
    """A git subprocess exited non-zero."""


_SAFE_GIT_CONFIG = (
    "-c", "core.hooksPath=/dev/null",
    "-c", "protocol.ext.allow=never",
    "-c", "advice.detachedHead=false",
)


class GitSourceProvider:
    """Clones each repository once (bare, blobless) and serves trees and blobs
    from that cache. Safe to share across concurrent index jobs: clone creation
    is serialized, reads are not."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._root = workspace_root or Path(
            tempfile.mkdtemp(prefix="argus-src-")
        )
        self._root.mkdir(parents=True, exist_ok=True)
        self._clones: dict[str, Path] = {}
        self._clone_lock = asyncio.Lock()

    async def list_files(
        self, repo_url: str, commit_sha: str
    ) -> list[FileEntry]:
        clone = await self._ensure_clone(repo_url)
        out = await self._git(clone, "ls-tree", "-r", "-l", "-z", commit_sha)
        entries: list[FileEntry] = []
        for record in out.split(b"\x00"):
            if not record:
                continue
            meta, _, raw_path = record.partition(b"\t")
            fields = meta.split()
            if len(fields) < 4 or fields[1] != b"blob":
                continue  # skip trees, symlinks (mode 120000 is still a blob),
            size = int(fields[3]) if fields[3].isdigit() else 0
            entries.append(
                FileEntry(
                    path=raw_path.decode("utf-8", "surrogateescape"),
                    blob_sha=fields[2].decode("ascii"),
                    size_bytes=size,
                )
            )
        return entries

    async def read_blob(self, repo_url: str, blob_sha: str) -> bytes:
        clone = await self._ensure_clone(repo_url)
        return await self._git(clone, "cat-file", "blob", blob_sha)

    # -- internals ------------------------------------------------------- #

    async def _ensure_clone(self, repo_url: str) -> Path:
        async with self._clone_lock:
            cached = self._clones.get(repo_url)
            if cached is not None:
                return cached
            dest = self._root / _slug(repo_url)
            if not dest.exists():
                await self._git(
                    None,
                    "clone",
                    "--bare",
                    "--filter=blob:none",
                    "--no-tags",
                    repo_url,
                    str(dest),
                )
            self._clones[repo_url] = dest
            return dest

    async def _git(self, cwd: Path | None, *args: str) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *_SAFE_GIT_CONFIG,
            *args,
            cwd=str(cwd) if cwd is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise GitCommandError(
                f"git {' '.join(args[:2])} failed ({proc.returncode}): "
                f"{stderr.decode('utf-8', 'replace').strip()}"
            )
        return stdout


def _slug(repo_url: str) -> str:
    """A filesystem-safe, collision-resistant directory name for a repo url."""
    digest = hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:16]
    tail = repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    safe_tail = "".join(c if c.isalnum() or c in "-_" else "_" for c in tail)[:40]
    return f"{safe_tail}-{digest}.git"
