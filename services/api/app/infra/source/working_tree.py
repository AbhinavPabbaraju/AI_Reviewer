"""A :class:`SourceProviderPort` over a checkout on disk, uncommitted edits included.

``GitSourceProvider`` reads a *commit*, which is the right thing for a webhook:
a pull request has a head sha and that sha is what gets reviewed. It is the wrong
thing for the change you have not committed yet, which is the one you most want
a reviewer to look at before you push it.

This reads the working tree instead. The distinction matters more than it looks:
the verification gate checks that a finding's line exists in the file, so if the
index is built from ``HEAD`` while the diff describes uncommitted edits, every
line number is off by whatever you just wrote and correct findings get rejected
for citing lines that "do not exist". Indexing what is actually on disk keeps the
diff, the retrieval context and the gate all describing the same tree.

**Tracked files only.** The file list comes from ``git ls-files``, so anything
untracked or ignored is invisible without re-implementing ``.gitignore``
semantics that git already applies. A file you have not ``git add``-ed is not yet
part of the repository, and reviewing it would mean reviewing scratch work.

**Blob shas stay honest.** They are the parse cache's key, so a modified file
must not present the sha of its committed version -- that is a cache hit against
the wrong content, and it would serve stale symbols for exactly the file being
reviewed. Clean files keep the sha git already has; dirty ones are hashed from
disk with git's own blob hash, so the two are the same identifier space.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from pathlib import Path

from app.domain.indexing.ports import FileEntry

__all__ = ["WorkingTreeSource"]

_NUL = b"\x00"


class WorkingTreeSource:
    """Serves a git checkout's tracked files as they exist on disk."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        # sha -> path, for the files whose content is not in the object database
        # (modified but not committed). Populated by `list_files`, which the
        # indexer always calls before any `read_blob`.
        self._on_disk: dict[str, str] = {}

    @property
    def root(self) -> Path:
        return self._root

    async def list_files(
        self, repo_url: str, commit_sha: str
    ) -> Sequence[FileEntry]:
        """Every tracked file, sized and hashed as it is right now.

        ``repo_url`` and ``commit_sha`` are part of the port and ignored here:
        this provider is bound to one checkout at construction, which is what
        makes it impossible to accidentally read a different tree than the one
        the diff and the gate are looking at.
        """
        staged = await self._staged_entries()
        dirty = await self._dirty_paths()
        self._on_disk = {}

        entries: list[FileEntry] = []
        for path, blob_sha in staged.items():
            absolute = self._root / path
            try:
                stat = absolute.stat()
            except OSError:
                # Tracked but absent: a deletion staged or in flight. There is
                # nothing to index, and the diff already describes the removal.
                continue
            if absolute.is_symlink():
                continue
            if path in dirty:
                content = absolute.read_bytes()
                blob_sha = _blob_sha(content)
                self._on_disk[blob_sha] = path
            entries.append(
                FileEntry(path=path, blob_sha=blob_sha, size_bytes=stat.st_size)
            )
        return entries

    async def read_blob(self, repo_url: str, blob_sha: str) -> bytes:
        """The bytes behind a sha, from disk when git does not have them yet."""
        path = self._on_disk.get(blob_sha)
        if path is not None:
            return (self._root / path).read_bytes()
        return await self._git("cat-file", "blob", blob_sha)

    # -- git ---------------------------------------------------------------- #

    async def _staged_entries(self) -> dict[str, str]:
        """``path -> blob sha`` for every tracked file, from the index."""
        raw = await self._git("ls-files", "-s", "-z")
        entries: dict[str, str] = {}
        for record in raw.split(_NUL):
            if not record:
                continue
            # "<mode> <sha> <stage>\t<path>"
            meta, _, path = record.partition(b"\t")
            fields = meta.split()
            if len(fields) < 3 or not path:
                continue
            mode = fields[0].decode()
            if mode.startswith("120") or mode.startswith("160"):
                # Symlink or submodule gitlink: not source this repository owns.
                continue
            entries[path.decode("utf-8", "surrogateescape")] = fields[1].decode()
        return entries

    async def _dirty_paths(self) -> frozenset[str]:
        """Tracked files whose content on disk differs from the index."""
        raw = await self._git("diff", "--name-only", "-z", "HEAD")
        return frozenset(
            part.decode("utf-8", "surrogateescape")
            for part in raw.split(_NUL)
            if part
        )

    async def _git(self, *args: str) -> bytes:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(self._root),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed in {self._root}: "
                f"{stderr.decode('utf-8', 'replace').strip()}"
            )
        return stdout


def _blob_sha(content: bytes) -> str:
    """Git's blob hash, computed without shelling out once per file.

    Git hashes ``blob <length>\\0<content>`` with SHA-1. Recomputing it here
    rather than calling ``git hash-object`` keeps a dirty tree of a few hundred
    files from becoming a few hundred subprocesses.
    """
    digest = hashlib.sha1(b"blob %d\x00" % len(content))
    digest.update(content)
    return digest.hexdigest()
