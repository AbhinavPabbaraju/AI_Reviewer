"""Ports the verification gate needs.

Both are deliberately narrower than the M0 port they will usually be backed by.
``GitHubPort.fetch_file`` takes a repo and a SHA; the gate is verifying findings
for *one* run against *one* head SHA, and threading those through every call
would let a caller accidentally verify a finding against the wrong commit. A
port bound to the run makes that unrepresentable.

``SyntaxCheckerPort`` exists because ``PATCH_PARSES`` needs a real grammar and
grammars live in infra. The domain must not import tree-sitter (ARCHITECTURE
sec. 3), and the M6 harness needs to run the gate without native grammars
loaded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.indexing.models import Language

__all__ = ["HeadFilePort", "SyntaxCheckerPort"]


@runtime_checkable
class HeadFilePort(Protocol):
    """Reads files as they exist at the run's head SHA."""

    async def read(self, path: str) -> str | None:
        """File contents, or ``None`` when the path does not exist at head.

        ``None`` rather than an exception because a missing path is the expected
        outcome the ``FILE_EXISTS`` gate is built to detect -- a model citing a
        file that is not there is the single most common fabrication, not an
        error condition.
        """
        ...


@runtime_checkable
class SyntaxCheckerPort(Protocol):
    """Decides whether a fragment parses under a language's grammar."""

    def parses(self, language: Language, source: str) -> bool:
        """True when ``source`` contains no syntax errors.

        Implementations must accept *fragments*, not just whole files: a
        suggested fix is usually a few lines lifted out of an indented block,
        and a checker that demanded a complete module would strip every patch it
        was shown.
        """
        ...
