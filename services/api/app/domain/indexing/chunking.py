"""Symbol-boundary chunking with content-addressed ids (ARCHITECTURE.md sec. 4.1).

> Chunking is by symbol, never by token window. A chunk that splits a function
> in half is worse than useless.

So a chunk is exactly one definition. A container (class/interface/enum) is
chunked as its *header* -- signature, docstring, class-level attributes -- with
its methods' line ranges removed, because each method is already its own chunk.
That avoids embedding a method body twice (once standalone, once inside its
class) while still giving retrieval a class-level summary chunk.

The id is ``sha256(repo_id | path | symbol_fqn | normalized_body)`` (sec. 4.1),
which is why re-indexing an unchanged file re-embeds nothing: identical inputs
hash identically, and the store skips a chunk whose hash it already holds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from uuid import UUID

from app.domain.contracts import CodeSpan
from app.domain.indexing.models import Chunk, ParsedFile, Symbol, SymbolKind

__all__ = ["CHUNKABLE_KINDS", "chunk_parsed_file", "content_hash"]

# Definitions that become their own chunk. Bare module-level variables are
# excluded: they are usually one-liners whose meaning lives in the symbol that
# uses them, and chunking each would flood the store with low-value embeddings.
CHUNKABLE_KINDS: frozenset[SymbolKind] = frozenset(
    {
        SymbolKind.FUNCTION,
        SymbolKind.METHOD,
        SymbolKind.CLASS,
        SymbolKind.INTERFACE,
        SymbolKind.TYPE_ALIAS,
        SymbolKind.ENUM,
    }
)

_CONTAINER_KINDS: frozenset[SymbolKind] = frozenset(
    {SymbolKind.CLASS, SymbolKind.INTERFACE, SymbolKind.ENUM}
)

# Rough chars-per-token estimate. The real tokenizer lives in the LLM/embedding
# adapter (M2); this only needs to be monotonic to enforce the pack budget.
_CHARS_PER_TOKEN = 4


def content_hash(
    repository_id: UUID, path: str, symbol_fqn: str | None, normalized_body: str
) -> str:
    """The content-addressed chunk id from ARCHITECTURE sec. 4.1."""
    material = "\x00".join(
        [str(repository_id), path, symbol_fqn or "", normalized_body]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def chunk_parsed_file(
    repository_id: UUID, parsed: ParsedFile, source: str
) -> list[Chunk]:
    """Produce one chunk per chunkable symbol in ``parsed``.

    ``source`` is the file's decoded text at the same blob the parser saw, so
    line numbers line up. Symbols whose normalized body is empty (a stub whose
    every line was claimed by a child) are skipped rather than stored empty.
    """
    lines = source.splitlines()
    chunkable = [s for s in parsed.symbols if s.kind in CHUNKABLE_KINDS]
    children_by_parent: dict[str, list[Symbol]] = {}
    for symbol in chunkable:
        if symbol.parent_fqn is not None:
            children_by_parent.setdefault(symbol.parent_fqn, []).append(symbol)

    chunks: list[Chunk] = []
    for symbol in chunkable:
        excluded = _excluded_lines(symbol, children_by_parent.get(symbol.fqn, ()))
        body = _normalized_body(lines, symbol, excluded)
        if not body:
            continue
        chunks.append(
            Chunk(
                content_hash=content_hash(
                    repository_id, symbol.path, symbol.fqn, body
                ),
                symbol_fqn=symbol.fqn,
                span=symbol.span,
                language=symbol.language,
                token_count=max(1, len(body) // _CHARS_PER_TOKEN),
                content=body,
            )
        )
    return chunks


def _excluded_lines(symbol: Symbol, children: Sequence[Symbol]) -> frozenset[int]:
    """1-indexed line numbers to drop from a container's chunk: the spans of its
    child definitions, which are chunked separately."""
    if symbol.kind not in _CONTAINER_KINDS:
        return frozenset()
    excluded: set[int] = set()
    for child in children:
        excluded.update(range(child.span.line_start, child.span.line_end + 1))
    return frozenset(excluded)


def _normalized_body(
    lines: list[str], symbol: Symbol, excluded: frozenset[int]
) -> str:
    """Extract the symbol's lines (minus excluded child lines), strip trailing
    whitespace, and drop leading/trailing blank lines. This normalization is what
    makes the content hash stable against whitespace-only churn."""
    span: CodeSpan = symbol.span
    kept = [
        lines[lineno - 1].rstrip()
        for lineno in range(span.line_start, span.line_end + 1)
        if lineno not in excluded and lineno - 1 < len(lines)
    ]
    while kept and not kept[0]:
        kept.pop(0)
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept)
