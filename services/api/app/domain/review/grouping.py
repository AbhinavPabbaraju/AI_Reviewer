"""Group diff hunks into review units -- one unit, one LLM call.

The unit of review is a **symbol**, not a hunk. Three hunks inside one function
are one change: reviewing them separately asks the model the same question three
times, each time with a third of the picture, and then produces three comments a
human reads as one. Conversely two functions edited in the same file are two
changes and deserve two answers, even though git may render them as adjacent
hunks.

Grouping uses the parsed symbol table rather than the ``@@`` heading git
supplies. Git's heading is a regex-driven guess at the enclosing function that
is usually right and silently wrong for decorated definitions, nested classes,
and most TypeScript; the symbol table is what the resolver actually parsed, and
it is already indexed by span.

The enclosing symbol is the **innermost** one: a hunk inside a method belongs to
that method, not to its class and not to the module. That matters because the
symbol chosen here becomes the anchor of the context pack, and anchoring on the
class would retrieve the class's neighbours rather than the method's.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.base import Frozen
from app.domain.contracts import CodeSpan
from app.domain.indexing.models import Symbol, SymbolKind
from app.domain.retrieval.ports import SymbolIndexPort
from app.domain.review.diff import FileDiff, FileStatus, Hunk, ParsedDiff

__all__ = ["HunkGroup", "group_hunks"]


class HunkGroup(Frozen):
    """One review unit: the hunks that share an enclosing symbol."""

    path: str
    symbol_fqn: str | None
    """``None`` for changes with no enclosing symbol -- a new import, a tweak to
    a module-level constant, a change to a file the parsers do not cover. These
    are still reviewed; they simply anchor on the file rather than on a symbol."""

    span: CodeSpan
    """The span to anchor retrieval on: the symbol's body when there is one, so
    the model sees the whole function it is judging rather than only the lines
    that changed, and the union of the hunks otherwise."""

    hunks: tuple[Hunk, ...]

    @property
    def changed_lines(self) -> frozenset[int]:
        return frozenset(line for hunk in self.hunks for line in hunk.added_lines)

    @property
    def is_anchored(self) -> bool:
        return self.symbol_fqn is not None


async def group_hunks(
    diff: ParsedDiff, index: SymbolIndexPort, repository_id: str
) -> tuple[HunkGroup, ...]:
    """Group every reviewable hunk in ``diff`` by its enclosing symbol.

    One index round trip for the whole diff, not one per hunk: the port is
    set-at-a-time for exactly this reason, and a 40-file PR would otherwise be
    40 queries before the review has started.
    """
    reviewable = [
        file
        for file in diff.files
        if file.status is not FileStatus.DELETED and not file.is_binary and file.hunks
    ]
    if not reviewable:
        return ()

    spans = [
        CodeSpan(path=file.path, line_start=hunk.new_start, line_end=hunk.new_end)
        for file in reviewable
        for hunk in file.hunks
    ]
    symbols = await index.symbols_in_spans(repository_id, spans)
    by_path: dict[str, list[Symbol]] = {}
    for symbol in symbols:
        by_path.setdefault(symbol.span.path, []).append(symbol)

    groups: list[HunkGroup] = []
    for file in reviewable:
        groups.extend(_group_file(file, by_path.get(file.path, [])))
    return tuple(groups)


def _group_file(file: FileDiff, symbols: Sequence[Symbol]) -> list[HunkGroup]:
    buckets: dict[str | None, list[Hunk]] = {}
    anchors: dict[str | None, Symbol] = {}

    for hunk in file.hunks:
        enclosing = _innermost(symbols, hunk)
        key = enclosing.fqn if enclosing is not None else None
        buckets.setdefault(key, []).append(hunk)
        if enclosing is not None:
            anchors[key] = enclosing

    groups: list[HunkGroup] = []
    for key, hunks in buckets.items():
        anchor = anchors.get(key)
        if anchor is not None:
            span = anchor.span
        else:
            span = CodeSpan(
                path=file.path,
                line_start=min(hunk.new_start for hunk in hunks),
                line_end=max(hunk.new_end for hunk in hunks),
            )
        groups.append(
            HunkGroup(
                path=file.path, symbol_fqn=key, span=span, hunks=tuple(hunks)
            )
        )
    groups.sort(key=lambda group: (group.span.line_start, group.symbol_fqn or ""))
    return groups


def _innermost(symbols: Sequence[Symbol], hunk: Hunk) -> Symbol | None:
    """The tightest symbol overlapping the hunk.

    Modules are excluded rather than ranked last: a module symbol spans the whole
    file, so it always overlaps and would make every hunk "anchored" while
    telling retrieval nothing it did not already know from the path.
    """
    candidates = [
        symbol
        for symbol in symbols
        if symbol.kind is not SymbolKind.MODULE
        and symbol.span.line_start <= hunk.new_end
        and hunk.new_start <= symbol.span.line_end
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: (s.span.line_count, s.span.line_start))
