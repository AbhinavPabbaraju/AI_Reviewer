"""Unified-diff parsing and hunk grouping -- Stage V's input, Stage VI's yardstick.

Two consumers with different needs, which is why this is one module:

* **The reviewer** needs the diff grouped into units worth one LLM call each.
  A hunk on its own is a poor unit -- three hunks inside one function are one
  change and reviewing them separately asks the model the same question three
  times with less context each time -- so hunks are grouped by their enclosing
  symbol.
* **The verification gate** needs the exact set of lines the PR changed. This is
  the yardstick for ``LINE_IN_RANGE``, and it is the only reason the gate can
  tell "a real bug on line 42" from "a real bug on line 42 of a file this PR did
  not touch", which is noise the author cannot act on.

**Changed lines are new-file line numbers, and deletions contribute none.** A
review comment is anchored at the head SHA; a line that the PR deleted does not
exist there, so there is nothing to anchor to and nothing GitHub would render.
Findings about deleted code have to attach to surviving lines or not at all.

The parser is deliberately strict about structure and forgiving about content:
it validates that hunk line counts match their ``@@`` header, because a diff
whose headers disagree with its body is corrupt and silently mis-numbering
findings is far worse than failing, but it passes through anything inside a line
untouched.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Final

from app.domain.base import Frozen

__all__ = [
    "FileDiff",
    "FileStatus",
    "Hunk",
    "ParsedDiff",
    "parse_unified_diff",
]

_DIFF_HEADER: Final = re.compile(r"^diff --git (?P<a>.+?) (?P<b>.+)$")
_HUNK_HEADER: Final = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<context>.*)$"
)
_DEV_NULL: Final = "/dev/null"


class FileStatus(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class Hunk(Frozen):
    """One ``@@`` block.

    ``added_lines`` are numbered in the *new* file and ``removed_lines`` in the
    old one; keeping both is what lets a caller tell a pure deletion (no added
    lines) from a rewrite.
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    added_lines: tuple[int, ...] = ()
    removed_lines: tuple[int, ...] = ()
    heading: str = ""
    """Trailing text on the ``@@`` line -- git's guess at the enclosing
    function. A hint for humans reading the diff, never used for grouping:
    ``symbols_in_spans`` answers that question from the parsed tree, which is
    right where git's heuristic is only usually right."""

    @property
    def new_end(self) -> int:
        """Last new-file line this hunk covers, inclusive."""
        return self.new_start + max(self.new_count, 1) - 1


class FileDiff(Frozen):
    """Every hunk touching one file, plus how the file itself changed."""

    path: str
    """The path at the head SHA -- the one a finding must cite. For a deletion
    there is no such path, so this carries the removed file's path and
    ``changed_lines`` is empty; nothing can be anchored in a file that is gone."""

    old_path: str | None = None
    status: FileStatus = FileStatus.MODIFIED
    is_binary: bool = False
    hunks: tuple[Hunk, ...] = ()

    @property
    def changed_lines(self) -> frozenset[int]:
        """New-file lines this diff added or rewrote."""
        return frozenset(line for hunk in self.hunks for line in hunk.added_lines)

    @property
    def touched_lines(self) -> frozenset[int]:
        """Changed lines plus the context around them.

        A finding may legitimately point at an unchanged line *inside* a hunk --
        "this new call is wrong because of the guard three lines above it" -- and
        that guard is on the reviewer's screen. Used by the gate's lenient mode;
        ``changed_lines`` remains the strict answer.
        """
        touched: set[int] = set()
        for hunk in self.hunks:
            touched.update(range(hunk.new_start, hunk.new_end + 1))
        return frozenset(touched)


class ParsedDiff(Frozen):
    """A whole pull request's diff."""

    files: tuple[FileDiff, ...] = ()

    @property
    def paths(self) -> frozenset[str]:
        """Paths present at head. Deletions are excluded -- they are not there."""
        return frozenset(
            f.path for f in self.files if f.status is not FileStatus.DELETED
        )

    def for_path(self, path: str) -> FileDiff | None:
        return next((f for f in self.files if f.path == path), None)

    def changed_lines(self, path: str) -> frozenset[int]:
        file = self.for_path(path)
        return file.changed_lines if file else frozenset()

    def touched_lines(self, path: str) -> frozenset[int]:
        file = self.for_path(path)
        return file.touched_lines if file else frozenset()

    @property
    def total_added_lines(self) -> int:
        return sum(len(f.changed_lines) for f in self.files)


def parse_unified_diff(text: str) -> ParsedDiff:
    """Parse ``git diff`` output.

    Raises ``ValueError`` on a hunk whose body disagrees with its ``@@`` header.
    That is not pedantry: line numbers from this parser decide which findings are
    postable and where GitHub renders them, so a corrupt diff must fail loudly
    rather than produce comments anchored a few lines off.
    """
    files: list[FileDiff] = []
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        header = _DIFF_HEADER.match(lines[index])
        if header is None:
            index += 1
            continue
        index += 1
        parsed, index = _parse_file(lines, index, header)
        if parsed is not None:
            files.append(parsed)

    return ParsedDiff(files=tuple(files))


def _parse_file(
    lines: Sequence[str], index: int, header: re.Match[str]
) -> tuple[FileDiff | None, int]:
    old_path = _strip_prefix(header.group("a"))
    new_path = _strip_prefix(header.group("b"))
    status = FileStatus.MODIFIED
    is_binary = False
    hunks: list[Hunk] = []

    while index < len(lines):
        line = lines[index]
        if _DIFF_HEADER.match(line):
            break
        if line.startswith("--- "):
            if line[4:].strip() == _DEV_NULL:
                status = FileStatus.ADDED
            index += 1
        elif line.startswith("+++ "):
            if line[4:].strip() == _DEV_NULL:
                status = FileStatus.DELETED
            index += 1
        elif line.startswith("rename from "):
            status = FileStatus.RENAMED
            old_path = line.removeprefix("rename from ").strip()
            index += 1
        elif line.startswith("rename to "):
            status = FileStatus.RENAMED
            new_path = line.removeprefix("rename to ").strip()
            index += 1
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            is_binary = True
            index += 1
        elif line.startswith("@@"):
            hunk, index = _parse_hunk(lines, index)
            hunks.append(hunk)
        else:
            index += 1

    path = old_path if status is FileStatus.DELETED else new_path
    if path is None:
        return None, index
    return (
        FileDiff(
            path=path,
            old_path=old_path if old_path != new_path else None,
            status=status,
            is_binary=is_binary,
            hunks=tuple(hunks),
        ),
        index,
    )


def _parse_hunk(lines: Sequence[str], index: int) -> tuple[Hunk, int]:
    match = _HUNK_HEADER.match(lines[index])
    if match is None:  # pragma: no cover - guarded by the caller
        raise ValueError(f"malformed hunk header: {lines[index]!r}")

    old_start = int(match.group("old_start"))
    new_start = int(match.group("new_start"))
    # An omitted count means 1 (`@@ -3 +3 @@`); an explicit 0 means the file is
    # empty on that side, and the start line is then one *below* the first real
    # line, which is why counts are tracked rather than inferred from the body.
    old_count = int(match.group("old_count") or 1)
    new_count = int(match.group("new_count") or 1)
    index += 1

    added: list[int] = []
    removed: list[int] = []
    old_line = old_start
    new_line = new_start
    seen_old = 0
    seen_new = 0

    while index < len(lines):
        line = lines[index]
        if line.startswith("@@") or line.startswith("diff --git"):
            break
        if line.startswith("\\"):  # "\ No newline at end of file"
            index += 1
            continue
        if seen_old >= old_count and seen_new >= new_count:
            break

        marker = line[:1]
        if marker == "+":
            added.append(new_line)
            new_line += 1
            seen_new += 1
        elif marker == "-":
            removed.append(old_line)
            old_line += 1
            seen_old += 1
        elif marker in (" ", ""):
            # An empty string is a context line whose trailing space was stripped
            # somewhere in transit -- common enough in payloads that rejecting it
            # would fail on real PRs.
            old_line += 1
            new_line += 1
            seen_old += 1
            seen_new += 1
        else:
            break
        index += 1

    if seen_old != old_count or seen_new != new_count:
        raise ValueError(
            f"hunk @@ -{old_start},{old_count} +{new_start},{new_count} @@ declares "
            f"{old_count} old and {new_count} new lines but its body has "
            f"{seen_old} and {seen_new}; refusing to guess line numbers"
        )

    return (
        Hunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            added_lines=tuple(added),
            removed_lines=tuple(removed),
            heading=match.group("context").strip(),
        ),
        index,
    )


def _strip_prefix(path: str) -> str | None:
    """Drop git's ``a/``/``b/`` prefixes and unquote a quoted path."""
    cleaned = path.strip()
    if cleaned == _DEV_NULL:
        return None
    if cleaned.startswith('"') and cleaned.endswith('"'):
        # Git quotes paths containing special characters, with C-style escapes.
        cleaned = cleaned[1:-1].encode().decode("unicode_escape")
    for prefix in ("a/", "b/"):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix) :]
    return cleaned


def changed_lines_by_path(diff: ParsedDiff) -> Mapping[str, frozenset[int]]:
    """``path -> changed lines``, the shape the verification gate wants."""
    return {file.path: file.changed_lines for file in diff.files}
