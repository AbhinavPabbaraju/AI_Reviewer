"""Incremental re-index planning (ARCHITECTURE.md sec. 4.1).

> Incrementality is content-addressed. On re-index, only chunks whose hash
> changed are re-embedded. A one-line change to a 50k-file repo re-embeds ~1
> chunk, not 50k.

This module is the pure diff: given the file->blob_sha map of the current commit
and of the previous snapshot, decide what changed. The expensive consequence --
"parse only added and modified files, take the rest from the content-addressed
cache" -- is what lets a single-file push re-index in under 10 s (M1 exit).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["IndexPlan", "plan_index"]


@dataclass(frozen=True, slots=True)
class IndexPlan:
    """The four disjoint buckets of a re-index, each a sorted tuple of paths."""

    added: tuple[str, ...]
    modified: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]

    @property
    def to_parse(self) -> tuple[str, ...]:
        """Files whose content changed and must be (re)parsed. Everything else
        is served from the parse cache or dropped."""
        return tuple(sorted((*self.added, *self.modified)))

    @property
    def is_full_reindex(self) -> bool:
        """True when there is no prior snapshot to diff against."""
        return not (self.modified or self.removed or self.unchanged)

    @property
    def reused_count(self) -> int:
        return len(self.unchanged)


def plan_index(
    current: Mapping[str, str], previous: Mapping[str, str]
) -> IndexPlan:
    """Diff two ``path -> blob_sha`` maps.

    ``current`` is the set of indexable files at the head commit (already
    filtered); ``previous`` is the same for the last ready snapshot. A path whose
    blob sha is unchanged is ``unchanged`` and will not be re-parsed.
    """
    added: list[str] = []
    modified: list[str] = []
    unchanged: list[str] = []
    for path, blob_sha in current.items():
        prior = previous.get(path)
        if prior is None:
            added.append(path)
        elif prior != blob_sha:
            modified.append(path)
        else:
            unchanged.append(path)
    removed = [path for path in previous if path not in current]
    return IndexPlan(
        added=tuple(sorted(added)),
        modified=tuple(sorted(modified)),
        removed=tuple(sorted(removed)),
        unchanged=tuple(sorted(unchanged)),
    )
