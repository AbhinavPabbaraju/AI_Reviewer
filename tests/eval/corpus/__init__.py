"""Hand-labeled resolution corpora — the M1 exit gate's ground truth.

Each corpus is a small but *realistic* repository: multi-package, with the import
shapes that actually occur in production code (relative imports from a package
``__init__``, module aliases, barrel re-exports, namespace imports, abstract base
classes, decorators, type annotations). Every expectation is a hand-labeled
ground-truth edge — the resolution a competent reader of the corpus would make,
written by reading the source, *not* by recording what the resolver happens to
emit.

That distinction is the whole point. A corpus labeled from resolver output can
only ever score 100% and gates nothing. These corpora deliberately contain
references the resolver gets wrong (dispatch through a parameter whose type is
only known from its annotation, a method called on a locally-constructed object),
because the gate is meant to have room to detect a regression *and* honest
headroom above the 0.85 floor.

Three label kinds, because "resolved" is not the only correct answer:

``"pkg.mod.symbol"``
    Must resolve to exactly that repository symbol.
``EXTERNAL("os.getcwd")``
    Must be recognized as leaving the repository (stdlib/vendor) under that
    textual name. Correct, not a failure — and excluded from the operational
    resolution-rate denominator.
``UNRESOLVED("execute")``
    Must be kept with that textual target and *not* bound to a repository symbol.
    This is the label that punishes over-eager resolution: a resolver that
    guesses ``self._conn.execute()`` into some same-named method in another class
    has fabricated an edge, and fabrication is the failure mode this project
    exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.indexing.models import EdgeKind

__all__ = ["EXTERNAL", "UNRESOLVED", "Expectation", "Target"]


@dataclass(frozen=True, slots=True)
class EXTERNAL:
    """Ground truth: leaves the repository, under this textual name."""

    name: str


@dataclass(frozen=True, slots=True)
class UNRESOLVED:
    """Ground truth: no repository symbol; must be kept textual, never guessed."""

    name: str


type Target = str | EXTERNAL | UNRESOLVED
type Expectation = tuple[EdgeKind, str, Target]
