"""The per-PR findings budget: at most N comments, chosen by severity x confidence.

ARCHITECTURE sec. 4.6 ends the pipeline here. Everything that survived the
verification gate is real as far as the system can tell; the budget is about
*attention*, which is a separate and scarcer resource. Twenty correct comments
on one pull request is not twenty times as useful as five -- past a handful,
reviewers skim, and the marginal comment costs more attention than it returns.

Two properties matter more than the ranking itself:

**Nothing is deleted.** Findings that miss the budget are marked suppressed and
kept. They are hidden from the pull request, not thrown away: the dashboard
shows them, the metrics count them, and a suppressed finding that later turns
out to matter is still on the record.

**The order is total and deterministic.** ``severity x confidence`` leaves ties,
and a tie broken by dict ordering would make the same run post different
comments on different days -- which would make prompt A/B comparison meaningless
and bug reports irreproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from app.domain.contracts import Finding, VerificationGate

__all__ = ["DEFAULT_BUDGET", "BudgetResult", "apply_findings_budget"]

DEFAULT_BUDGET: Final = 10
"""ARCHITECTURE sec. 4.6's default. Per-repository overrides live in
``repositories.config`` and are applied by the caller, not here."""


@dataclass(frozen=True, slots=True)
class BudgetResult:
    posted: tuple[Finding, ...]
    suppressed: tuple[Finding, ...]

    @property
    def total(self) -> int:
        return len(self.posted) + len(self.suppressed)


def _rank(finding: Finding) -> tuple[float, int, float, str, str]:
    """Total order over findings, most postable first.

    ``priority`` (severity x confidence) leads, as specified. The tiebreakers
    exist only to make the order total: severity rank so a CRITICAL outranks a
    HIGH at equal priority, then confidence, then the stable fingerprint, then
    the id. Negated where a larger value should sort earlier.
    """
    return (
        -finding.priority,
        -finding.severity.rank,
        -finding.confidence,
        finding.fingerprint,
        str(finding.id),
    )


def apply_findings_budget(
    findings: Sequence[Finding], *, limit: int = DEFAULT_BUDGET
) -> BudgetResult:
    """Select the findings worth a reviewer's attention; suppress the rest.

    Only postable findings compete: anything the verification gate already
    rejected is passed through as suppressed rather than being ranked, so a
    rejected finding can never consume a slot a real one wanted.
    """
    if limit < 0:
        raise ValueError("findings budget must not be negative")

    eligible = [f for f in findings if f.verification.is_postable]
    already_rejected = [f for f in findings if not f.verification.is_postable]

    ordered = sorted(eligible, key=_rank)
    posted = tuple(ordered[:limit])
    over_budget = tuple(_suppress(f) for f in ordered[limit:])

    return BudgetResult(
        posted=posted, suppressed=tuple(already_rejected) + over_budget
    )


def _suppress(finding: Finding) -> Finding:
    """Mark a postable finding as budgeted out.

    Recorded as a ``CONFIDENCE_FLOOR`` failure because that is the gate whose
    stated outcome is "suppress, retain in DB" -- the finding is not wrong, it
    simply ranked below the cut. Keeping it inside the existing vocabulary means
    the dashboard and the drop metrics need no special case for the budget.
    """
    return finding.reject(
        VerificationGate.CONFIDENCE_FLOOR,
        "over the per-pull-request findings budget",
    )
