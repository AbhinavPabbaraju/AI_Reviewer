"""Scoring posted comments against the labels.

ROADMAP M6 names the metrics: *precision, recall, FP rate, verification drop rate
by gate, cost, latency.* They are computed here and nowhere else, from two inputs
only -- what the pipeline posted, and what ``labeled_prs`` says is true. The
transcript is not imported, and must not be: a scorer that could see what the
reviewer intended would be grading intent rather than output.

**How a comment is matched to a defect.** Location overlap with the labeled
defect line, *and* one of the defect's ``signals`` present in the title or
explanation. Location alone is not enough -- the corpus contains cases where the
reviewer says something confident and wrong on exactly the right line, and
scoring those as hits would report a recall the system has not earned. The
signal check is a stand-in for a human adjudicator and is imperfect in the usual
way: it can miss a right answer phrased unusually. It is stated in the label, in
the open, so a disputed case is arguable rather than buried in the scorer.

**Redundant comments count against precision.** Two comments on one defect is one
finding and one nuisance, and the ``NOT_DUPLICATE`` gate exists to prevent
exactly that; scoring both as hits would remove the gate's incentive.

**Every number is computed at two stages.** Once over what the model emitted
(post-decode, pre-verification) and once over what was posted. The pair is the
point: the absolute figures are a property of a fixed transcript and prove
little on their own, while the *difference* between them is the pipeline's
contribution and is a property of the code under test.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.domain.contracts import Finding, VerificationGate
from tests.eval.corpus.labeled_prs import LabeledPR
from tests.eval.harness.runner import CaseRun

__all__ = ["CaseScore", "EvalReport", "Stage", "score"]


def _matches(finding: Finding, case: LabeledPR) -> bool:
    """Whether this comment is about this case's labeled defect."""
    defect = case.defect
    if defect is None:
        return False
    location = finding.location
    if location.path != case.path:
        return False
    if not (location.line_start <= defect.line <= location.line_end):
        return False
    return defect.described_by(f"{finding.title}\n{finding.explanation}")


@dataclass(frozen=True, slots=True)
class Stage:
    """Precision, recall and false-positive rate over one set of comments."""

    label: str
    agreed: int
    """Comments that describe this case's labeled defect -- the ones a human
    reviewer would agree with, which is how ARCHITECTURE sec. 1 defines
    ``comment_precision``."""

    wrong: int
    """Comments that describe no labeled defect. On a clean pull request, every
    comment is one of these."""

    redundant: int
    """Of ``agreed``, the ones repeating a defect another comment already
    covered. A reviewer agrees with a redundant comment and is still annoyed by
    it, so it is counted in both precisions but only one of them penalizes it."""

    defects_found: int
    defects_total: int
    cases_total: int
    noisy_clean_cases: int
    """Clean pull requests that received at least one comment. The sharpest
    precision signal in the corpus: on these the correct output is silence."""

    @property
    def comments(self) -> int:
        return self.agreed + self.wrong

    @property
    def precision(self) -> float:
        """The SLO's metric: posted findings a human agrees with.

        A duplicate of a true finding is counted here as correct, because it
        is: the author reads it and agrees. Charging it as a false positive
        would be a stricter bar than the one the system is held to, and
        reporting that number as *the* precision would understate the product
        against its own definition.
        """
        return self.agreed / self.comments if self.comments else 1.0

    @property
    def precision_strict(self) -> float:
        """The same, with redundancy charged as noise.

        Reported alongside because the gap between the two is exactly what
        imperfect dedup costs a reader, and a single number would hide it.
        """
        useful = self.agreed - self.redundant
        return useful / self.comments if self.comments else 1.0

    @property
    def recall(self) -> float:
        return self.defects_found / self.defects_total if self.defects_total else 0.0

    @property
    def false_positives_per_pr(self) -> float:
        return self.wrong / self.cases_total if self.cases_total else 0.0


@dataclass(frozen=True, slots=True)
class CaseScore:
    """One case's verdict, kept so a disagreement can be argued case by case."""

    slug: str
    is_clean: bool
    defect_kind: str | None
    found: bool
    posted_true: tuple[str, ...]
    posted_false: tuple[str, ...]
    suppressed_true: tuple[str, ...]
    """True findings the pipeline emitted and then withheld. ARCHITECTURE sec. 11
    lists 'verification gate suppresses true positives' as a live risk; this is
    the measurement of it."""

    @property
    def lost_to_verification(self) -> bool:
        """The severe form of that risk: the reviewer found the defect and the
        pipeline silenced every comment about it. Withholding one of two true
        comments costs nothing; withholding the last one costs the defect."""
        return bool(self.suppressed_true) and not self.found


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The whole run: both stages, the per-gate drops, and the costs."""

    transcript: str
    emitted: Stage
    posted: Stage
    cases: tuple[CaseScore, ...]
    drops_by_gate: Mapping[VerificationGate, int] = field(default_factory=dict)
    decode_rejects: int = 0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    durations: tuple[float, ...] = ()
    cassette_misses: int = 0

    @property
    def precision_gain(self) -> float:
        """What verification and the budget are worth on this corpus."""
        return self.posted.precision - self.emitted.precision

    @property
    def recall_cost(self) -> float:
        """What they cost, in the same units. Never hidden: a gate that buys
        precision by suppressing everything must be visible as a number."""
        return self.emitted.recall - self.posted.recall

    @property
    def suppressed_true_findings(self) -> int:
        return sum(len(c.suppressed_true) for c in self.cases)

    @property
    def defects_lost_to_verification(self) -> int:
        """Defects the reviewer found and the pipeline then buried. This is the
        number that must stay at zero: precision bought by silencing true
        findings is not precision, it is a quieter product."""
        return sum(1 for c in self.cases if c.lost_to_verification)

    @property
    def p50_seconds(self) -> float:
        return statistics.median(self.durations) if self.durations else 0.0

    @property
    def p95_seconds(self) -> float:
        if not self.durations:
            return 0.0
        ordered = sorted(self.durations)
        index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
        return ordered[index]

    @property
    def total_seconds(self) -> float:
        return sum(self.durations)


def _stage(
    label: str, comments_by_case: Sequence[tuple[LabeledPR, Sequence[Finding]]]
) -> Stage:
    agreed = redundant = wrong = 0
    defects_found = defects_total = 0
    noisy_clean = 0

    for case, comments in comments_by_case:
        matched = [f for f in comments if _matches(f, case)]
        if case.defect is not None:
            defects_total += 1
            if matched:
                defects_found += 1
        agreed += len(matched)
        redundant += max(len(matched) - 1, 0)
        wrong += len(comments) - len(matched)
        if case.is_clean and comments:
            noisy_clean += 1

    return Stage(
        label=label,
        agreed=agreed,
        wrong=wrong,
        redundant=redundant,
        defects_found=defects_found,
        defects_total=defects_total,
        cases_total=len(comments_by_case),
        noisy_clean_cases=noisy_clean,
    )


def score(runs: Sequence[CaseRun], *, name: str = "") -> EvalReport:
    """Score a whole harness run. Pure: no I/O, no pipeline, no transcript.

    ``name`` labels the configuration in the report and is the only thing the
    scorer is told about the reviewer that produced these runs.
    """
    drops: dict[VerificationGate, int] = {}
    for run in runs:
        for gate, count in run.report.drops_by_gate.items():
            drops[gate] = drops.get(gate, 0) + count

    cases: list[CaseScore] = []
    for run in runs:
        case = run.case
        posted_true = [f for f in run.posted if _matches(f, case)]
        posted_false = [f for f in run.posted if not _matches(f, case)]
        withheld_true = [
            f
            for f in run.emitted
            if _matches(f, case) and not any(p.fingerprint == f.fingerprint
                                             for p in run.posted)
        ]
        cases.append(
            CaseScore(
                slug=case.slug,
                is_clean=case.is_clean,
                defect_kind=case.defect.kind if case.defect else None,
                found=bool(posted_true),
                posted_true=tuple(f.title for f in posted_true),
                posted_false=tuple(f.title for f in posted_false),
                suppressed_true=tuple(f.title for f in withheld_true),
            )
        )

    return EvalReport(
        transcript=name,
        emitted=_stage("emitted", [(r.case, r.emitted) for r in runs]),
        posted=_stage("posted", [(r.case, r.posted) for r in runs]),
        cases=tuple(cases),
        drops_by_gate=drops,
        decode_rejects=sum(len(r.decode_rejects) for r in runs),
        cost_usd=sum(r.run.cost_usd for r in runs),
        tokens_in=sum(r.run.tokens_in for r in runs),
        tokens_out=sum(r.run.tokens_out for r in runs),
        durations=tuple(r.duration_s for r in runs),
        cassette_misses=sum(r.cassette_misses for r in runs),
    )
