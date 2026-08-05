"""The M3 exit gate: zero fabrications escape, measured over 20 seeded PRs.

ROADMAP M3: *"on 20 seeded-defect PRs, **zero** findings escape the gate with a
nonexistent file, an out-of-range line, or an unparseable patch. That is a zero,
not a low number."*

Two design points make this a gate rather than a formality.

**The survivors are re-checked independently.** Asserting that the verifier
marked something postable would only prove the verifier agrees with itself. Every
postable finding is instead re-derived from scratch against the head tree here:
is the path in the tree, is the last cited line within the file, does the patch
parse. If the gate had a bug, that bug would have to exist identically in two
separately written implementations to pass.

**A gate that rejects everything is also broken.** Suppressing all output scores
a perfect zero on fabrications and makes the product useless, so the legitimate
finding seeded into every PR is asserted to survive. Both numbers are reported.

Run with ``-s`` to see them.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.domain.contracts import Finding, VerificationGate
from app.domain.indexing.models import Language
from app.domain.review.verification import (
    VerificationContext,
    VerificationReport,
    Verifier,
)
from app.infra.parsing.syntax import TreeSitterSyntaxChecker
from app.infra.review.head_files import MappingHeadFiles
from tests.eval.corpus.seeded_prs import (
    AdversarialReviewer,
    SeededPR,
    known_symbols,
    seeded_prs,
)

REQUIRED_PRS = 20
SYNTAX = TreeSitterSyntaxChecker()


@pytest.fixture(scope="module")
def prs() -> tuple[SeededPR, ...]:
    return seeded_prs()


@pytest.fixture(scope="module")
async def outcomes(
    prs: tuple[SeededPR, ...],
) -> list[tuple[SeededPR, tuple[Finding, ...], VerificationReport]]:
    reviewer = AdversarialReviewer()
    results = []
    for pr in prs:
        emitted = reviewer.review(pr)
        verifier = Verifier(
            files=MappingHeadFiles(pr.head_files),
            syntax=SYNTAX,
        )
        report = await verifier.verify(
            emitted,
            VerificationContext(
                diff=pr.diff, known_symbols=known_symbols(list(pr.head_files))
            ),
        )
        results.append((pr, emitted, report))
    return results


# -- independent re-verification ------------------------------------------- #


def _cites_missing_file(finding: Finding, pr: SeededPR) -> bool:
    return finding.location.path not in pr.head_files


def _cites_line_out_of_range(finding: Finding, pr: SeededPR) -> bool:
    source = pr.head_files.get(finding.location.path)
    if source is None:
        return True
    return finding.location.line_end > len(source.splitlines())


def _carries_unparseable_patch(finding: Finding) -> bool:
    if finding.improved_code is None:
        return False
    language = Language.for_path(finding.location.path)
    if language is None:
        return False
    import textwrap

    return not (
        SYNTAX.parses(language, finding.improved_code)
        or SYNTAX.parses(language, textwrap.dedent(finding.improved_code))
    )


class TestVerificationGate:
    def test_corpus_has_twenty_seeded_prs(
        self, prs: tuple[SeededPR, ...]
    ) -> None:
        assert len(prs) == REQUIRED_PRS
        # Every PR must actually change something, or the gate has nothing to
        # measure "within changed lines" against.
        for pr in prs:
            assert pr.diff.changed_lines(pr.path), f"{pr.slug} changed nothing"

    def test_zero_fabrications_escape(
        self,
        outcomes: list[tuple[SeededPR, tuple[Finding, ...], VerificationReport]],
    ) -> None:
        """The M3 exit criterion. A zero, not a low number."""
        emitted = sum(len(e) for _, e, _ in outcomes)
        postable = sum(len(r.postable) for _, _, r in outcomes)

        escapes: list[str] = []
        for pr, _, report in outcomes:
            for finding in report.postable:
                if _cites_missing_file(finding, pr):
                    escapes.append(
                        f"{pr.slug}: nonexistent file {finding.location}"
                    )
                if _cites_line_out_of_range(finding, pr):
                    escapes.append(
                        f"{pr.slug}: out-of-range line {finding.location}"
                    )
                if _carries_unparseable_patch(finding):
                    escapes.append(
                        f"{pr.slug}: unparseable patch on {finding.location}"
                    )

        print(
            f"\n[gate] {len(outcomes)} seeded PRs | {emitted} findings emitted, "
            f"{postable} postable, {emitted - postable} stopped "
            f"({(emitted - postable) / emitted:.0%} drop rate)"
        )
        for escape in escapes:
            print(f"    ESCAPED: {escape}")
        assert escapes == [], (
            f"{len(escapes)} fabrications escaped the gate; the M3 exit "
            "criterion is zero"
        )

    def test_drops_are_attributed_per_gate(
        self,
        outcomes: list[tuple[SeededPR, tuple[Finding, ...], VerificationReport]],
    ) -> None:
        """sec. 4.6: a bad prompt change must show as a shift in one gate, not a
        vague quality dip. Every gate that the corpus provokes must fire."""
        totals: dict[VerificationGate, int] = {}
        for _, _, report in outcomes:
            for gate, count in report.drops_by_gate.items():
                totals[gate] = totals.get(gate, 0) + count

        for gate in sorted(totals, key=lambda g: -totals[g]):
            print(f"[gate] {gate.value:18} {totals[gate]:4d}")

        # The adversarial reviewer seeds one of each; all seven must be exercised
        # or the corpus has stopped covering a gate.
        assert set(totals) == set(VerificationGate), (
            f"gates never exercised: {set(VerificationGate) - set(totals)}"
        )

    def test_the_legitimate_finding_survives_every_pr(
        self,
        outcomes: list[tuple[SeededPR, tuple[Finding, ...], VerificationReport]],
    ) -> None:
        """The other half of the criterion.

        A gate that suppressed everything would pass the zero-escapes test
        perfectly and ship a product that never comments. Each PR seeds exactly
        one finding that is true, well-anchored and confident; it must come out
        the other side.
        """
        survived = 0
        for pr, _, report in outcomes:
            anchored = [
                f
                for f in report.postable
                if f.location.path == pr.path
                and f.location.line_start == pr.changed_line
            ]
            if anchored:
                survived += 1
            else:
                print(f"    LOST: {pr.slug} kept no finding on its changed line")
        print(f"[gate] legitimate finding survived in {survived}/{len(outcomes)} PRs")
        assert survived == len(outcomes)

    def test_rejected_findings_are_retained_for_metrics(
        self,
        outcomes: list[tuple[SeededPR, tuple[Finding, ...], VerificationReport]],
    ) -> None:
        """Suppressed findings are hidden from the PR, not thrown away."""
        for _, emitted, report in outcomes:
            assert len(report.findings) == len(emitted)

    def test_every_postable_finding_still_satisfies_the_contract(
        self,
        outcomes: list[tuple[SeededPR, tuple[Finding, ...], VerificationReport]],
    ) -> None:
        """Demotion and patch-stripping rebuild findings; the invariants that
        make a finding trustworthy have to survive that."""
        for _, _, report in outcomes:
            for finding in report.postable:
                assert finding.evidence, "evidence is required and non-empty"
                assert any(
                    e.span.overlaps(finding.location) for e in finding.evidence
                ), "a defect site must overlap the location"
                assert 0.0 <= finding.confidence <= 1.0
                if finding.improved_code is not None:
                    assert finding.suggested_fix is not None, (
                        "a patch must be explained in prose"
                    )


class TestGateCannotTriviallyPass:
    """Guards on the measurement itself."""

    def test_the_adversarial_reviewer_really_fabricates(
        self, prs: tuple[SeededPR, ...]
    ) -> None:
        """If the fabrications stopped being invalid, the exit criterion would
        be measuring nothing."""
        reviewer = AdversarialReviewer()
        missing = out_of_range = unparseable = 0
        for pr in prs:
            for finding in reviewer.review(pr):
                missing += _cites_missing_file(finding, pr)
                out_of_range += _cites_line_out_of_range(finding, pr)
                unparseable += _carries_unparseable_patch(finding)
        print(
            f"\n[gate] pre-verification the corpus contains {missing} missing-file, "
            f"{out_of_range} out-of-range and {unparseable} unparseable-patch "
            "findings"
        )
        assert missing >= len(prs)
        assert out_of_range >= len(prs)
        assert unparseable >= len(prs)

    def test_removing_the_gate_would_fail_the_criterion(
        self, prs: tuple[SeededPR, ...]
    ) -> None:
        """The counterfactual, stated as a test: with no verification at all,
        every one of those fabrications is postable. This is what the gate is
        worth."""
        reviewer = AdversarialReviewer()
        unverified: Sequence[Finding] = [
            finding for pr in prs for finding in reviewer.review(pr)
        ]
        escapes = sum(
            1
            for pr in prs
            for finding in reviewer.review(pr)
            if _cites_missing_file(finding, pr)
            or _cites_line_out_of_range(finding, pr)
            or _carries_unparseable_patch(finding)
        )
        assert escapes > 0
        print(
            f"[gate] without verification, {escapes} of {len(unverified)} "
            "findings would reach the pull request"
        )
