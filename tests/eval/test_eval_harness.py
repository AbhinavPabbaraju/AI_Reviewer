"""M6 (partial): the evaluation harness, measured over 30 labeled pull requests.

ROADMAP sequencing puts a partial M6 -- *harness + 30 cases* -- immediately after
M3, before the analyzers and the GitHub App. This is that gate.

What it establishes:

* the pipeline runs end to end against a fake ``GitHubPort`` and a recorded
  ``LLMPort``, **free** and **byte-for-byte reproducible**;
* precision, recall, false-positive rate, per-gate drops, cost and wall clock are
  *measured and printed* over hand-labeled ground truth;
* the metric is **sensitive** -- a deliberately worse reviewer scores visibly
  worse, which is the property M6-full's CI gate is built on.

The last one carries the milestone. A number that cannot move is not a gate, and
the demonstration here is sharper than "precision fell": the degraded reviewer
posts fourteen extra false positives and the verification gate's drop counts do
not change *at all*. Every gate metric the system had before this milestone
would have reported that regression as a perfectly healthy run. That is the
argument for a labeled corpus, stated as an assertion.

Run with ``-s`` to see the numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from app.domain.contracts import VerificationGate
from tests.eval.corpus.labeled_prs import LabeledPR, labeled_prs
from tests.eval.harness.metrics import EvalReport, score
from tests.eval.harness.runner import CaseRun, EvalHarness
from tests.eval.harness.transcript import REVIEWER_V1, with_extra_speculation

REQUIRED_CASES = 30
RUN_BUDGET_SECONDS = 600.0
"""M6's exit criterion is a full eval run in under ten minutes. Asserted here at
30 cases so the budget is already in force when the corpus grows to 120."""

PRECISION_SLO = 0.80
RECALL_SLO = 0.55
"""ARCHITECTURE sec. 1. Reported against, not asserted at: see
``test_precision_and_recall`` for why the enforced floor sits below the SLO."""


@pytest.fixture(scope="module")
def cases() -> tuple[LabeledPR, ...]:
    return labeled_prs()


@dataclass(frozen=True, slots=True)
class Baseline:
    """One run of the corpus, kept alongside the harness that produced it.

    The harness has to outlive the run: its ``github`` is the boundary the
    posted comments are read back off, and that is where the output is scored.
    """

    harness: EvalHarness
    runs: tuple[CaseRun, ...]


@pytest.fixture(scope="module")
async def baseline(cases: tuple[LabeledPR, ...]) -> Baseline:
    harness = EvalHarness(transcript=REVIEWER_V1)
    return Baseline(harness=harness, runs=await harness.run(cases))


@pytest.fixture(scope="module")
def runs(baseline: Baseline) -> tuple[CaseRun, ...]:
    return baseline.runs


@pytest.fixture(scope="module")
def report(runs: tuple[CaseRun, ...]) -> EvalReport:
    return score(runs, name=REVIEWER_V1.name)


@pytest.fixture(scope="module")
async def degraded(cases: tuple[LabeledPR, ...]) -> EvalReport:
    """The same corpus reviewed by a deliberately more trigger-happy reviewer."""
    transcript = with_extra_speculation(REVIEWER_V1, name="reviewer/v1-speculative")
    return score(
        await EvalHarness(transcript=transcript).run(cases), name=transcript.name
    )


def _fingerprint(runs: Sequence[CaseRun]) -> tuple[tuple[object, ...], ...]:
    """A stable projection of everything posted.

    Deliberately excludes ``Finding.id`` and ``created_at``: both are fresh on
    every construction by design, and comparing them would prove only that uuid4
    works. Everything a pull request would actually display is included.
    """
    return tuple(
        (
            run.slug,
            finding.location.path,
            finding.location.line_start,
            finding.location.line_end,
            finding.title,
            finding.severity.value,
            round(finding.confidence, 9),
            finding.verification.status.value,
            tuple(g.value for g in finding.verification.gates_failed),
            finding.improved_code,
        )
        for run in runs
        for finding in run.posted
    )


class TestTheHarnessRunsFreeAndReproducibly:
    """M6's stated reason for the port boundary, verified rather than assumed."""

    def test_the_corpus_has_thirty_labeled_cases(
        self, cases: tuple[LabeledPR, ...]
    ) -> None:
        """The partial milestone's size. ``labeled_prs`` validates each case as
        it builds it, so reaching thirty already means thirty *valid* ones."""
        clean = sum(1 for case in cases if case.is_clean)
        print(
            f"\n[eval] {len(cases)} labeled pull requests | "
            f"{len(cases) - clean} seeded defects, {clean} clean | "
            f"{sum(1 for c in cases if not c.typescript)} Python, "
            f"{sum(1 for c in cases if c.typescript)} TypeScript"
        )
        assert len(cases) == REQUIRED_CASES
        assert clean >= 8, (
            "precision is measured on the clean cases; too few and a "
            "trigger-happy reviewer has nothing to be caught by"
        )

    def test_the_whole_run_costs_nothing(self, report: EvalReport) -> None:
        """The number every configuration comparison is anchored on. An adapter
        that reported a price would corrupt the M6 harness's only cost axis."""
        print(
            f"[eval] cost ${report.cost_usd:.4f} | "
            f"{report.tokens_in} tokens in, {report.tokens_out} out"
        )
        assert report.cost_usd == 0.0
        assert report.tokens_in > 0, "a free run is not the same as an empty one"

    def test_no_provider_was_ever_reachable(self, runs: tuple[CaseRun, ...]) -> None:
        """Every response came from the cassette, and a miss would have raised.

        This is what makes the run reproducible: the cassette is keyed by the
        exact prompt, so a prompt change is a loud failure here rather than a
        quiet shift in the numbers.
        """
        assert sum(r.cassette_misses for r in runs) == 0
        # One call per review unit, every one of them served from the recording.
        assert sum(r.cassette_hits for r in runs) == sum(len(r.groups) for r in runs)

    async def test_two_runs_post_identical_comments(
        self, cases: tuple[LabeledPR, ...], runs: tuple[CaseRun, ...]
    ) -> None:
        """Determinism, end to end. Without it every metric below is noise, and
        an A/B between two prompts measures the weather."""
        again = await EvalHarness(transcript=REVIEWER_V1).run(cases)
        assert _fingerprint(again) == _fingerprint(runs)

    async def test_the_output_was_measured_at_the_github_boundary(
        self, baseline: Baseline
    ) -> None:
        """Every case reached ``post_review``, and what it carried is what was
        scored.

        The harness measures the port call rather than its own bookkeeping, so
        the numbers below describe what a pull request would have received. A
        review is posted even when there is nothing to say -- staying silent by
        never calling is a different behaviour from reviewing and finding
        nothing, and only one of them is honest to the author.
        """
        posted = baseline.harness.github.posted
        assert len(posted) == len(baseline.runs)
        assert [tuple(f.title for f in review.findings) for review in posted] == [
            tuple(f.title for f in run.posted) for run in baseline.runs
        ]
        quiet = [review for review in posted if not review.findings]
        assert quiet, "no case produced a clean review"
        assert all("no comments" in review.summary for review in quiet)

    def test_a_full_run_fits_the_time_budget(self, report: EvalReport) -> None:
        print(
            f"[eval] {report.total_seconds:.1f}s total of a "
            f"{RUN_BUDGET_SECONDS:.0f}s budget | p50 {report.p50_seconds * 1000:.0f}ms, "
            f"p95 {report.p95_seconds * 1000:.0f}ms per pull request"
        )
        assert report.total_seconds < RUN_BUDGET_SECONDS


class TestMeasuredQuality:
    """The metrics ROADMAP M6 names, over hand-labeled ground truth."""

    def test_precision_and_recall(self, report: EvalReport) -> None:
        """The headline numbers, and the floors that make them a gate.

        The enforced floors sit *below* the published SLOs on purpose. The
        Two precisions are reported. ``precision`` is the SLO's own metric --
        posted findings a human agrees with -- and a duplicate of a true finding
        counts as agreed with, because it is. ``precision_strict`` charges that
        duplicate as noise. The gap between them is what imperfect dedup costs a
        reader, and reporting only one number would hide it.

        The absolute values are a property of a fixed transcript rather than of
        a model, so the SLO is not the thing being proved here; what these
        floors catch is a *regression*, since transcript and labels are fixed
        and any movement is therefore movement in the pipeline.
        """
        posted, emitted = report.posted, report.emitted
        for stage in (emitted, posted):
            print(
                f"[eval] {stage.label:8} {stage.comments:3d} comments | "
                f"precision {stage.precision:.3f} "
                f"(strict {stage.precision_strict:.3f}) | "
                f"recall {stage.recall:.3f} "
                f"({stage.defects_found}/{stage.defects_total}) | "
                f"{stage.false_positives_per_pr:.2f} false positives per PR"
            )
        print(
            f"[eval] against the SLOs: precision {posted.precision:.3f} vs "
            f"{PRECISION_SLO:.2f} "
            f"({'PASS' if posted.precision >= PRECISION_SLO else 'BELOW'}), "
            f"recall {posted.recall:.3f} vs {RECALL_SLO:.2f} "
            f"({'PASS' if posted.recall >= RECALL_SLO else 'BELOW'})"
        )
        assert posted.precision >= PRECISION_SLO
        assert posted.recall >= RECALL_SLO
        # The stricter reading has no SLO to meet, but it must not silently
        # collapse: a widening gap between the two is dedup regressing.
        assert posted.precision_strict >= 0.75

    def test_verification_is_what_buys_the_precision(
        self, report: EvalReport
    ) -> None:
        """The claim the whole architecture rests on, as a measured delta.

        The same claims, scored before and after Stage VI against the same
        labels. This number is not a property of the transcript -- both stages
        read the same one -- it is what the gate and the budget did to it.
        """
        print(
            f"[eval] verification: precision {report.emitted.precision:.3f} -> "
            f"{report.posted.precision:.3f} (+{report.precision_gain:.3f}), "
            f"recall cost {report.recall_cost:.3f}, "
            f"{report.emitted.comments - report.posted.comments} comments stopped"
        )
        assert report.precision_gain > 0.15

    def test_no_defect_is_lost_to_verification(self, report: EvalReport) -> None:
        """ARCHITECTURE sec. 11's live risk, measured instead of asserted.

        Precision bought by suppressing true findings is not precision. A
        suppressed true comment is tolerable when a second one about the same
        defect survives; losing the defect entirely is not.
        """
        print(
            f"[eval] {report.suppressed_true_findings} true comment(s) withheld, "
            f"{report.defects_lost_to_verification} defect(s) lost entirely"
        )
        for case in report.cases:
            if case.suppressed_true:
                print(f"    withheld on {case.slug}: {case.suppressed_true[0]}")
        assert report.defects_lost_to_verification == 0
        assert report.recall_cost == 0.0

    def test_clean_pull_requests_stay_quiet(self, report: EvalReport) -> None:
        """The sharpest precision signal in the corpus: on a clean PR the only
        correct output is silence, and there is no partial credit."""
        noisy = [c.slug for c in report.cases if c.is_clean and c.posted_false]
        clean_total = sum(1 for c in report.cases if c.is_clean)
        print(
            f"[eval] {clean_total - len(noisy)}/{clean_total} clean pull requests "
            f"received no comment at all"
        )
        for slug in noisy:
            print(f"    commented on clean PR: {slug}")
        assert len(noisy) <= 2

    def test_drops_are_attributed_per_gate(self, report: EvalReport) -> None:
        """Per-gate counts, so a regression names its own cause."""
        total = sum(report.drops_by_gate.values())
        for gate in sorted(report.drops_by_gate, key=lambda g: -report.drops_by_gate[g]):
            print(f"[eval] {gate.value:18} {report.drops_by_gate[gate]:4d}")
        print(f"[eval] {report.decode_rejects} claim(s) rejected at decode")
        assert total > 0
        # FILE_EXISTS is unreachable from Stage V by construction: `decode_findings`
        # is given the group's path as `allowed_paths`, so a claim about another
        # file is dropped before the gate ever opens one. It is exercised by the
        # M3 gate, which builds findings directly.
        assert VerificationGate.FILE_EXISTS not in report.drops_by_gate
        assert report.decode_rejects > 0

    def test_every_review_actually_saw_retrieved_context(
        self, runs: tuple[CaseRun, ...]
    ) -> None:
        """ADR-002, checked end to end rather than in isolation.

        A pack that came back empty would leave the model reviewing a diff with
        no surrounding code, and the review would still *look* fine here -- the
        transcript is fixed, so the findings would be identical. Retrieval
        failing silently is therefore invisible in every other number on this
        page, which is exactly why it gets its own assertion.
        """
        starved = [run.slug for run in runs if run.retrieved_items == 0]
        print(
            f"[eval] {sum(r.retrieved_items for r in runs)} context items across "
            f"{sum(len(r.groups) for r in runs)} review units"
        )
        assert not starved, f"reviewed with an empty context pack: {starved}"
        assert all(run.groups for run in runs), "a diff produced no review unit"

    def test_nothing_is_deleted_between_the_model_and_the_report(
        self, runs: tuple[CaseRun, ...]
    ) -> None:
        """Every decoded claim ends up either posted or suppressed, never gone.

        The budget and the gate both suppress rather than delete, so the two
        buckets must account for the whole of the model's output. A leak here
        would make the drop rate and the precision gain describe a smaller run
        than the one that happened.
        """
        for run in runs:
            assert len(run.posted) + len(run.suppressed) == len(run.emitted), (
                f"{run.slug}: {len(run.emitted)} emitted but "
                f"{len(run.posted)} posted + {len(run.suppressed)} suppressed"
            )
            assert run.run.findings_posted == len(run.posted)
            assert run.run.findings_suppressed == len(run.suppressed)

    def test_every_case_is_accounted_for(
        self, report: EvalReport, cases: tuple[LabeledPR, ...]
    ) -> None:
        """No case may be silently skipped -- a corpus that quietly shrank would
        move every metric for a reason nobody could see."""
        assert len(report.cases) == len(cases)
        assert {c.slug for c in report.cases} == {c.slug for c in cases}


class TestTheGateCouldFail:
    """Guards on the measurement, in the tradition of the M3 gate."""

    async def test_a_more_speculative_reviewer_scores_visibly_worse(
        self, report: EvalReport, degraded: EvalReport
    ) -> None:
        """M6-full blocks a change that regresses precision by more than 2
        points. That gate is only worth building if the metric moves, so here it
        is moving: one extra confident, unfalsifiable comment per pull request.
        """
        drop = report.posted.precision - degraded.posted.precision
        print(
            f"\n[eval] {report.transcript}: precision "
            f"{report.posted.precision:.3f}, "
            f"{report.posted.wrong} wrong comments"
        )
        print(
            f"[eval] {degraded.transcript}: precision "
            f"{degraded.posted.precision:.3f}, "
            f"{degraded.posted.wrong} wrong comments "
            f"(-{drop:.3f})"
        )
        assert drop > 0.02

    def test_the_drop_rate_alone_would_not_have_noticed(
        self, report: EvalReport, degraded: EvalReport
    ) -> None:
        """Why the labeled corpus had to exist.

        The speculative reviewer's extra comments are matters of taste: real
        file, changed line, real symbols, nothing for a mechanism to check. Every
        one passes verification, so the per-gate drop counts are *identical* to
        the healthy run. Every quality signal Argus had before this milestone
        would have called that regression a clean bill of health.
        """
        assert degraded.drops_by_gate == report.drops_by_gate
        assert degraded.posted.comments > report.posted.comments
        print(
            f"[eval] both runs dropped {sum(report.drops_by_gate.values())} "
            f"findings across identical gates, while posted false positives went "
            f"{report.posted.wrong} -> {degraded.posted.wrong}"
        )

    def test_the_reviewer_misses_defects_and_invents_others(
        self, report: EvalReport
    ) -> None:
        """A transcript that was always right would measure nothing: precision
        would be 1.0 whatever the pipeline did, and recall would be 1.0 whatever
        retrieval found."""
        assert report.emitted.recall < 1.0, "the reviewer finds everything"
        assert report.emitted.wrong > 0, "the reviewer is never wrong"
        assert report.posted.wrong > 0, (
            "no false positive survives the gate, which would mean the corpus "
            "contains only mistakes a mechanism can catch -- real ones are not "
            "all like that"
        )

    def test_the_labels_are_not_a_recording_of_the_output(
        self, report: EvalReport
    ) -> None:
        """Ground truth includes defects the reviewer never mentioned and
        silence where it spoke. Labels recorded from output could not."""
        missed = [c.slug for c in report.cases if not c.is_clean and not c.found]
        assert missed, "every seeded defect was found; the labels look recorded"
        print(f"[eval] defects the reviewer missed: {', '.join(missed)}")


class TestSymbolResolvesAfterTheHarnessFoundIt:
    """The harness's first run found this gate demoting *correct* findings.

    ``SYMBOL_RESOLVES`` checked backticked tokens against the symbol table
    alone, which holds modules, classes, functions and methods -- not
    parameters, locals, fields or module-level constants. A reviewer writing the
    way reviewers write ("slicing to ``limit``...") was charged 0.30 confidence
    for quoting the parameter it was talking about. Four demotions in this
    corpus were of that kind; one left a true finding sitting exactly on its
    severity floor with no margin.

    ``build_vocabulary`` now also harvests identifiers from the context pack --
    the half of ``known_symbols``'s own docstring that was never built. These
    tests hold both ends of the fix: the false demotions are gone, and the
    fabrications are still caught.
    """

    def test_real_identifiers_no_longer_cost_confidence(
        self, runs: tuple[CaseRun, ...]
    ) -> None:
        demoted = [
            (run.slug, f.title, f.confidence)
            for run in runs
            for f in run.report.findings
            if VerificationGate.SYMBOL_RESOLVES in f.verification.gates_failed
        ]
        for slug, title, confidence in demoted:
            print(f"\n[gate] {slug}: {confidence:.2f} after demotion -- {title[:44]}")
        # Only the two claims that name a helper existing nowhere in the tree.
        assert len(demoted) == 2

    def test_invented_symbols_are_still_caught(
        self, runs: tuple[CaseRun, ...]
    ) -> None:
        """The other end of the fix, and the one that matters.

        A vocabulary wide enough to stop punishing real names could easily be
        wide enough to accept anything. These two claims name helpers that
        appear in no symbol table and in no retrieved chunk, and both must still
        be demoted below their floor and dropped.
        """
        invented = {"py-03-put-skips-validation", "ts-01-truncate-off-by-one"}
        for run in runs:
            if run.slug not in invented:
                continue
            fabrications = [
                f
                for f in run.report.findings
                if VerificationGate.SYMBOL_RESOLVES in f.verification.gates_failed
            ]
            assert fabrications, f"{run.slug}: invented symbol went unnoticed"
            assert all(not f.verification.is_postable for f in fabrications)

    def test_the_widened_vocabulary_did_not_swallow_the_gate(
        self, report: EvalReport
    ) -> None:
        """A gate that cannot fail measures nothing. It still fires."""
        assert report.drops_by_gate.get(VerificationGate.SYMBOL_RESOLVES, 0) > 0


class TestKnownLimitations:
    """A weakness the harness surfaced that is *not* fixed here, pinned so it
    cannot rot into a surprise."""

    def test_a_paraphrased_duplicate_escapes_the_dedup_gate(
        self, report: EvalReport
    ) -> None:
        """``NOT_DUPLICATE`` compares title word sets. That separates two
        different bugs with similar titles -- the reason it is not character
        similarity -- but does not catch one bug described twice in different
        words. ``ts-04`` posts both comments; their overlap is 0.44 against a
        0.70 threshold.

        A lexical fix was tried and rejected rather than shipped. The decisive
        pair is "...without being validated" / "...without validation running
        first" (must merge, 0.44) against the documented counterexample
        "Missing null check on user lookup" / "Missing bounds check on index
        lookup" (must not merge, 0.50). Stopword removal plus stemming separates
        them -- 0.667 against 0.429 -- but only with a stemmer special-cased to
        collapse *validated* and *validation*; without that one hand-tuned rule
        both pairs score 0.429 and no threshold exists. That is fitting a
        threshold to a single word pair, not fixing dedup, and the next
        paraphrase would land somewhere else. Separating "same bug, different
        words" from "different bug, similar words" is a semantic problem and
        wants a semantic tool.

        The cost is bounded and measured: one redundant comment, which the SLO's
        own definition counts as correct (the author agrees with it) and which
        ``precision_strict`` charges as noise. The gap between the two numbers
        is the whole of it.
        """
        assert report.posted.redundant == 1
        gap = report.posted.precision - report.posted.precision_strict
        print(
            f"\n[known] one paraphrased duplicate survives dedup: precision "
            f"{report.posted.precision:.3f}, strict "
            f"{report.posted.precision_strict:.3f} (gap {gap:.3f})"
        )
        assert gap > 0.0
