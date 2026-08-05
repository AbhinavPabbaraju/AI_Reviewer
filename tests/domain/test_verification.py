"""The verification gate, one test class per gate.

ARCHITECTURE sec. 4.6 calls this the load-bearing component, and M3's exit
criterion is about it, so each gate is exercised on its own: a gate that only
ever fires as part of a pipeline is a gate nobody has actually checked. Each
class below asserts three things about its gate -- that it fires when it should,
that it does *not* fire when it should not, and that the outcome is the one the
specification names (drop, demote, strip, merge, suppress).

The false-positive half matters as much as the other. A gate that rejects
everything scores a perfect zero on "fabrications posted" while making the
product useless, so every class has at least one test that a legitimate finding
survives.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from app.domain.contracts import (
    Category,
    CodeSpan,
    Evidence,
    EvidenceRole,
    Finding,
    FindingSource,
    Severity,
    VerificationGate,
    VerificationStatus,
)
from app.domain.indexing.models import Language
from app.domain.review.diff import parse_unified_diff
from app.domain.review.verification import (
    VerificationContext,
    VerificationPolicy,
    Verifier,
)
from app.infra.parsing.syntax import TreeSitterSyntaxChecker
from app.infra.review.head_files import MappingHeadFiles

RUN_ID = UUID("11111111-1111-1111-1111-111111111111")

# 20 lines, with a bug on the line the diff below changes.
STORE_PY = """import os


class Store:
    def __init__(self, root):
        self.root = root

    def read(self, name):
        path = os.path.join(self.root, name)
        with open(path) as handle:
            return handle.read()

    def write(self, name, value):
        path = os.path.join(self.root, name)
        with open(path, "w") as handle:
            handle.write(value)

    def clear(self):
        for name in os.listdir(self.root):
            os.remove(os.path.join(self.root, name))
"""

# Changes line 9 (the os.path.join in `read`).
DIFF = """diff --git a/app/store.py b/app/store.py
--- a/app/store.py
+++ b/app/store.py
@@ -6,5 +6,5 @@ class Store:
         self.root = root

     def read(self, name):
-        path = self.root + name
+        path = os.path.join(self.root, name)
         with open(path) as handle:
"""

HEAD_FILES = {"app/store.py": STORE_PY}


def make_finding(
    *,
    path: str = "app/store.py",
    line_start: int = 9,
    line_end: int = 9,
    title: str = "User-controlled name is joined without normalization",
    explanation: str = (
        "The joined path is not normalized, so a traversal segment escapes the "
        "root directory and reads an arbitrary file."
    ),
    severity: Severity = Severity.HIGH,
    confidence: float = 0.9,
    source: FindingSource = FindingSource.LLM,
    improved_code: str | None = None,
    suggested_fix: str | None = None,
    extra_evidence: tuple[Evidence, ...] = (),
) -> Finding:
    location = CodeSpan(path=path, line_start=line_start, line_end=line_end)
    evidence_span = CodeSpan(path=path, line_start=line_start, line_end=line_end)
    return Finding(
        run_id=RUN_ID,
        severity=severity,
        category=Category.SECURITY,
        source=source,
        title=title,
        explanation=explanation,
        location=location,
        evidence=(
            Evidence(
                span=evidence_span,
                role=EvidenceRole.DEFECT_SITE,
                excerpt="path = os.path.join(self.root, name)",
            ),
            *extra_evidence,
        ),
        confidence=confidence,
        improved_code=improved_code,
        suggested_fix=suggested_fix
        or ("Normalize the path first." if improved_code else None),
        prompt_version="reviewer/v1" if source is FindingSource.LLM else None,
        rule_id=None if source is FindingSource.LLM else "S001",
    )


def make_verifier(
    *, policy: VerificationPolicy | None = None, files: dict[str, str] | None = None
) -> Verifier:
    return Verifier(
        files=MappingHeadFiles(files if files is not None else HEAD_FILES),
        syntax=TreeSitterSyntaxChecker(),
        policy=policy or VerificationPolicy(),
    )


def make_context(known: frozenset[str] = frozenset()) -> VerificationContext:
    return VerificationContext(diff=parse_unified_diff(DIFF), known_symbols=known)


async def verify_one(finding: Finding, **kwargs: object) -> Finding:
    verifier = kwargs.pop("verifier", None) or make_verifier()
    context = kwargs.pop("context", None) or make_context()
    assert isinstance(verifier, Verifier)
    assert isinstance(context, VerificationContext)
    report = await verifier.verify([finding], context)
    [result] = report.findings
    return result


class TestFileExists:
    """Cited path exists at head SHA -> drop."""

    async def test_nonexistent_file_is_dropped(self) -> None:
        result = await verify_one(make_finding(path="app/does_not_exist.py"))
        assert result.verification.status is VerificationStatus.REJECTED
        assert VerificationGate.FILE_EXISTS in result.verification.gates_failed
        assert not result.verification.is_postable

    async def test_existing_file_passes(self) -> None:
        result = await verify_one(make_finding())
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_rejection_short_circuits_later_gates(self) -> None:
        """No point parsing a patch for a file that is not there, and reporting
        four failures for one root cause would distort the per-gate metrics."""
        result = await verify_one(
            make_finding(path="ghost.py", improved_code="!!! not python !!!")
        )
        assert result.verification.gates_failed == (VerificationGate.FILE_EXISTS,)

    async def test_deleted_file_is_dropped(self) -> None:
        """A PR that deletes a file gets findings about it; the file is gone at
        head, so there is nowhere to render them."""
        verifier = make_verifier(files={})
        result = await verify_one(make_finding(), verifier=verifier)
        assert VerificationGate.FILE_EXISTS in result.verification.gates_failed


class TestLineInRange:
    """Span within the file and within changed lines -> drop."""

    async def test_line_past_end_of_file_is_dropped(self) -> None:
        result = await verify_one(make_finding(line_start=900, line_end=901))
        assert result.verification.status is VerificationStatus.REJECTED
        assert VerificationGate.LINE_IN_RANGE in result.verification.gates_failed
        assert result.verification.notes is not None
        assert "has 20 lines" in result.verification.notes

    async def test_line_inside_file_but_outside_the_diff_is_dropped(self) -> None:
        """Line 19 exists but this PR did not touch it. A real bug there is not
        one the author introduced and cannot be actioned in this review."""
        result = await verify_one(make_finding(line_start=19, line_end=19))
        assert result.verification.status is VerificationStatus.REJECTED
        assert VerificationGate.LINE_IN_RANGE in result.verification.gates_failed

    async def test_changed_line_passes(self) -> None:
        result = await verify_one(make_finding(line_start=9, line_end=9))
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_context_line_inside_the_hunk_passes_by_default(self) -> None:
        """Line 10 is unchanged context inside the hunk -- on the reviewer's
        screen, and often where the actual bug is."""
        result = await verify_one(make_finding(line_start=10, line_end=10))
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_strict_policy_rejects_context_lines(self) -> None:
        verifier = make_verifier(
            policy=VerificationPolicy(allow_context_lines=False)
        )
        result = await verify_one(
            make_finding(line_start=10, line_end=10), verifier=verifier
        )
        assert VerificationGate.LINE_IN_RANGE in result.verification.gates_failed

    async def test_span_overlapping_the_diff_passes(self) -> None:
        """A finding about a whole function that contains changed lines is
        legitimate: intersection, not containment, is the test."""
        result = await verify_one(make_finding(line_start=8, line_end=11))
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_untouched_file_needs_evidence_in_a_changed_file(self) -> None:
        files = {**HEAD_FILES, "app/other.py": "x = 1\ny = 2\n"}
        verifier = make_verifier(files=files)

        orphan = make_finding(path="app/other.py", line_start=1, line_end=1)
        result = await verify_one(orphan, verifier=verifier)
        assert VerificationGate.LINE_IN_RANGE in result.verification.gates_failed

        # The same location, but now the finding ties itself back to the change:
        # "your new caller breaks this" is a real review comment.
        connected = make_finding(
            path="app/other.py",
            line_start=1,
            line_end=1,
            extra_evidence=(
                Evidence(
                    span=CodeSpan(path="app/store.py", line_start=9, line_end=9),
                    role=EvidenceRole.CALLER,
                    excerpt="path = os.path.join(self.root, name)",
                ),
            ),
        )
        result = await verify_one(connected, verifier=verifier)
        assert result.verification.status is VerificationStatus.VERIFIED


class TestSymbolResolves:
    """Symbols named in the explanation exist -> demote confidence."""

    KNOWN = frozenset({"app.store.Store", "Store", "read", "write", "clear"})

    async def test_invented_symbol_demotes(self) -> None:
        finding = make_finding(
            explanation=(
                "The `Store.sanitize_path` helper is never called here, so the "
                "traversal check is skipped entirely on this path."
            )
        )
        result = await verify_one(finding, context=make_context(self.KNOWN))
        assert result.verification.status is VerificationStatus.DEMOTED
        assert VerificationGate.SYMBOL_RESOLVES in result.verification.gates_failed
        assert result.confidence == pytest.approx(0.6)

    async def test_real_symbol_does_not_demote(self) -> None:
        finding = make_finding(
            explanation=(
                "The `Store.read` method joins an unnormalized name, so a "
                "traversal segment escapes the configured root directory."
            )
        )
        result = await verify_one(finding, context=make_context(self.KNOWN))
        assert result.verification.status is VerificationStatus.VERIFIED
        assert result.confidence == pytest.approx(0.9)

    async def test_language_builtins_are_not_treated_as_repo_symbols(self) -> None:
        """Demoting a finding for saying `open()` or `None` would punish exactly
        the explanations that are most concrete."""
        finding = make_finding(
            explanation=(
                "Passing the joined path to `open()` returns `None` guards "
                "nothing, and a `ValueError` here would escape uncaught."
            )
        )
        result = await verify_one(finding, context=make_context(self.KNOWN))
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_prose_in_backticks_is_not_a_symbol_claim(self) -> None:
        finding = make_finding(
            explanation=(
                "The path is built with `self.root + name` which is not "
                "normalized, so traversal is possible from any caller."
            )
        )
        result = await verify_one(finding, context=make_context(self.KNOWN))
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_empty_vocabulary_disables_the_gate(self) -> None:
        """A gate with nothing to check against must not manufacture verdicts."""
        finding = make_finding(explanation="The `Totally.Invented.Thing` is wrong here.")
        result = await verify_one(finding, context=make_context(frozenset()))
        assert result.verification.status is VerificationStatus.VERIFIED


class TestPatchParses:
    """improved_code parses under the file's grammar -> strip patch, keep prose."""

    async def test_unparseable_patch_is_stripped_and_prose_kept(self) -> None:
        finding = make_finding(
            improved_code="path = os.path.join(self.root,,, name",
            suggested_fix="Normalize the resolved path before opening it.",
        )
        result = await verify_one(finding)
        assert result.improved_code is None
        assert result.suggested_fix == "Normalize the resolved path before opening it."
        assert result.verification.status is VerificationStatus.DEMOTED
        assert VerificationGate.PATCH_PARSES in result.verification.gates_failed
        # Demoted, not rejected: the prose is still worth posting.
        assert result.verification.is_postable

    async def test_valid_patch_survives(self) -> None:
        finding = make_finding(
            improved_code='path = os.path.realpath(os.path.join(self.root, name))',
            suggested_fix="Resolve the path and confirm it stays under the root.",
        )
        result = await verify_one(finding)
        assert result.improved_code is not None
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_indented_fragment_is_accepted(self) -> None:
        """A suggestion lifted from a method body arrives indented, and in Python
        indentation is syntax. Retrying dedented is what keeps this gate from
        stripping most real patches."""
        finding = make_finding(
            improved_code="        path = os.path.realpath(self.root)",
            suggested_fix="Resolve the root.",
        )
        result = await verify_one(finding)
        assert result.improved_code is not None
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_empty_patch_is_stripped(self) -> None:
        finding = make_finding(improved_code="   ", suggested_fix="Remove it.")
        result = await verify_one(finding)
        assert result.improved_code is None
        assert VerificationGate.PATCH_PARSES in result.verification.gates_failed

    async def test_typescript_patch_is_checked_with_its_own_grammar(self) -> None:
        files = {"src/store.ts": "export const x = 1;\nexport const y = 2;\n"}
        diff = parse_unified_diff(
            """diff --git a/src/store.ts b/src/store.ts
--- a/src/store.ts
+++ b/src/store.ts
@@ -1 +1,2 @@
 export const x = 1;
+export const y = 2;
"""
        )
        verifier = make_verifier(files=files)
        context = VerificationContext(diff=diff, known_symbols=frozenset())

        broken = make_finding(
            path="src/store.ts",
            line_start=2,
            line_end=2,
            improved_code="export const y = {{{;",
            suggested_fix="Fix it.",
        )
        report = await verifier.verify([broken], context)
        assert report.findings[0].improved_code is None

        ok = make_finding(
            path="src/store.ts",
            line_start=2,
            line_end=2,
            improved_code="export const y: number = 2;",
            suggested_fix="Annotate it.",
        )
        report = await verifier.verify([ok], context)
        assert report.findings[0].improved_code is not None

    async def test_unknown_language_is_not_judged(self) -> None:
        """The checker has no grammar for YAML, so it must have no opinion --
        stripping every patch on a config file would be asserting what it cannot
        know."""
        files = {"deploy/values.yaml": "a: 1\nb: 2\n"}
        diff = parse_unified_diff(
            """diff --git a/deploy/values.yaml b/deploy/values.yaml
--- a/deploy/values.yaml
+++ b/deploy/values.yaml
@@ -1 +1,2 @@
 a: 1
+b: 2
"""
        )
        verifier = make_verifier(files=files)
        finding = make_finding(
            path="deploy/values.yaml",
            line_start=2,
            line_end=2,
            improved_code="b: 3",
            suggested_fix="Bump it.",
        )
        report = await verifier.verify(
            [finding], VerificationContext(diff=diff, known_symbols=frozenset())
        )
        assert report.findings[0].improved_code == "b: 3"


class TestPatchApplies:
    """Suggestion applies as a GitHub suggested-change block -> plain comment."""

    async def test_patch_outside_the_diff_is_downgraded(self) -> None:
        """GitHub only accepts a suggestion on lines in the diff. Lines 8-11 are
        in the hunk; a span reaching line 14 is not, so the patch cannot be one
        click and degrades to prose."""
        finding = make_finding(
            line_start=9,
            line_end=14,
            improved_code="path = os.path.realpath(self.root)",
            suggested_fix="Resolve the root.",
        )
        result = await verify_one(finding)
        assert result.improved_code is None
        assert VerificationGate.PATCH_APPLIES in result.verification.gates_failed
        assert result.verification.is_postable

    async def test_patch_inside_the_diff_applies(self) -> None:
        finding = make_finding(
            line_start=9,
            line_end=9,
            improved_code="path = os.path.realpath(self.root)",
            suggested_fix="Resolve the root.",
        )
        result = await verify_one(finding)
        assert result.improved_code is not None
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_a_stripped_patch_is_not_judged_again(self) -> None:
        """PATCH_PARSES already removed the patch, so PATCH_APPLIES has nothing
        to say and must not add a second failure for one root cause."""
        finding = make_finding(
            line_start=9,
            line_end=14,
            improved_code="not,,, python",
            suggested_fix="Fix it.",
        )
        result = await verify_one(finding)
        assert result.verification.gates_failed == (VerificationGate.PATCH_PARSES,)


class TestNotDuplicate:
    """Near-duplicate of another finding -> merge."""

    async def test_overlapping_similar_findings_are_merged(self) -> None:
        first = make_finding(
            title="Unnormalized path join allows directory traversal",
            confidence=0.9,
        )
        second = make_finding(
            title="Unnormalized path join allows traversal",
            confidence=0.7,
        )
        report = await make_verifier().verify([first, second], make_context())

        assert len(report.postable) == 1
        [survivor] = report.postable
        assert survivor.confidence == pytest.approx(0.9)
        [merged] = report.rejected
        assert VerificationGate.NOT_DUPLICATE in merged.verification.gates_failed
        # Both carry the same group id, so the dashboard can show what was folded in.
        assert merged.dedup_group == survivor.dedup_group is not None

    async def test_different_bugs_in_the_same_span_are_both_kept(self) -> None:
        """Span overlap alone must not merge: two real bugs often share a line."""
        traversal = make_finding(
            title="Unnormalized path join allows directory traversal"
        )
        handle = make_finding(
            title="File handle is not closed when read raises midway"
        )
        report = await make_verifier().verify([traversal, handle], make_context())
        assert len(report.postable) == 2

    async def test_titles_that_only_look_alike_are_both_kept(self) -> None:
        """The case that chose the similarity metric.

        "Missing null check on user lookup" and "Missing bounds check on index
        lookup" are two different bugs, but they score 0.78 by character-level
        similarity -- *higher* than genuine duplicates -- because the words that
        differ share most of their letters. Word-set overlap scores them 0.50 and
        keeps both. If this test starts failing, the metric has regressed to
        comparing characters.
        """
        report = await make_verifier().verify(
            [
                make_finding(title="Missing null check on user lookup"),
                make_finding(title="Missing bounds check on index lookup"),
            ],
            make_context(),
        )
        assert len(report.postable) == 2

    async def test_similar_titles_in_different_files_are_both_kept(self) -> None:
        files = {**HEAD_FILES, "app/other.py": "x = 1\n"}
        diff = parse_unified_diff(
            DIFF
            + """diff --git a/app/other.py b/app/other.py
--- /dev/null
+++ b/app/other.py
@@ -0,0 +1 @@
+x = 1
"""
        )
        report = await make_verifier(files=files).verify(
            [
                make_finding(title="Unnormalized path join allows traversal"),
                make_finding(
                    path="app/other.py",
                    line_start=1,
                    line_end=1,
                    title="Unnormalized path join allows traversal",
                ),
            ],
            VerificationContext(diff=diff, known_symbols=frozenset()),
        )
        assert len(report.postable) == 2

    async def test_deterministic_source_wins_an_exact_tie(self) -> None:
        """An analyzer cannot have hallucinated its location, so on equal
        priority it is the one to keep."""
        llm = make_finding(
            title="Unnormalized path join allows directory traversal",
            source=FindingSource.LLM,
            confidence=0.8,
        )
        analyzer = make_finding(
            title="Unnormalized path join allows directory traversal",
            source=FindingSource.SEMGREP,
            confidence=0.8,
        )
        report = await make_verifier().verify([llm, analyzer], make_context())
        [survivor] = report.postable
        assert survivor.source is FindingSource.SEMGREP


class TestConfidenceFloor:
    """Confidence below the per-severity floor -> suppress, retain in DB."""

    async def test_below_floor_is_suppressed_but_retained(self) -> None:
        result = await verify_one(
            make_finding(severity=Severity.LOW, confidence=0.5)
        )
        assert result.verification.status is VerificationStatus.REJECTED
        assert VerificationGate.CONFIDENCE_FLOOR in result.verification.gates_failed
        assert not result.verification.is_postable

    async def test_above_floor_passes(self) -> None:
        result = await verify_one(
            make_finding(severity=Severity.LOW, confidence=0.8)
        )
        assert result.verification.status is VerificationStatus.VERIFIED

    async def test_floor_is_per_severity(self) -> None:
        """0.55 clears the critical floor and misses the low one: the cost of
        missing a critical bug is asymmetric."""
        critical = await verify_one(
            make_finding(severity=Severity.CRITICAL, confidence=0.55)
        )
        low = await verify_one(make_finding(severity=Severity.LOW, confidence=0.55))
        assert critical.verification.status is VerificationStatus.VERIFIED
        assert low.verification.status is VerificationStatus.REJECTED

    async def test_floor_is_applied_after_demotions(self) -> None:
        """Ordering that matters: a finding at 0.75 which loses 0.3 to a soft
        gate is at 0.45 and must be measured there, not at its original score."""
        finding = make_finding(
            severity=Severity.HIGH,
            confidence=0.75,
            explanation="The `Store.invented_helper` is never called on this path at all.",
        )
        result = await verify_one(
            finding, context=make_context(TestSymbolResolves.KNOWN)
        )
        assert result.confidence == pytest.approx(0.45)
        assert result.verification.status is VerificationStatus.REJECTED
        assert result.verification.gates_failed == (
            VerificationGate.SYMBOL_RESOLVES,
            VerificationGate.CONFIDENCE_FLOOR,
        )


class TestReport:
    async def test_drop_rate_and_per_gate_counts(self) -> None:
        """sec. 4.6: every drop is counted per gate, because a bad prompt change
        shows up as a shift in one gate rather than a vague quality dip."""
        findings = [
            make_finding(),
            make_finding(path="ghost.py"),
            make_finding(path="also_missing.py"),
            make_finding(line_start=900, line_end=900),
        ]
        report = await make_verifier().verify(findings, make_context())

        assert len(report.findings) == 4
        assert len(report.postable) == 1
        assert len(report.rejected) == 3
        assert report.drop_rate == pytest.approx(0.75)
        assert report.drops_by_gate[VerificationGate.FILE_EXISTS] == 2
        assert report.drops_by_gate[VerificationGate.LINE_IN_RANGE] == 1

    async def test_empty_input_is_not_a_division_by_zero(self) -> None:
        report = await make_verifier().verify([], make_context())
        assert report.findings == ()
        assert report.drop_rate == 0.0

    async def test_rejected_findings_are_retained_not_discarded(self) -> None:
        """Suppressed findings are hidden from the PR, not thrown away -- the
        dashboard shows them and the metrics count them."""
        report = await make_verifier().verify(
            [make_finding(path="ghost.py")], make_context()
        )
        assert len(report.findings) == 1
        assert report.postable == ()

    async def test_findings_are_never_mutated_in_place(self) -> None:
        """Findings are immutable and the audit trail depends on it."""
        original = make_finding(path="ghost.py")
        report = await make_verifier().verify([original], make_context())
        assert original.verification.status is VerificationStatus.PENDING
        assert report.findings[0].verification.status is VerificationStatus.REJECTED
        assert report.findings[0].id == original.id


class TestSyntaxChecker:
    """The infra checker on its own -- tree-sitter is error-tolerant, so
    'did we get a tree' would accept everything."""

    def test_valid_python(self) -> None:
        assert TreeSitterSyntaxChecker().parses(Language.PYTHON, "x = 1\n")

    def test_invalid_python(self) -> None:
        assert not TreeSitterSyntaxChecker().parses(Language.PYTHON, "def f(:\n")

    def test_nested_error_is_detected(self) -> None:
        assert not TreeSitterSyntaxChecker().parses(
            Language.PYTHON, "def f():\n    return 1 +\n"
        )

    def test_valid_typescript(self) -> None:
        assert TreeSitterSyntaxChecker().parses(
            Language.TYPESCRIPT, "const x: number = 1;"
        )

    def test_jsx_parses_through_the_tsx_dialect(self) -> None:
        assert TreeSitterSyntaxChecker().parses(
            Language.TYPESCRIPT, "const el = <div>hi</div>;"
        )

    def test_blank_source_does_not_parse(self) -> None:
        assert not TreeSitterSyntaxChecker().parses(Language.PYTHON, "  \n ")
