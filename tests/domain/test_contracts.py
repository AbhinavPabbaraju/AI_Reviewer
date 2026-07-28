"""Contract tests.

These are the guard rails for the invariants described in ARCHITECTURE.md.
Each test names the invariant it protects, because a failing test whose purpose
you have to reverse-engineer gets deleted rather than fixed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.contracts import (
    Category,
    CodeSpan,
    Evidence,
    EvidenceRole,
    Finding,
    FindingSource,
    ReviewRun,
    RunStatus,
    Severity,
    VerificationGate,
    VerificationResult,
    VerificationStatus,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


def make_finding(**overrides: object) -> Finding:
    span = overrides.pop("span", CodeSpan(path="app/files.py", line_start=42, line_end=47))
    assert isinstance(span, CodeSpan)
    base: dict[str, object] = {
        "run_id": uuid4(),
        "severity": Severity.HIGH,
        "category": Category.SECURITY,
        "source": FindingSource.LLM,
        "title": "User-controlled path reaches open() without normalization",
        "explanation": (
            "The `name` parameter flows from the HTTP query string into open() "
            "with no normalization, so `../` segments escape the upload root."
        ),
        "location": span,
        "evidence": (
            Evidence(span=span, role=EvidenceRole.DEFECT_SITE, excerpt="open(root + name)"),
        ),
        "confidence": 0.86,
        "prompt_version": "reviewer/v3",
    }
    base.update(overrides)
    return Finding(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CodeSpan
# --------------------------------------------------------------------------- #


class TestCodeSpan:
    """Cloned repositories are untrusted input; spans are the boundary."""

    @pytest.mark.parametrize(
        "path",
        ["/etc/passwd", "../../secrets.env", "app/../../etc/hosts", "a/../../b"],
    )
    def test_rejects_absolute_and_traversal_paths(self, path: str) -> None:
        with pytest.raises(ValidationError, match="repo-relative"):
            CodeSpan(path=path, line_start=1, line_end=1)

    def test_accepts_dotfiles_and_nested_paths(self) -> None:
        # '..' is forbidden but a leading dot is perfectly normal.
        assert CodeSpan(path=".github/workflows/ci.yml", line_start=1, line_end=1)
        assert CodeSpan(path="src/a.b..c/file.py", line_start=1, line_end=1)

    def test_normalizes_windows_separators(self) -> None:
        assert CodeSpan(path="src\\app\\main.py", line_start=1, line_end=2).path == (
            "src/app/main.py"
        )

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValidationError, match="line_end"):
            CodeSpan(path="a.py", line_start=10, line_end=3)

    def test_rejects_zero_indexed_lines(self) -> None:
        # Line numbers are 1-indexed everywhere: git, GitHub, editors.
        with pytest.raises(ValidationError):
            CodeSpan(path="a.py", line_start=0, line_end=5)

    def test_overlap_requires_same_path(self) -> None:
        a = CodeSpan(path="a.py", line_start=1, line_end=10)
        b = CodeSpan(path="b.py", line_start=1, line_end=10)
        assert not a.overlaps(b)

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [(1, 5, True), (5, 9, True), (11, 20, False), (10, 10, True), (0 + 1, 100, True)],
    )
    def test_overlap_boundaries(self, start: int, end: int, expected: bool) -> None:
        a = CodeSpan(path="a.py", line_start=5, line_end=10)
        b = CodeSpan(path="a.py", line_start=start, line_end=end)
        assert a.overlaps(b) is expected
        assert b.overlaps(a) is expected  # symmetric

    def test_span_is_frozen(self) -> None:
        span = CodeSpan(path="a.py", line_start=1, line_end=2)
        with pytest.raises(ValidationError):
            span.line_start = 5  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Finding — the evidence invariant
# --------------------------------------------------------------------------- #


class TestFindingEvidence:
    """The core anti-hallucination invariant: no finding without a citation."""

    def test_rejects_empty_evidence(self) -> None:
        with pytest.raises(ValidationError):
            make_finding(evidence=())

    def test_rejects_evidence_without_a_defect_site(self) -> None:
        span = CodeSpan(path="app/routes.py", line_start=88, line_end=90)
        with pytest.raises(ValidationError, match="DEFECT_SITE"):
            make_finding(
                evidence=(
                    Evidence(span=span, role=EvidenceRole.CALLER, excerpt="read(name)"),
                )
            )

    def test_rejects_location_disagreeing_with_defect_site(self) -> None:
        """Catches the subtle failure where the model reasons about one span and
        points the comment at another."""
        reasoned_about = CodeSpan(path="app/files.py", line_start=42, line_end=47)
        commented_on = CodeSpan(path="app/files.py", line_start=200, line_end=201)
        with pytest.raises(ValidationError, match="does not overlap"):
            make_finding(
                location=commented_on,
                evidence=(
                    Evidence(
                        span=reasoned_about,
                        role=EvidenceRole.DEFECT_SITE,
                        excerpt="open(root + name)",
                    ),
                ),
            )

    def test_accepts_multi_file_evidence(self) -> None:
        defect = CodeSpan(path="app/files.py", line_start=42, line_end=47)
        caller = CodeSpan(path="app/routes.py", line_start=88, line_end=90)
        finding = make_finding(
            evidence=(
                Evidence(span=defect, role=EvidenceRole.DEFECT_SITE, excerpt="open(p)"),
                Evidence(span=caller, role=EvidenceRole.CALLER, excerpt="files.read(q)"),
            )
        )
        assert len(finding.evidence) == 2


class TestFindingQuality:
    """Guards against the specific failure the brief calls out by name."""

    @pytest.mark.parametrize(
        "title",
        [
            "This code could be improved somewhat",
            "Consider refactoring this function body",
            "Potential issue in the request handler",
        ],
    )
    def test_rejects_vague_titles(self, title: str) -> None:
        with pytest.raises(ValidationError, match="non-specific"):
            make_finding(title=title)

    def test_rejects_patch_without_prose(self) -> None:
        with pytest.raises(ValidationError, match="explained in prose"):
            make_finding(improved_code="path = os.path.normpath(name)")

    def test_accepts_patch_with_prose(self) -> None:
        finding = make_finding(
            suggested_fix="Normalize and confine the path to the upload root.",
            improved_code="path = safe_join(UPLOAD_ROOT, name)",
        )
        assert finding.improved_code is not None


class TestFindingSourceRules:
    def test_analyzer_finding_requires_rule_id(self) -> None:
        with pytest.raises(ValidationError, match="requires a rule_id"):
            make_finding(source=FindingSource.SEMGREP, prompt_version=None)

    def test_llm_finding_requires_prompt_version(self) -> None:
        """Without attribution, a regression cannot be traced to a change."""
        with pytest.raises(ValidationError, match="prompt_version"):
            make_finding(prompt_version=None)

    def test_analyzer_sources_are_deterministic(self) -> None:
        assert FindingSource.SEMGREP.is_deterministic
        assert not FindingSource.LLM.is_deterministic


# --------------------------------------------------------------------------- #
# Ranking, identity, and the verification transitions
# --------------------------------------------------------------------------- #


class TestFindingRanking:
    def test_severity_dominates_confidence(self) -> None:
        """A 0.99-confidence LOW must not outrank a 0.60-confidence CRITICAL."""
        low = make_finding(severity=Severity.LOW, confidence=0.99)
        crit = make_finding(severity=Severity.CRITICAL, confidence=0.60)
        assert crit.priority > low.priority

    def test_confidence_breaks_ties_within_a_band(self) -> None:
        a = make_finding(severity=Severity.HIGH, confidence=0.9)
        b = make_finding(severity=Severity.HIGH, confidence=0.5)
        assert a.priority > b.priority

    def test_info_never_blocks_merge(self) -> None:
        assert not Severity.INFO.blocks_merge
        assert not Severity.LOW.blocks_merge
        assert Severity.CRITICAL.blocks_merge


class TestFingerprint:
    def test_stable_across_prose_and_confidence_changes(self) -> None:
        """A dismissed comment must not come back because a prompt tweak reworded
        the explanation."""
        a = make_finding(confidence=0.9, explanation="A" * 50)
        b = make_finding(confidence=0.4, explanation="B" * 50)
        assert a.fingerprint == b.fingerprint

    def test_differs_across_location(self) -> None:
        other = CodeSpan(path="app/files.py", line_start=900, line_end=901)
        a = make_finding()
        b = make_finding(
            location=other,
            evidence=(
                Evidence(span=other, role=EvidenceRole.DEFECT_SITE, excerpt="x"),
            ),
        )
        assert a.fingerprint != b.fingerprint


class TestVerificationTransitions:
    def test_verified_cannot_carry_failed_gates(self) -> None:
        with pytest.raises(ValidationError, match="cannot have failed gates"):
            VerificationResult(
                status=VerificationStatus.VERIFIED,
                gates_failed=(VerificationGate.FILE_EXISTS,),
            )

    def test_rejected_requires_a_gate(self) -> None:
        with pytest.raises(ValidationError, match="requires >=1 failed gate"):
            VerificationResult(status=VerificationStatus.REJECTED)

    def test_demote_reduces_confidence_and_records_gate(self) -> None:
        original = make_finding(confidence=0.9)
        demoted = original.demote(VerificationGate.SYMBOL_RESOLVES)
        assert demoted.confidence == pytest.approx(0.6)
        assert demoted.verification.status is VerificationStatus.DEMOTED
        assert VerificationGate.SYMBOL_RESOLVES in demoted.verification.gates_failed
        # Immutability: the original is untouched.
        assert original.confidence == pytest.approx(0.9)
        assert original.verification.status is VerificationStatus.PENDING

    def test_demotion_clamps_at_zero(self) -> None:
        assert make_finding(confidence=0.1).demote(
            VerificationGate.SYMBOL_RESOLVES
        ).confidence == pytest.approx(0.0)

    def test_repeated_demotion_does_not_duplicate_gates(self) -> None:
        f = make_finding().demote(VerificationGate.SYMBOL_RESOLVES)
        f = f.demote(VerificationGate.SYMBOL_RESOLVES)
        assert f.verification.gates_failed == (VerificationGate.SYMBOL_RESOLVES,)

    def test_rejected_findings_are_not_postable(self) -> None:
        rejected = make_finding().reject(VerificationGate.FILE_EXISTS, "no such file")
        assert not rejected.verification.is_postable
        assert rejected.verification.notes == "no such file"

    def test_demoted_findings_remain_postable(self) -> None:
        assert make_finding().demote(VerificationGate.PATCH_APPLIES).verification.is_postable

    def test_confidence_is_uncalibrated_by_default(self) -> None:
        """Guards the M6 contract: nothing may treat raw scores as probabilities."""
        assert make_finding().confidence_calibrated is False


# --------------------------------------------------------------------------- #
# ReviewRun
# --------------------------------------------------------------------------- #


def make_run(**overrides: object) -> ReviewRun:
    base: dict[str, object] = {
        "repository_id": uuid4(),
        "pr_number": 42,
        "head_sha": SHA_A,
        "base_sha": SHA_B,
        "ruleset_version": "rules/2026.07",
        "prompt_version": "reviewer/v3",
        "model": "test-model",
    }
    base.update(overrides)
    return ReviewRun(**base)  # type: ignore[arg-type]


class TestReviewRun:
    def test_idempotency_key_is_stable_and_config_sensitive(self) -> None:
        repo = uuid4()
        a = make_run(repository_id=repo)
        b = make_run(repository_id=repo)
        assert a.idempotency_key == b.idempotency_key

        # A prompt change is a different review and must be allowed to re-run.
        c = make_run(repository_id=repo, prompt_version="reviewer/v4")
        assert c.idempotency_key != a.idempotency_key

    def test_rejects_empty_diff(self) -> None:
        with pytest.raises(ValidationError, match="nothing to review"):
            make_run(head_sha=SHA_A, base_sha=SHA_A)

    @pytest.mark.parametrize("sha", ["abc", "A" * 40, "g" * 40, "a" * 41])
    def test_rejects_malformed_sha(self, sha: str) -> None:
        with pytest.raises(ValidationError):
            make_run(head_sha=sha)

    def test_terminal_status_requires_finished_at(self) -> None:
        with pytest.raises(ValidationError, match="requires finished_at"):
            make_run(status=RunStatus.SUCCEEDED)

    def test_failed_run_requires_error(self) -> None:
        with pytest.raises(ValidationError, match="must record an error"):
            make_run(status=RunStatus.FAILED, finished_at=datetime.now(UTC))

    def test_non_failed_run_may_not_carry_error(self) -> None:
        with pytest.raises(ValidationError, match="only failed runs"):
            make_run(
                status=RunStatus.SUCCEEDED,
                finished_at=datetime.now(UTC),
                error="spurious",
            )

    def test_duration_is_none_while_running(self) -> None:
        assert make_run(status=RunStatus.REVIEWING).duration_seconds is None

    def test_duration_computed_on_completion(self) -> None:
        started = datetime.now(UTC)
        run = make_run(
            status=RunStatus.SUCCEEDED,
            started_at=started,
            finished_at=started + timedelta(seconds=87.5),
        )
        assert run.duration_seconds == pytest.approx(87.5)

    def test_status_terminality(self) -> None:
        assert RunStatus.SUCCEEDED.is_terminal
        assert RunStatus.CANCELLED.is_terminal
        assert not RunStatus.REVIEWING.is_terminal


class TestSerializationRoundTrip:
    def test_finding_round_trips_through_json(self) -> None:
        """The API, the queue, and the eval harness all cross a JSON boundary."""
        original = make_finding(
            suggested_fix="Confine the path to the upload root.",
            improved_code="path = safe_join(ROOT, name)",
            references=("https://cwe.mitre.org/data/definitions/22.html",),
        )
        restored = Finding.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_extra_fields_are_rejected(self) -> None:
        """Model output is parsed into these types; silently accepting unknown
        keys would hide schema drift."""
        payload = make_finding().model_dump(mode="json")
        payload["hallucinated_field"] = "surprise"
        with pytest.raises(ValidationError):
            Finding.model_validate(payload)
