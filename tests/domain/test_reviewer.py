"""Stage V orchestration and the per-PR findings budget.

The reviewer's interesting behaviour is all in the failure paths: one provider
error must not lose the rest of the review, and a partial review must say it is
partial rather than looking like a clean one.
"""

from __future__ import annotations

import asyncio
import json
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
from app.domain.ports import LLMResponse
from app.domain.retrieval.models import ContextPack, RetrievalStats
from app.domain.review.budget import apply_findings_budget
from app.domain.review.diff import Hunk
from app.domain.review.grouping import HunkGroup
from app.domain.review.reviewer import (
    Reviewer,
    ReviewerConfig,
    ReviewRequest,
)

RUN_ID = UUID("22222222-2222-2222-2222-222222222222")


def make_group(path: str = "app/store.py", line: int = 9) -> HunkGroup:
    return HunkGroup(
        path=path,
        symbol_fqn=f"{path.replace('/', '.').removesuffix('.py')}.fn",
        span=CodeSpan(path=path, line_start=line - 1, line_end=line + 2),
        hunks=(
            Hunk(
                old_start=line,
                old_count=1,
                new_start=line,
                new_count=1,
                added_lines=(line,),
            ),
        ),
    )


def make_pack() -> ContextPack:
    return ContextPack(
        repository_id="repo",
        hunks=(),
        items=(),
        stats=RetrievalStats(
            anchors=0,
            graph_candidates=0,
            semantic_candidates=0,
            dropped_by_budget=0,
            tokens_used=0,
            token_budget=100,
            duration_ms=1,
        ),
    )


def make_request(path: str = "app/store.py", line: int = 9) -> ReviewRequest:
    return ReviewRequest(
        group=make_group(path, line), pack=make_pack(), diff_text="+ changed"
    )


def response_for(path: str, line: int) -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "severity": "high",
                    "category": "correctness",
                    "title": "Return value is ignored on the error path",
                    "explanation": "The result is discarded, so the failure is "
                    "silently swallowed by the caller.",
                    "path": path,
                    "line_start": line,
                    "line_end": line,
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "path": path,
                            "line_start": line,
                            "line_end": line,
                            "role": "defect_site",
                            "excerpt": "store.read(name)",
                        }
                    ],
                }
            ]
        }
    )


class ScriptedLLM:
    """An ``LLMPort`` that replies per-call from a script, or raises."""

    def __init__(self, replies: list[str | Exception]) -> None:
        self._replies = list(replies)
        self.calls = 0
        self.max_concurrent = 0
        self._active = 0
        self.last_schema: dict[str, object] | None = None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        self.calls += 1
        self.last_schema = json_schema
        try:
            await asyncio.sleep(0)  # yield, so overlap is observable
            reply = self._replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return LLMResponse(
                content=reply,
                tokens_in=100,
                tokens_out=50,
                cost_usd=0.0,
                model="scripted",
            )
        finally:
            self._active -= 1


class TestReviewer:
    async def test_one_call_per_group(self) -> None:
        llm = ScriptedLLM([response_for("a.py", 9), response_for("b.py", 9)])
        reviewer = Reviewer(llm=llm)
        result = await reviewer.review(
            [make_request("a.py"), make_request("b.py")], run_id=RUN_ID
        )
        assert llm.calls == 2
        assert len(result.findings) == 2
        assert result.groups_reviewed == 2
        assert result.completeness == 1.0

    async def test_no_requests_makes_no_calls(self) -> None:
        llm = ScriptedLLM([])
        result = await Reviewer(llm=llm).review([], run_id=RUN_ID)
        assert llm.calls == 0
        assert result.findings == ()
        assert result.completeness == 1.0

    async def test_concurrency_is_bounded(self) -> None:
        """A 40-file PR firing 40 simultaneous calls is how a review trips a
        rate limit and fails wholesale."""
        llm = ScriptedLLM([response_for(f"f{n}.py", 9) for n in range(8)])
        reviewer = Reviewer(llm=llm, config=ReviewerConfig(concurrency=2))
        await reviewer.review(
            [make_request(f"f{n}.py") for n in range(8)], run_id=RUN_ID
        )
        assert llm.max_concurrent <= 2

    async def test_a_provider_failure_does_not_lose_the_other_groups(self) -> None:
        llm = ScriptedLLM(
            [
                response_for("a.py", 9),
                RuntimeError("connection reset"),
                response_for("c.py", 9),
            ]
        )
        reviewer = Reviewer(llm=llm, config=ReviewerConfig(concurrency=1))
        result = await reviewer.review(
            [make_request("a.py"), make_request("b.py"), make_request("c.py")],
            run_id=RUN_ID,
        )
        assert len(result.findings) == 2
        assert len(result.failures) == 1
        assert "connection reset" in result.failures[0]

    async def test_a_partial_review_reports_itself_as_partial(self) -> None:
        """A review that silently covered two thirds of the diff is worse than
        one that says so."""
        llm = ScriptedLLM(
            [response_for("a.py", 9), RuntimeError("boom"), RuntimeError("boom")]
        )
        reviewer = Reviewer(llm=llm, config=ReviewerConfig(concurrency=1))
        result = await reviewer.review(
            [make_request("a.py"), make_request("b.py"), make_request("c.py")],
            run_id=RUN_ID,
        )
        assert result.groups_reviewed == 1
        assert result.groups_total == 3
        assert result.completeness == pytest.approx(1 / 3)

    async def test_undecodable_response_is_a_reject_not_a_failure(self) -> None:
        """The call succeeded and the response did not decode -- a different
        cause and a different fix from a transport error."""
        llm = ScriptedLLM(["I think the code looks fine to me!"])
        result = await Reviewer(llm=llm).review([make_request()], run_id=RUN_ID)
        assert result.findings == ()
        assert result.failures == ()
        assert len(result.rejects) == 1
        assert result.groups_reviewed == 1

    async def test_findings_about_other_files_are_dropped(self) -> None:
        """A finding about a file this group was not reviewing cannot be
        anchored, so it is contained here rather than carried to the gate."""
        llm = ScriptedLLM([response_for("somewhere/else.py", 9)])
        result = await Reviewer(llm=llm).review(
            [make_request("app/store.py")], run_id=RUN_ID
        )
        assert result.findings == ()
        assert "not among the files under review" in result.rejects[0]

    async def test_usage_is_summed_across_groups(self) -> None:
        llm = ScriptedLLM([response_for("a.py", 9), response_for("b.py", 9)])
        result = await Reviewer(llm=llm).review(
            [make_request("a.py"), make_request("b.py")], run_id=RUN_ID
        )
        assert result.tokens_in == 200
        assert result.tokens_out == 100
        assert result.cost_usd == 0.0
        assert result.models == ("scripted",)

    async def test_schema_is_passed_for_constrained_decoding(self) -> None:
        llm = ScriptedLLM([response_for("a.py", 9)])
        await Reviewer(llm=llm).review([make_request("a.py")], run_id=RUN_ID)
        assert llm.last_schema is not None
        assert "findings" in llm.last_schema["properties"]  # type: ignore[index]

    async def test_schema_can_be_disabled_for_providers_without_it(self) -> None:
        llm = ScriptedLLM([response_for("a.py", 9)])
        reviewer = Reviewer(llm=llm, config=ReviewerConfig(constrain_json=False))
        await reviewer.review([make_request("a.py")], run_id=RUN_ID)
        assert llm.last_schema is None

    def test_config_rejects_nonsense(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            ReviewerConfig(concurrency=0)
        with pytest.raises(ValueError, match="max_tokens"):
            ReviewerConfig(max_tokens=0)


# -- findings budget -------------------------------------------------------- #


def finding(
    *,
    severity: Severity = Severity.HIGH,
    confidence: float = 0.9,
    title: str = "Return value is ignored on the error path",
    rejected: bool = False,
) -> Finding:
    span = CodeSpan(path="app/store.py", line_start=9, line_end=9)
    built = Finding(
        run_id=RUN_ID,
        severity=severity,
        category=Category.CORRECTNESS,
        source=FindingSource.LLM,
        title=title,
        explanation="The result is discarded, so the failure is silently lost.",
        location=span,
        evidence=(
            Evidence(span=span, role=EvidenceRole.DEFECT_SITE, excerpt="store.read()"),
        ),
        confidence=confidence,
        prompt_version="reviewer/v1",
    )
    if rejected:
        return built.reject(VerificationGate.FILE_EXISTS, "gone")
    return built.model_copy(
        update={
            "verification": built.verification.model_copy(
                update={"status": VerificationStatus.VERIFIED}
            )
        }
    )


class TestFindingsBudget:
    def test_keeps_the_highest_priority_findings(self) -> None:
        findings = [
            finding(severity=Severity.LOW, confidence=0.9, title="Low severity bug"),
            finding(severity=Severity.CRITICAL, confidence=0.9, title="Critical bug"),
            finding(severity=Severity.MEDIUM, confidence=0.9, title="Medium bug here"),
        ]
        result = apply_findings_budget(findings, limit=1)
        assert [f.severity for f in result.posted] == [Severity.CRITICAL]

    def test_suppressed_findings_are_retained_not_deleted(self) -> None:
        """Hidden from the pull request, not thrown away: the dashboard shows
        them and the metrics count them."""
        findings = [finding(title=f"Distinct defect number {n} here") for n in range(5)]
        result = apply_findings_budget(findings, limit=2)
        assert len(result.posted) == 2
        assert len(result.suppressed) == 3
        assert result.total == 5
        for suppressed in result.suppressed:
            assert suppressed.verification.status is VerificationStatus.REJECTED
            assert VerificationGate.CONFIDENCE_FLOOR in (
                suppressed.verification.gates_failed
            )

    def test_confidence_breaks_ties_within_a_severity(self) -> None:
        findings = [
            finding(confidence=0.6, title="Lower confidence defect here"),
            finding(confidence=0.95, title="Higher confidence defect here"),
        ]
        result = apply_findings_budget(findings, limit=1)
        assert result.posted[0].confidence == 0.95

    def test_a_low_confidence_critical_does_not_outrank_a_confident_high(self) -> None:
        """`severity x confidence` as a product, not a sum -- which matches how
        humans triage."""
        shaky_critical = finding(
            severity=Severity.CRITICAL, confidence=0.5, title="Shaky critical claim"
        )
        solid_high = finding(
            severity=Severity.HIGH, confidence=0.95, title="Solid high severity bug"
        )
        result = apply_findings_budget([shaky_critical, solid_high], limit=1)
        assert result.posted[0].severity is Severity.HIGH

    def test_rejected_findings_never_consume_a_slot(self) -> None:
        rejected = finding(severity=Severity.CRITICAL, rejected=True)
        real = finding(severity=Severity.LOW, title="A real low severity bug")
        result = apply_findings_budget([rejected, real], limit=1)
        assert result.posted == (real,)

    def test_ordering_is_deterministic_across_runs(self) -> None:
        """A tie broken by dict ordering would make the same run post different
        comments on different days."""
        findings = [
            finding(title=f"Identically ranked defect {n}") for n in range(6)
        ]
        first = apply_findings_budget(findings, limit=3)
        second = apply_findings_budget(list(reversed(findings)), limit=3)
        assert [f.id for f in first.posted] == [f.id for f in second.posted]

    def test_under_budget_posts_everything(self) -> None:
        findings = [finding(title=f"Distinct defect number {n} here") for n in range(3)]
        result = apply_findings_budget(findings, limit=10)
        assert len(result.posted) == 3
        assert result.suppressed == ()

    def test_zero_budget_suppresses_all(self) -> None:
        result = apply_findings_budget([finding()], limit=0)
        assert result.posted == ()
        assert len(result.suppressed) == 1

    def test_negative_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            apply_findings_budget([], limit=-1)

    def test_empty_input(self) -> None:
        result = apply_findings_budget([], limit=10)
        assert result.posted == ()
        assert result.suppressed == ()
