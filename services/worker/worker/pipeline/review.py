"""Stage III/V/VI orchestrator: diff -> groups -> context -> review -> gate -> budget.

The sibling of ``indexer.py``, and the same shape: it owns no logic of its own,
sequences the domain algorithms behind ports, and reports what happened. Where
the indexer turns a commit into a symbol graph, this turns a diff into postable
comments.

It exists as a component rather than as glue inside a caller because it has
three consumers with nothing else in common: the CLI drives it against a local
checkout and a local model, the M6 eval harness drives it against fakes and a
recorded transcript, and the M5 GitHub App will drive it against a real
installation. Any sequencing that lives in one of those is sequencing the other
two are not testing.

**Two phases, deliberately.** ``prepare`` does everything up to the first token
of model output -- parse the diff, group by symbol, retrieve context -- and
``execute`` spends the model. They are separate because the interesting things a
caller wants to do between them all need the plan: show the user what is about
to be reviewed, record a cassette against the exact prompts, refuse a diff too
large to afford. ``run`` is the two of them for callers that want neither.

**Verification is not optional and there is no flag for it.** A caller that
could skip the gate would be a caller that can post fabrications, and the entire
precision argument rests on that being impossible rather than discouraged.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from uuid import UUID

from app.domain.contracts import Finding, VerificationGate
from app.domain.ports import EmbeddingPort, LLMPort, VectorStorePort
from app.domain.retrieval.ports import SymbolIndexPort
from app.domain.retrieval.retriever import ContextRetriever, RetrievalConfig
from app.domain.review.budget import DEFAULT_BUDGET, apply_findings_budget
from app.domain.review.diff import ParsedDiff, parse_unified_diff
from app.domain.review.grouping import HunkGroup, group_hunks
from app.domain.review.ports import HeadFilePort, SyntaxCheckerPort
from app.domain.review.reviewer import (
    Reviewer,
    ReviewerConfig,
    ReviewRequest,
)
from app.domain.review.verification import (
    DEFAULT_POLICY,
    VerificationContext,
    VerificationPolicy,
    VerificationReport,
    Verifier,
)
from app.domain.review.vocabulary import build_vocabulary

__all__ = ["ReviewOutcome", "ReviewPipeline", "ReviewPlan"]


@dataclass(frozen=True, slots=True)
class ReviewPlan:
    """Everything settled before the model is called.

    Held rather than recomputed because the prompts in ``requests`` are the
    cassette keys the eval harness records against: rebuilding them would risk
    recording one prompt and sending another, which is the one way a replayed
    run can silently stop testing what it claims to.
    """

    diff: ParsedDiff
    groups: tuple[HunkGroup, ...]
    requests: tuple[ReviewRequest, ...]
    retrieval_ms: int

    @property
    def is_empty(self) -> bool:
        return not self.requests

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.diff.paths))

    @property
    def retrieved_items(self) -> int:
        return sum(len(request.pack.items) for request in self.requests)


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """One review, at every stage a caller might need to report on."""

    plan: ReviewPlan
    emitted: tuple[Finding, ...]
    """Decoded model output, before verification. Kept because the gate's
    contribution is only measurable against what it was given."""

    decode_rejects: tuple[str, ...]
    report: VerificationReport
    posted: tuple[Finding, ...]
    suppressed: tuple[Finding, ...]
    failures: tuple[str, ...]
    """Groups whose provider call raised. One failure loses a group, not the
    review, so this is reported rather than thrown."""

    groups_reviewed: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    models: tuple[str, ...]
    duration_ms: int

    @property
    def completeness(self) -> float:
        """Fraction of review units that produced an answer. A review that
        silently covered half the diff is worse than one that says so."""
        total = len(self.plan.requests)
        return self.groups_reviewed / total if total else 1.0

    @property
    def drops_by_gate(self) -> dict[VerificationGate, int]:
        return dict(self.report.drops_by_gate)


class ReviewPipeline:
    """Reviews one diff against one indexed snapshot. Reusable across runs."""

    def __init__(
        self,
        *,
        index: SymbolIndexPort,
        files: HeadFilePort,
        vectors: VectorStorePort | None = None,
        embeddings: EmbeddingPort | None = None,
        syntax: SyntaxCheckerPort | None = None,
        known_symbols: Iterable[str] = (),
        retrieval: RetrievalConfig | None = None,
        reviewer: ReviewerConfig | None = None,
        policy: VerificationPolicy = DEFAULT_POLICY,
        findings_limit: int = DEFAULT_BUDGET,
    ) -> None:
        self._retriever = ContextRetriever(
            index=index,
            vectors=vectors,
            embeddings=embeddings,
            config=retrieval,
        )
        self._index = index
        self._files = files
        self._syntax = syntax
        self._symbols = tuple(known_symbols)
        self._reviewer_config = reviewer
        self._policy = policy
        self._findings_limit = findings_limit

    async def prepare(self, *, repository_id: str, diff_text: str) -> ReviewPlan:
        """Parse, group and retrieve. Costs nothing but local computation."""
        started = time.perf_counter()
        diff = parse_unified_diff(diff_text)
        groups = await group_hunks(diff, self._index, repository_id)

        requests: list[ReviewRequest] = []
        for group in groups:
            pack = await self._retriever.retrieve(
                repository_id=repository_id, hunks=[group.span]
            )
            requests.append(
                ReviewRequest(group=group, pack=pack, diff_text=diff_text)
            )
        return ReviewPlan(
            diff=diff,
            groups=groups,
            requests=tuple(requests),
            retrieval_ms=round((time.perf_counter() - started) * 1000),
        )

    async def execute(
        self, plan: ReviewPlan, *, llm: LLMPort, run_id: UUID
    ) -> ReviewOutcome:
        """Spend the model, then check every word of it against the tree."""
        started = time.perf_counter()
        result = await Reviewer(llm=llm, config=self._reviewer_config).review(
            plan.requests, run_id=run_id
        )

        verifier = Verifier(
            files=self._files, syntax=self._syntax, policy=self._policy
        )
        report = await verifier.verify(
            result.findings,
            VerificationContext(
                diff=plan.diff,
                # The symbol table *and* the packs the model was handed: a name
                # it read in its own context is not a name it invented.
                known_symbols=build_vocabulary(
                    symbols=self._symbols,
                    packs=[request.pack for request in plan.requests],
                ),
            ),
        )
        # The whole report, not just the postable half. The budget ranks only
        # what survived the gate but passes the rest through as suppressed, and
        # a caller handed `postable` would lose every casualty from the record.
        budgeted = apply_findings_budget(report.findings, limit=self._findings_limit)

        return ReviewOutcome(
            plan=plan,
            emitted=result.findings,
            decode_rejects=result.rejects,
            report=report,
            posted=budgeted.posted,
            suppressed=budgeted.suppressed,
            failures=result.failures,
            groups_reviewed=result.groups_reviewed,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            models=result.models,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )

    async def run(
        self,
        *,
        repository_id: str,
        diff_text: str,
        llm: LLMPort,
        run_id: UUID,
    ) -> ReviewOutcome:
        plan = await self.prepare(repository_id=repository_id, diff_text=diff_text)
        return await self.execute(plan, llm=llm, run_id=run_id)


def postable_by_severity(findings: Sequence[Finding]) -> dict[str, int]:
    """Counts per severity, for a summary line. Ordered by the enum, not by
    count, so the shape of a run is comparable at a glance across runs."""
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
    return counts
