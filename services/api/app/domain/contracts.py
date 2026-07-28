"""Core domain contracts for Argus.

This module is the spine of the system: the API serializes these, the worker
produces them, the eval harness scores them, and the frontend's TypeScript
types are generated from them. It is deliberately dependency-free apart from
Pydantic so that it can be imported anywhere, including into the sandboxed
analyzer runners.

Design notes worth keeping in mind while reading:

* ``Finding.evidence`` is required and non-empty. A finding that cannot point at
  code is not a finding. This one constraint kills most fabrications, because
  the model must emit a span that the verifier will independently check against
  the real tree.
* ``Finding.confidence`` is *calibrated*, not self-reported. Until the eval
  harness (M6) fits a calibration map, ``confidence_calibrated`` is False and
  consumers must treat the value as an ordinal, not a probability.
* Findings are immutable. Re-review creates a new ``ReviewRun``; nothing is
  updated in place. This keeps A/B comparison of prompt versions trivial.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

__all__ = [
    "Category",
    "CodeSpan",
    "Evidence",
    "EvidenceRole",
    "Finding",
    "FindingSource",
    "ReviewRun",
    "ReviewScores",
    "RunStatus",
    "Severity",
    "VerificationGate",
    "VerificationResult",
    "VerificationStatus",
]

# Paths are repo-relative POSIX paths. Anything else is a bug or an attack.
_PATH_RE = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[^\x00]+$")

# Rough token estimate; the real tokenizer lives in the LLM adapter. Used only
# to keep prose fields from becoming unbounded.
MAX_EXPLANATION_CHARS = 4000
MAX_EXCERPT_CHARS = 2000


class _Frozen(BaseModel):
    """Base for immutable, strictly-validated value objects."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Severity(StrEnum):
    """Impact if the finding is real.

    Deliberately *not* a blend of impact and confidence -- keeping them
    orthogonal is what makes the ``severity x confidence`` ranking in the
    findings budget meaningful.
    """

    CRITICAL = "critical"  # exploitable now, or data loss
    HIGH = "high"  # incorrect behaviour on a realistic path
    MEDIUM = "medium"  # incorrect on an edge case, or real perf cost
    LOW = "low"  # maintainability, weak test, minor smell
    INFO = "info"  # observation; never blocks

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    @property
    def blocks_merge(self) -> bool:
        return self in (Severity.CRITICAL, Severity.HIGH)


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Category(StrEnum):
    """What kind of problem this is.

    Categories exist for calibration and reporting, not for routing: we do not
    run a separate pipeline stage per category (see ARCHITECTURE.md sec. 4).
    """

    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CONCURRENCY = "concurrency"
    API_CONTRACT = "api_contract"
    ERROR_HANDLING = "error_handling"
    TESTING = "testing"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"


class FindingSource(StrEnum):
    """Which subsystem produced the finding.

    Kept on every finding because precision differs enormously by source, and
    the calibration map in M6 is fit per (source, category) pair.
    """

    LLM = "llm"
    SEMGREP = "semgrep"
    RUFF = "ruff"
    ESLINT = "eslint"
    TSC = "tsc"

    @property
    def is_deterministic(self) -> bool:
        """Deterministic sources cannot hallucinate a location."""
        return self is not FindingSource.LLM


class EvidenceRole(StrEnum):
    """Why this span was cited.

    ``DEFECT_SITE`` is mandatory on every finding; the others explain the
    cross-file reasoning that justified it, and are what the dashboard's
    retrieval inspector renders.
    """

    DEFECT_SITE = "defect_site"
    CALLER = "caller"
    CALLEE = "callee"
    DEFINITION = "definition"
    TEST = "test"
    CONFIG = "config"
    SIMILAR_PATTERN = "similar_pattern"


class VerificationGate(StrEnum):
    """Gates from ARCHITECTURE.md sec. 4.6. Failures are counted per gate so a
    bad prompt change shows up as a shift in a specific gate rather than a
    vague quality dip."""

    FILE_EXISTS = "file_exists"
    LINE_IN_RANGE = "line_in_range"
    SYMBOL_RESOLVES = "symbol_resolves"
    PATCH_PARSES = "patch_parses"
    PATCH_APPLIES = "patch_applies"
    NOT_DUPLICATE = "not_duplicate"
    CONFIDENCE_FLOOR = "confidence_floor"


class VerificationStatus(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    DEMOTED = "demoted"  # kept, but confidence reduced or patch stripped
    REJECTED = "rejected"  # never posted; retained for metrics


class RunStatus(StrEnum):
    QUEUED = "queued"
    INDEXING = "indexing"
    RETRIEVING = "retrieving"
    ANALYZING = "analyzing"
    REVIEWING = "reviewing"
    VERIFYING = "verifying"
    PUBLISHING = "publishing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


class CodeSpan(_Frozen):
    """An inclusive 1-indexed line range in a repo-relative file."""

    path: Annotated[str, Field(min_length=1, max_length=1024)]
    line_start: Annotated[int, Field(ge=1)]
    line_end: Annotated[int, Field(ge=1)]

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        # Absolute paths and traversal are rejected at the type boundary rather
        # than downstream: cloned repositories are untrusted input, and a
        # finding is one of the few objects that flows from model output all
        # the way to a filesystem read.
        if not _PATH_RE.match(v):
            raise ValueError(
                f"path must be a repo-relative POSIX path without '..' segments: {v!r}"
            )
        return v.replace("\\", "/")

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self

    @property
    def line_count(self) -> int:
        return self.line_end - self.line_start + 1

    def overlaps(self, other: CodeSpan) -> bool:
        """Used by the dedup gate to merge findings that point at the same code."""
        if self.path != other.path:
            return False
        return self.line_start <= other.line_end and other.line_start <= self.line_end

    def __str__(self) -> str:
        if self.line_start == self.line_end:
            return f"{self.path}:{self.line_start}"
        return f"{self.path}:{self.line_start}-{self.line_end}"


class Evidence(_Frozen):
    """A cited span that supports a finding.

    ``excerpt`` is the code as it existed at the run's head SHA. It is stored
    rather than re-read so that a finding remains explainable after the branch
    is deleted -- which, for a PR review tool, is the normal case.
    """

    span: CodeSpan
    role: EvidenceRole
    excerpt: Annotated[str, Field(min_length=1, max_length=MAX_EXCERPT_CHARS)]
    symbol_fqn: str | None = Field(
        default=None,
        description="Fully-qualified symbol name, when the span is a symbol body.",
    )


class VerificationResult(_Frozen):
    """Outcome of the verification gate for a single finding."""

    status: VerificationStatus = VerificationStatus.PENDING
    gates_failed: tuple[VerificationGate, ...] = ()
    notes: str | None = None

    @model_validator(mode="after")
    def _status_matches_gates(self) -> Self:
        if self.status is VerificationStatus.VERIFIED and self.gates_failed:
            raise ValueError("a verified finding cannot have failed gates")
        soft_or_hard_fail = (VerificationStatus.DEMOTED, VerificationStatus.REJECTED)
        if self.status in soft_or_hard_fail and not self.gates_failed:
            raise ValueError(f"status={self.status.value} requires >=1 failed gate")
        return self

    @property
    def is_postable(self) -> bool:
        return self.status in (VerificationStatus.VERIFIED, VerificationStatus.DEMOTED)


# --------------------------------------------------------------------------- #
# Finding
# --------------------------------------------------------------------------- #


class Finding(_Frozen):
    """A single reviewable issue.

    Immutable by construction. Mutating operations (demotion, patch stripping)
    return new instances via ``model_copy``, which keeps the audit trail honest.
    """

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID

    severity: Severity
    category: Category
    source: FindingSource

    title: Annotated[str, Field(min_length=8, max_length=160)]
    explanation: Annotated[str, Field(min_length=20, max_length=MAX_EXPLANATION_CHARS)]

    location: CodeSpan
    evidence: Annotated[tuple[Evidence, ...], Field(min_length=1)]

    suggested_fix: str | None = None
    improved_code: str | None = None

    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    confidence_calibrated: bool = Field(
        default=False,
        description="False until an M6 calibration map has been applied. When "
        "False, treat `confidence` as an ordinal ranking signal only.",
    )

    references: tuple[HttpUrl, ...] = ()
    rule_id: str | None = Field(
        default=None, description="Analyzer rule identifier, e.g. 'S105'. LLM: None."
    )

    verification: VerificationResult = VerificationResult()
    dedup_group: str | None = None

    prompt_version: str | None = None
    model: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    # -- validators -------------------------------------------------------- #

    @field_validator("title")
    @classmethod
    def _title_is_specific(cls, v: str) -> str:
        """Reject the failure mode the brief explicitly calls out.

        A title like "This could be improved" is unactionable, and a model that
        emits one has not done the work. Cheap to check here, and it fails the
        run loudly during development rather than quietly shipping noise.
        """
        vague = (
            "could be improved",
            "consider refactoring",
            "might be better",
            "should be reviewed",
            "potential issue",
            "code smell detected",
        )
        lowered = v.lower()
        for phrase in vague:
            if phrase in lowered:
                raise ValueError(
                    f"title is non-specific ({phrase!r}); a finding must name the "
                    "concrete problem"
                )
        return v

    @model_validator(mode="after")
    def _has_defect_site_evidence(self) -> Self:
        """Every finding must cite the defect site, and it must agree with
        ``location``. Disagreement means the model pointed the comment at one
        place while reasoning about another -- a real and subtle failure that is
        otherwise very hard to spot in review output."""
        sites = [e for e in self.evidence if e.role is EvidenceRole.DEFECT_SITE]
        if not sites:
            raise ValueError("evidence must contain at least one DEFECT_SITE span")
        if not any(s.span.overlaps(self.location) for s in sites):
            raise ValueError(
                f"location {self.location} does not overlap any DEFECT_SITE evidence "
                f"({', '.join(str(s.span) for s in sites)})"
            )
        return self

    @model_validator(mode="after")
    def _analyzer_findings_are_not_llm_shaped(self) -> Self:
        if self.source.is_deterministic and self.rule_id is None:
            raise ValueError(f"source={self.source.value} requires a rule_id")
        if self.source is FindingSource.LLM and self.prompt_version is None:
            raise ValueError("LLM findings must record prompt_version for attribution")
        return self

    @model_validator(mode="after")
    def _suggestion_coherence(self) -> Self:
        if self.improved_code is not None and self.suggested_fix is None:
            raise ValueError(
                "improved_code without suggested_fix: a patch must be explained in "
                "prose so a reviewer can judge it without applying it"
            )
        return self

    # -- derived ----------------------------------------------------------- #

    @property
    def priority(self) -> float:
        """Ranking key for the per-PR findings budget.

        Severity dominates; confidence breaks ties within a severity band. Using
        a product rather than a sum means a low-confidence CRITICAL does not
        outrank a high-confidence HIGH, which matches how humans triage.
        """
        return self.severity.rank * self.confidence

    @property
    def fingerprint(self) -> str:
        """Stable identity across runs, for "is this the same finding I already
        showed you?" Excludes confidence and prose so that a prompt tweak does
        not resurrect a dismissed comment."""
        material = "|".join(
            [
                self.location.path,
                str(self.location.line_start),
                self.category.value,
                self.source.value,
                self.rule_id or self.title.lower(),
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def demote(self, gate: VerificationGate, *, penalty: float = 0.3) -> Finding:
        """Return a copy with reduced confidence after a soft gate failure."""
        failed = tuple(dict.fromkeys((*self.verification.gates_failed, gate)))
        return self.model_copy(
            update={
                "confidence": max(0.0, self.confidence - penalty),
                "verification": VerificationResult(
                    status=VerificationStatus.DEMOTED, gates_failed=failed
                ),
            }
        )

    def reject(self, gate: VerificationGate, note: str | None = None) -> Finding:
        """Return a copy marked unpostable. Retained for metrics, never shown."""
        failed = tuple(dict.fromkeys((*self.verification.gates_failed, gate)))
        return self.model_copy(
            update={
                "verification": VerificationResult(
                    status=VerificationStatus.REJECTED,
                    gates_failed=failed,
                    notes=note,
                )
            }
        )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


class ReviewScores(_Frozen):
    """Roll-up scores surfaced on the GitHub Check Run.

    These are computed from findings, never asked of the model. Asking an LLM
    for a "maintainability score out of 100" produces a number with no referent;
    deriving it from verified findings produces one that moves for a reason.
    """

    risk: Annotated[float, Field(ge=0.0, le=1.0)]
    security: Annotated[float, Field(ge=0.0, le=1.0)]
    performance: Annotated[float, Field(ge=0.0, le=1.0)]
    maintainability: Annotated[float, Field(ge=0.0, le=1.0)]
    completeness: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        description="Fraction of changed hunks that were reviewed with a full "
        "context pack. Low values mean the review is partial and the summary "
        "must say so."
    )


class ReviewRun(_Frozen):
    """One review of one (repo, PR, head SHA) under one configuration.

    The idempotency key is the tuple (repository_id, pr_number, head_sha,
    ruleset_version, prompt_version) and is enforced by a UNIQUE constraint in
    Postgres, not by the queue. GitHub redelivers webhooks; a redelivery must be
    a no-op rather than a second paid review.
    """

    id: UUID = Field(default_factory=uuid4)
    repository_id: UUID
    pr_number: Annotated[int, Field(ge=1)]
    head_sha: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    base_sha: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]

    status: RunStatus = RunStatus.QUEUED
    ruleset_version: str
    prompt_version: str
    model: str

    scores: ReviewScores | None = None
    findings_posted: Annotated[int, Field(ge=0)] = 0
    findings_suppressed: Annotated[int, Field(ge=0)] = 0

    cost_usd: Annotated[float, Field(ge=0.0)] = 0.0
    tokens_in: Annotated[int, Field(ge=0)] = 0
    tokens_out: Annotated[int, Field(ge=0)] = 0

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _terminal_state_coherence(self) -> Self:
        if self.status.is_terminal and self.finished_at is None:
            raise ValueError(f"terminal status {self.status.value} requires finished_at")
        if self.status is RunStatus.FAILED and not self.error:
            raise ValueError("failed runs must record an error")
        if self.status is not RunStatus.FAILED and self.error:
            raise ValueError("only failed runs may carry an error")
        if self.head_sha == self.base_sha:
            raise ValueError("head_sha equals base_sha: nothing to review")
        return self

    @property
    def idempotency_key(self) -> str:
        material = "|".join(
            [
                str(self.repository_id),
                str(self.pr_number),
                self.head_sha,
                self.ruleset_version,
                self.prompt_version,
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()
