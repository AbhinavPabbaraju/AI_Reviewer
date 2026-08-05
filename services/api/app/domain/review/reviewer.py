"""Stage V: one structured LLM call per hunk group, fanned out with a bound.

The unit of review is the hunk group (see ``grouping.py``), so this module's job
is small and mostly about honesty under failure:

* **One call per group, bounded concurrency.** A 40-file PR is 40-odd calls;
  firing them all at once is how a review trips a provider rate limit and fails
  wholesale instead of finishing slightly slower.
* **A failed call is a recorded fact, not an exception.** One group whose
  provider call errors or whose response will not decode must not lose the other
  thirty-nine reviews. Failures are counted and reported alongside the findings,
  because a review that silently covered half the diff is worse than one that
  says so -- ``ReviewScores.completeness`` exists for exactly this.
* **Cost is measured, not estimated.** Per-run cost is an SLO, and a budget you
  only see on the provider's dashboard is not enforced. Tokens and cost come
  back from the adapter and are summed here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final
from uuid import UUID

from app.domain.contracts import Finding
from app.domain.ports import LLMPort
from app.domain.retrieval.models import ContextPack
from app.domain.review.decoding import decode_findings, draft_schema
from app.domain.review.grouping import HunkGroup
from app.domain.review.prompts import PROMPT_VERSION, build_review_prompt, system_prompt

__all__ = ["ReviewRequest", "ReviewResult", "Reviewer", "ReviewerConfig"]


@dataclass(frozen=True, slots=True)
class ReviewerConfig:
    concurrency: int = 4
    """Simultaneous provider calls. Four is a deliberately unambitious default:
    the wall-clock win from more is modest next to the failure mode of getting
    rate-limited halfway through a review."""

    max_tokens: int = 8192
    temperature: float = 0.0
    """Kept because local providers still honour it. Anthropic's current models
    removed the parameter -- an adapter for those must drop it rather than pass
    it through, or every request is a 400."""

    constrain_json: bool = True
    """Pass the response schema to providers that can constrain decoding to it.
    Adapters that cannot simply ignore it; the decoder validates either way."""

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True, slots=True)
class ReviewRequest:
    """One reviewable unit: what changed, and what to read while judging it."""

    group: HunkGroup
    pack: ContextPack
    diff_text: str


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """Everything one review produced, including what it failed to produce."""

    findings: tuple[Finding, ...] = ()
    rejects: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    """Groups whose provider call raised. Distinct from ``rejects`` (the call
    succeeded, the response did not decode) because they have different causes
    and different fixes."""

    groups_reviewed: int = 0
    groups_total: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    models: tuple[str, ...] = field(default=())

    @property
    def completeness(self) -> float:
        """Fraction of hunk groups actually reviewed. Feeds
        ``ReviewScores.completeness``: a partial review must say so."""
        if not self.groups_total:
            return 1.0
        return self.groups_reviewed / self.groups_total


_EMPTY: Final = ReviewResult()


class Reviewer:
    """Runs Stage V. One instance per run; ``review`` is the whole API."""

    def __init__(
        self,
        *,
        llm: LLMPort,
        config: ReviewerConfig | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self._llm = llm
        self._config = config or ReviewerConfig()
        self._prompt_version = prompt_version

    async def review(
        self, requests: Sequence[ReviewRequest], *, run_id: UUID
    ) -> ReviewResult:
        if not requests:
            return _EMPTY

        semaphore = asyncio.Semaphore(self._config.concurrency)

        async def one(request: ReviewRequest) -> _GroupOutcome:
            async with semaphore:
                return await self._review_group(request, run_id=run_id)

        outcomes = await asyncio.gather(*(one(request) for request in requests))
        return _merge(outcomes, groups_total=len(requests))

    async def _review_group(
        self, request: ReviewRequest, *, run_id: UUID
    ) -> _GroupOutcome:
        prompt = build_review_prompt(
            group=request.group, pack=request.pack, diff_text=request.diff_text
        )
        try:
            response = await self._llm.complete(
                system=system_prompt(),
                user=prompt,
                json_schema=draft_schema() if self._config.constrain_json else None,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except Exception as error:
            # Deliberately broad: adapters raise provider-specific exceptions
            # (timeouts, rate limits, transport errors) that the domain must not
            # import to catch. One group's provider failure is recorded and the
            # other groups still produce a review.
            return _GroupOutcome(
                failures=(f"{request.group.path}: {type(error).__name__}: {error}",)
            )

        decoded = decode_findings(
            response.content,
            run_id=run_id,
            prompt_version=self._prompt_version,
            model=response.model,
            allowed_paths=(request.group.path,),
        )
        return _GroupOutcome(
            findings=decoded.findings,
            rejects=tuple(
                f"{request.group.path}: {reason}" for reason in decoded.rejects
            ),
            reviewed=1,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=response.cost_usd,
            model=response.model,
        )


@dataclass(frozen=True, slots=True)
class _GroupOutcome:
    findings: tuple[Finding, ...] = ()
    rejects: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    reviewed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str | None = None


def _merge(outcomes: Sequence[_GroupOutcome], *, groups_total: int) -> ReviewResult:
    models: dict[str, None] = {}
    for outcome in outcomes:
        if outcome.model:
            models.setdefault(outcome.model, None)
    return ReviewResult(
        findings=tuple(f for o in outcomes for f in o.findings),
        rejects=tuple(r for o in outcomes for r in o.rejects),
        failures=tuple(f for o in outcomes for f in o.failures),
        groups_reviewed=sum(o.reviewed for o in outcomes),
        groups_total=groups_total,
        tokens_in=sum(o.tokens_in for o in outcomes),
        tokens_out=sum(o.tokens_out for o in outcomes),
        cost_usd=sum(o.cost_usd for o in outcomes),
        models=tuple(models),
    )
