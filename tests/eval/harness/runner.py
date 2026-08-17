"""One labeled pull request, all the way through, against fakes only.

The chain is M1 -> M3 with nothing stubbed in the middle: index the head tree,
fetch the diff through a ``GitHubPort``, group the hunks by enclosing symbol,
retrieve a context pack, review each group, verify every claim against the head
tree read back through the *same* port, apply the findings budget, and post.

Two boundaries are fakes and everything between them is production code:

* ``FakeGitHub`` is the tree and the diff, and it is also where the output lands
  -- the harness scores ``post_review`` calls, not internal state, so it measures
  what a pull request would actually have received.
* ``RecordedLLM`` replays a transcript with **no provider attached**, so a miss
  raises rather than silently reaching for a model. The cassette is built by
  rendering the same prompts the reviewer is about to send, which makes a prompt
  change a loud cassette miss instead of a quiet change in the numbers.

The cost of a run is therefore zero by construction, and asserted rather than
assumed: ``cost_usd`` is the one number the harness compares configurations on.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid5

from app.domain.contracts import Finding, ReviewRun, RunStatus
from app.domain.ports import PullRequestDiff
from app.domain.retrieval.retriever import ContextRetriever, RetrievalConfig
from app.domain.review.budget import DEFAULT_BUDGET, apply_findings_budget
from app.domain.review.diff import parse_unified_diff
from app.domain.review.grouping import HunkGroup, group_hunks
from app.domain.review.prompts import PROMPT_VERSION, build_review_prompt, system_prompt
from app.domain.review.reviewer import Reviewer, ReviewerConfig, ReviewRequest
from app.domain.review.verification import (
    VerificationContext,
    VerificationReport,
    Verifier,
)
from app.domain.review.vocabulary import build_vocabulary
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.github.fake import FakeGitHub
from app.infra.llm.recorded import RecordedLLM
from app.infra.parsing.syntax import TreeSitterSyntaxChecker
from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore
from app.infra.review.head_files import GitHubHeadFiles
from tests.eval.corpus.indexed import EMBEDDING_DIMENSIONS, build_corpus
from tests.eval.corpus.labeled_prs import LabeledPR
from tests.eval.harness.transcript import Transcript, render_response

__all__ = ["RULESET_VERSION", "CaseRun", "EvalHarness"]

RULESET_VERSION = "argus/m6-partial"
_NAMESPACE = UUID("6d6d6d6d-0000-4000-8000-000000000006")
_MODEL = "transcript"
_SYNTAX = TreeSitterSyntaxChecker()


def _sha(*parts: str) -> str:
    """A stable 40-hex commit id. Real shas are opaque; these only need to be
    distinct, well-formed, and the same on every run."""
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return digest[:40]


@dataclass(frozen=True, slots=True)
class CaseRun:
    """Everything one case produced, at every stage the metrics care about."""

    case: LabeledPR
    groups: tuple[HunkGroup, ...]
    emitted: tuple[Finding, ...]
    """Decoded model output, before verification. The baseline the gate's
    contribution is measured against."""

    decode_rejects: tuple[str, ...]
    report: VerificationReport
    posted: tuple[Finding, ...]
    """What reached the pull request -- read back off the ``GitHubPort``."""

    suppressed: tuple[Finding, ...]
    run: ReviewRun
    duration_s: float
    retrieved_items: int
    cassette_hits: int
    cassette_misses: int

    @property
    def slug(self) -> str:
        return self.case.slug


class EvalHarness:
    """Runs labeled pull requests. One instance per configuration under test."""

    def __init__(
        self,
        *,
        transcript: Transcript,
        retrieval_budget: int = 4000,
        findings_limit: int = DEFAULT_BUDGET,
        repo: str = "argus/eval-corpus",
    ) -> None:
        self._transcript = transcript
        self._retrieval = RetrievalConfig(token_budget=retrieval_budget)
        self._findings_limit = findings_limit
        self._repo = repo
        self.github = FakeGitHub()

    async def run(self, cases: Sequence[LabeledPR]) -> tuple[CaseRun, ...]:
        """Every case, in corpus order. Sequential on purpose: the harness
        measures per-case wall clock, and overlapping the cases would make that
        number a function of the machine's core count."""
        runs: list[CaseRun] = []
        for number, case in enumerate(cases, start=1):
            runs.append(await self.run_case(case, number))
        return tuple(runs)

    async def run_case(self, case: LabeledPR, pr_number: int = 1) -> CaseRun:
        started = time.perf_counter()

        head_sha = _sha(case.slug, "head")
        base_sha = _sha(case.slug, "base")
        run_id = uuid5(_NAMESPACE, f"{case.slug}|{self._transcript.name}")
        repository_id = uuid5(_NAMESPACE, case.slug)

        # -- the fake remote, populated as GitHub would have it --------------- #
        self.github.add_tree(head_sha, case.head_files)
        self.github.add_diff(
            self._repo,
            pr_number,
            PullRequestDiff(
                base_sha=base_sha,
                head_sha=head_sha,
                unified_diff=case.unified_diff,
                changed_paths=[case.path],
                commit_messages=[f"{case.slug}: one-file change"],
            ),
        )

        # -- Stage I/II: index the head tree --------------------------------- #
        corpus = await build_corpus(
            case.head_files, case.test_files, typescript=case.typescript
        )
        index = InMemorySymbolIndex(corpus.snapshot)

        # -- the diff, fetched through the port ------------------------------ #
        fetched = await self.github.fetch_diff(self._repo, pr_number)
        diff = parse_unified_diff(fetched.unified_diff)
        groups = await group_hunks(diff, index, corpus.repository_key)

        # -- Stage III: retrieval -------------------------------------------- #
        retriever = ContextRetriever(
            index=index,
            vectors=InMemoryVectorStore(corpus.snapshot),
            embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
            config=self._retrieval,
        )
        requests: list[ReviewRequest] = []
        for group in groups:
            pack = await retriever.retrieve(
                repository_id=corpus.repository_key, hunks=[group.span]
            )
            requests.append(
                ReviewRequest(group=group, pack=pack, diff_text=case.unified_diff)
            )

        # -- Stage V: review, replayed --------------------------------------- #
        cassette = self._cassette(case, requests)
        result = await Reviewer(
            llm=cassette, config=ReviewerConfig(concurrency=1)
        ).review(requests, run_id=run_id)

        # -- Stage VI: verify against the head tree, read back through the port #
        verifier = Verifier(
            files=GitHubHeadFiles(self.github, repo=self._repo, head_sha=head_sha),
            syntax=_SYNTAX,
        )
        report: VerificationReport = await verifier.verify(
            result.findings,
            VerificationContext(
                diff=diff,
                # The symbol table *and* the packs the model was actually given:
                # a name it read in its own context is not a name it invented.
                known_symbols=build_vocabulary(
                    symbols=corpus.symbols,
                    packs=[request.pack for request in requests],
                ),
            ),
        )
        # The whole report, not just the postable half: the budget ranks only
        # what survived verification but passes the rejected findings through as
        # suppressed, so handing it `postable` would drop every gate casualty
        # from the record the dashboard and the metrics read.
        budgeted = apply_findings_budget(report.findings, limit=self._findings_limit)

        await self.github.post_review(
            self._repo, pr_number, budgeted.posted, _summary(budgeted.posted)
        )
        duration = time.perf_counter() - started

        return CaseRun(
            case=case,
            groups=groups,
            emitted=result.findings,
            decode_rejects=result.rejects,
            report=report,
            posted=budgeted.posted,
            suppressed=budgeted.suppressed,
            run=ReviewRun(
                id=run_id,
                repository_id=repository_id,
                pr_number=pr_number,
                head_sha=head_sha,
                base_sha=base_sha,
                status=RunStatus.SUCCEEDED,
                ruleset_version=RULESET_VERSION,
                prompt_version=PROMPT_VERSION,
                model=_MODEL,
                findings_posted=len(budgeted.posted),
                findings_suppressed=len(budgeted.suppressed),
                cost_usd=result.cost_usd,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                finished_at=datetime.now(UTC),
            ),
            duration_s=duration,
            retrieved_items=sum(len(r.pack.items) for r in requests),
            cassette_hits=cassette.hits,
            cassette_misses=cassette.misses,
        )

    # -- the recorded provider --------------------------------------------- #

    def _cassette(
        self, case: LabeledPR, requests: Sequence[ReviewRequest]
    ) -> RecordedLLM:
        """Record the transcript against the exact prompts about to be sent.

        The case's claims go to the group that encloses the changed line; every
        other group is told there is nothing to report. Assigning by span rather
        than by call order matters because the reviewer fans its calls out
        concurrently -- "the first group asked" is not a stable identity.
        """
        cassette = RecordedLLM(model=_MODEL)
        claims = self._transcript.for_case(case.slug)
        owner = _owning_request(case, requests)

        for request in requests:
            user = build_review_prompt(
                group=request.group, pack=request.pack, diff_text=request.diff_text
            )
            content = (
                render_response(case, claims)
                if request is owner
                else '{"findings": []}'
            )
            cassette.record(
                system=system_prompt(),
                user=user,
                content=content,
                # A local model reports what it actually consumed; a rough
                # character estimate keeps the token figures honest in shape
                # while the price stays what local inference costs.
                tokens_in=len(user) // 4,
                tokens_out=len(content) // 4,
                cost_usd=0.0,
                model=_MODEL,
            )
        return cassette.replaying()


def _owning_request(
    case: LabeledPR, requests: Sequence[ReviewRequest]
) -> ReviewRequest | None:
    for request in requests:
        span = request.group.span
        if (
            span.path == case.path
            and span.line_start <= case.changed_line <= span.line_end
        ):
            return request
    return requests[0] if requests else None


def _summary(posted: Sequence[Finding]) -> str:
    if not posted:
        return "Argus reviewed this pull request and has no comments."
    return f"Argus posted {len(posted)} comment(s) after verification."
