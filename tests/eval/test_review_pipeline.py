"""The whole pipeline, end to end, at zero cost.

This is the first test that runs M1 through M3 as one chain: index a repository,
diff a change, group the hunks, retrieve context, review, decode, verify, and
apply the findings budget. Everything before Stage V was already free -- an
offline hashing embedder, structure-first retrieval, a pure verification gate --
and this proves the last step is too.

"Free" is asserted, not assumed: ``cost_usd`` must be exactly zero across the
run. That number is the one the eval harness compares configurations on, so a
provider that quietly reported a price would corrupt it.

The run is also *deterministic*, which is the property the M6 harness needs: the
same corpus and the same cassette produce byte-identical findings on every run,
so a change in output is a change in the code rather than in the weather.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

import pytest

from app.domain.contracts import VerificationStatus
from app.domain.ports import LLMResponse
from app.domain.retrieval.retriever import ContextRetriever, RetrievalConfig
from app.domain.review.budget import BudgetResult, apply_findings_budget
from app.domain.review.diff import parse_unified_diff
from app.domain.review.grouping import group_hunks
from app.domain.review.prompts import build_review_prompt, system_prompt
from app.domain.review.reviewer import (
    Reviewer,
    ReviewerConfig,
    ReviewRequest,
    ReviewResult,
)
from app.domain.review.verification import (
    VerificationContext,
    VerificationReport,
    Verifier,
)
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.llm.recorded import RecordedLLM
from app.infra.parsing.syntax import TreeSitterSyntaxChecker
from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore
from app.infra.review.head_files import MappingHeadFiles
from tests.eval.corpus.indexed import EMBEDDING_DIMENSIONS, build_corpus
from tests.eval.corpus.python_corpus import PY_CORPUS, PY_TEST_FILES

TARGET_PATH = "shop/store/memory.py"


def _seed_defect() -> tuple[str, str, int]:
    """Introduce one real defect: drop the validation call before persisting.

    A genuine change to a real corpus file, diffed with difflib, so every line
    number downstream is a line number in a file that exists.
    """
    before = PY_CORPUS[TARGET_PATH]
    lines = before.splitlines()
    target = next(
        n for n, line in enumerate(lines) if "validate" in line and line.startswith(" ")
    )
    lines[target] = " " * (len(lines[target]) - len(lines[target].lstrip())) + "pass"
    return before, "\n".join(lines) + "\n", target + 1


def _unified(path: str, before: str, after: str) -> str:
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


def _model_response(path: str, line: int) -> str:
    """What a competent reviewer should say about the seeded defect.

    Hand-authored rather than captured from a live model: the pipeline is what
    is under test, and a canned response makes the assertions about *the
    pipeline* rather than about a particular model's mood.
    """
    return (
        '{"findings": [{'
        '"severity": "high", "category": "correctness",'
        '"title": "Entity is persisted without being validated",'
        '"explanation": "The validation call was removed, so an invalid entity '
        'is written to the store and every later read returns it.",'
        f'"path": "{path}", "line_start": {line}, "line_end": {line},'
        '"confidence": 0.88,'
        f'"evidence": [{{"path": "{path}", "line_start": {line}, '
        f'"line_end": {line}, "role": "defect_site", "excerpt": "pass"}}]'
        "}]}"
    )


class CannedLLM:
    """A stand-in provider used once, to populate the cassette."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=self._content,
            tokens_in=1200,
            tokens_out=180,
            cost_usd=0.0,
            model="canned",
        )


@dataclass(frozen=True, slots=True)
class Pipeline:
    """One full offline run, kept whole so the assertions can be specific."""

    changed_line: int
    groups: Sequence[object]
    requests: Sequence[ReviewRequest]
    result: ReviewResult
    report: VerificationReport
    budgeted: BudgetResult
    canned: CannedLLM
    replay: RecordedLLM


@pytest.fixture(scope="module")
async def pipeline() -> Pipeline:
    """Run the full pipeline once and hand the pieces to the assertions."""
    before, after, changed_line = _seed_defect()
    head_files = {**PY_CORPUS, TARGET_PATH: after}

    corpus = await build_corpus(head_files, PY_TEST_FILES, typescript=False)
    diff = parse_unified_diff(_unified(TARGET_PATH, before, after))

    index = InMemorySymbolIndex(corpus.snapshot)
    groups = await group_hunks(diff, index, corpus.repository_key)

    retriever = ContextRetriever(
        index=index,
        vectors=InMemoryVectorStore(corpus.snapshot),
        embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
        config=RetrievalConfig(token_budget=4000),
    )

    requests = []
    for group in groups:
        pack = await retriever.retrieve(
            repository_id=corpus.repository_key, hunks=[group.span]
        )
        requests.append(
            ReviewRequest(group=group, pack=pack, diff_text=_unified(
                TARGET_PATH, before, after
            ))
        )

    # Record once through a canned provider, then replay with no provider at all.
    canned = CannedLLM(_model_response(TARGET_PATH, changed_line))
    recorder = RecordedLLM(delegate=canned)
    run_id = uuid4()
    await Reviewer(llm=recorder, config=ReviewerConfig(concurrency=2)).review(
        requests, run_id=run_id
    )

    replay = recorder.replaying()
    result = await Reviewer(llm=replay, config=ReviewerConfig(concurrency=2)).review(
        requests, run_id=run_id
    )

    verifier = Verifier(
        files=MappingHeadFiles(head_files), syntax=TreeSitterSyntaxChecker()
    )
    report = await verifier.verify(
        result.findings,
        VerificationContext(
            diff=diff, known_symbols=frozenset(corpus.symbols) | {"validate"}
        ),
    )
    budgeted = apply_findings_budget(report.postable, limit=10)

    return Pipeline(
        changed_line=changed_line,
        groups=groups,
        requests=requests,
        result=result,
        report=report,
        budgeted=budgeted,
        canned=canned,
        replay=replay,
    )


class TestEndToEndReview:
    async def test_the_diff_groups_into_review_units(
        self, pipeline: Pipeline
    ) -> None:
        assert pipeline.groups, "the seeded change produced no reviewable unit"

    async def test_a_finding_survives_to_the_pull_request(
        self, pipeline: Pipeline
    ) -> None:
        """The whole point: a real defect, reviewed offline, reaches the top."""
        assert pipeline.budgeted.posted, "no finding survived the pipeline"
        posted = pipeline.budgeted.posted[0]
        assert posted.location.path == TARGET_PATH
        assert posted.verification.is_postable
        assert posted.verification.status is not VerificationStatus.PENDING

    async def test_the_whole_run_costs_nothing(self, pipeline: Pipeline) -> None:
        """The requirement, asserted rather than assumed."""
        assert pipeline.result.cost_usd == 0.0

    async def test_replay_needed_no_provider_at_all(
        self, pipeline: Pipeline
    ) -> None:
        """The second pass ran with no delegate: every response came from the
        cassette. This is the shape the M6 harness runs in."""
        assert pipeline.replay.hits > 0
        assert pipeline.replay.misses == 0
        # The canned provider was consulted only during the recording pass.
        assert pipeline.canned.calls == len(pipeline.requests)

    async def test_every_group_was_reviewed(self, pipeline: Pipeline) -> None:
        assert pipeline.result.completeness == 1.0
        assert pipeline.result.failures == ()

    async def test_the_prompt_carried_real_retrieved_context(
        self, pipeline: Pipeline
    ) -> None:
        """A review with an empty pack is a review of a diff in isolation, which
        is the thing ADR-002 exists to avoid."""
        anchored = [r for r in pipeline.requests if r.pack.items]
        assert anchored, "no request carried any retrieved context"
        prompt = build_review_prompt(
            group=anchored[0].group,
            pack=anchored[0].pack,
            diff_text=anchored[0].diff_text,
        )
        assert "CHANGED CODE" in prompt
        assert "<untrusted-diff>" in prompt
        assert len(system_prompt()) > 500

    async def test_verification_ran_on_the_model_output(
        self, pipeline: Pipeline
    ) -> None:
        assert pipeline.report.findings
        for finding in pipeline.report.findings:
            assert finding.verification.status is not VerificationStatus.PENDING
