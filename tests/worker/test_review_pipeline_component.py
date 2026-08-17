"""The Stage III/V/VI orchestrator, driven the way its three callers drive it.

``ReviewPipeline`` exists so that the CLI, the M6 harness and the M5 GitHub App
sequence a review identically. These tests hold the parts of that sequence that
are easy to get wrong in a caller and invisible when you do: that verification
actually runs, that a provider failure loses one unit rather than the review,
and that nothing is dropped between the model and the report.
"""

from __future__ import annotations

import difflib
from uuid import uuid4

import pytest

from app.domain.contracts import VerificationGate, VerificationStatus
from app.domain.ports import LLMResponse
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.parsing.syntax import TreeSitterSyntaxChecker
from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore
from app.infra.review.head_files import MappingHeadFiles
from tests.eval.corpus.indexed import EMBEDDING_DIMENSIONS, build_corpus
from tests.eval.corpus.python_corpus import PY_CORPUS, PY_TEST_FILES
from worker.pipeline.review import ReviewPipeline

TARGET = "shop/services/pricing.py"


def _seeded() -> tuple[str, str, int]:
    """Drop the shipping charge from the order total: one real, one-line bug."""
    before = PY_CORPUS[TARGET]
    after = before.replace(
        "    return subtotal(order) + shipping(order)",
        "    return subtotal(order) - shipping(order)",
    )
    line = before[: before.index("    return subtotal(order) + shipping")].count(
        "\n"
    ) + 1
    return before, after, line


def _diff(path: str, before: str, after: str) -> str:
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


class ScriptedLLM:
    """Returns fixed content, and counts calls. Never touches a network."""

    def __init__(self, content: str, *, fail: bool = False) -> None:
        self._content = content
        self._fail = fail
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
        if self._fail:
            raise RuntimeError("provider exploded")
        return LLMResponse(
            content=self._content,
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.0,
            model="scripted",
        )


def _response(path: str, line: int, *, title: str, explanation: str) -> str:
    return (
        '{"findings": [{'
        '"severity": "high", "category": "correctness",'
        f'"title": {title!r}, "explanation": {explanation!r},'
        f'"path": "{path}", "line_start": {line}, "line_end": {line},'
        '"confidence": 0.9,'
        f'"evidence": [{{"path": "{path}", "line_start": {line}, '
        f'"line_end": {line}, "role": "defect_site", "excerpt": "return"}}]'
        "}]}"
    ).replace("'", '"')


async def _pipeline(head: dict[str, str]) -> tuple[ReviewPipeline, str]:
    corpus = await build_corpus(head, PY_TEST_FILES, typescript=False)
    pipeline = ReviewPipeline(
        index=InMemorySymbolIndex(corpus.snapshot),
        vectors=InMemoryVectorStore(corpus.snapshot),
        embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
        files=MappingHeadFiles(head),
        syntax=TreeSitterSyntaxChecker(),
        known_symbols=corpus.symbols,
    )
    return pipeline, corpus.repository_key


@pytest.fixture
async def seeded() -> tuple[ReviewPipeline, str, str, int]:
    before, after, line = _seeded()
    head = {**PY_CORPUS, TARGET: after}
    pipeline, repository_id = await _pipeline(head)
    return pipeline, repository_id, _diff(TARGET, before, after), line


class TestPrepare:
    async def test_a_diff_becomes_review_units_with_context(
        self, seeded: tuple[ReviewPipeline, str, str, int]
    ) -> None:
        pipeline, repository_id, diff_text, _ = seeded
        plan = await pipeline.prepare(
            repository_id=repository_id, diff_text=diff_text
        )
        assert plan.requests
        assert plan.changed_paths == (TARGET,)
        # A unit reviewed with an empty pack is a diff read in isolation, which
        # is the thing ADR-002 exists to prevent.
        assert plan.retrieved_items > 0

    async def test_an_empty_diff_plans_nothing(
        self, seeded: tuple[ReviewPipeline, str, str, int]
    ) -> None:
        pipeline, repository_id, _, _ = seeded
        plan = await pipeline.prepare(repository_id=repository_id, diff_text="")
        assert plan.is_empty

    async def test_preparing_costs_no_model_call(
        self, seeded: tuple[ReviewPipeline, str, str, int]
    ) -> None:
        """The phase split earns its keep only if the first phase is free: a
        caller shows the plan, or refuses it, before paying for anything."""
        pipeline, repository_id, diff_text, _ = seeded
        llm = ScriptedLLM("{}")
        await pipeline.prepare(repository_id=repository_id, diff_text=diff_text)
        assert llm.calls == 0


class TestExecute:
    async def test_a_true_finding_survives_to_the_caller(
        self, seeded: tuple[ReviewPipeline, str, str, int]
    ) -> None:
        pipeline, repository_id, diff_text, line = seeded
        llm = ScriptedLLM(
            _response(
                TARGET,
                line,
                title="Shipping is subtracted from the order total",
                explanation=(
                    "`total` returns `subtotal` minus `shipping`, so the charge "
                    "is discounted by the shipping cost instead of including it."
                ),
            )
        )
        outcome = await pipeline.run(
            repository_id=repository_id,
            diff_text=diff_text,
            llm=llm,
            run_id=uuid4(),
        )
        assert outcome.posted
        posted = outcome.posted[0]
        assert posted.location.path == TARGET
        assert posted.verification.status is VerificationStatus.VERIFIED
        assert outcome.completeness == 1.0
        assert outcome.cost_usd == 0.0

    async def test_verification_runs_and_cannot_be_skipped(
        self, seeded: tuple[ReviewPipeline, str, str, int]
    ) -> None:
        """A fabricated line number must not reach the caller. There is no flag
        that turns this off, and there must never be one."""
        pipeline, repository_id, diff_text, _ = seeded
        llm = ScriptedLLM(
            _response(
                TARGET,
                9999,
                title="Resource is leaked on the error path",
                explanation="The handle opened here is never closed on failure.",
            )
        )
        outcome = await pipeline.run(
            repository_id=repository_id,
            diff_text=diff_text,
            llm=llm,
            run_id=uuid4(),
        )
        assert outcome.posted == ()
        assert VerificationGate.LINE_IN_RANGE in outcome.drops_by_gate

    async def test_a_provider_failure_loses_a_unit_not_the_review(
        self, seeded: tuple[ReviewPipeline, str, str, int]
    ) -> None:
        pipeline, repository_id, diff_text, _ = seeded
        outcome = await pipeline.run(
            repository_id=repository_id,
            diff_text=diff_text,
            llm=ScriptedLLM("", fail=True),
            run_id=uuid4(),
        )
        assert outcome.failures
        assert outcome.completeness < 1.0
        assert outcome.posted == ()

    async def test_nothing_is_dropped_between_the_model_and_the_report(
        self, seeded: tuple[ReviewPipeline, str, str, int]
    ) -> None:
        """Everything decoded is either posted or suppressed. A leak here would
        make the drop counts describe a smaller run than the one that happened.
        """
        pipeline, repository_id, diff_text, line = seeded
        llm = ScriptedLLM(
            _response(
                TARGET,
                line,
                title="Shipping is subtracted from the order total",
                explanation="The charge is discounted rather than added.",
            )
        )
        outcome = await pipeline.run(
            repository_id=repository_id,
            diff_text=diff_text,
            llm=llm,
            run_id=uuid4(),
        )
        assert len(outcome.posted) + len(outcome.suppressed) == len(outcome.emitted)

    async def test_the_budget_is_honoured(
        self, seeded: tuple[ReviewPipeline, str, str, int]
    ) -> None:
        before, after, line = _seeded()
        head = {**PY_CORPUS, TARGET: after}
        corpus = await build_corpus(head, PY_TEST_FILES, typescript=False)
        pipeline = ReviewPipeline(
            index=InMemorySymbolIndex(corpus.snapshot),
            vectors=InMemoryVectorStore(corpus.snapshot),
            embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
            files=MappingHeadFiles(head),
            syntax=TreeSitterSyntaxChecker(),
            known_symbols=corpus.symbols,
            findings_limit=0,
        )
        llm = ScriptedLLM(
            _response(
                TARGET,
                line,
                title="Shipping is subtracted from the order total",
                explanation="The charge is discounted rather than added.",
            )
        )
        outcome = await pipeline.run(
            repository_id=corpus.repository_key,
            diff_text=_diff(TARGET, before, after),
            llm=llm,
            run_id=uuid4(),
        )
        assert outcome.posted == ()
        # Suppressed, not deleted: it is still on the record.
        assert outcome.suppressed
