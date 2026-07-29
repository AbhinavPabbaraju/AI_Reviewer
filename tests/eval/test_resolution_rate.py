"""The M1 exit gate: symbol-resolution accuracy on a hand-labeled corpus.

ROADMAP.md M1 requires "symbol resolution rate >= 85% on a hand-labeled sample of
200 references per language -- measured, not asserted." This module *measures*
it: :mod:`tests.eval.corpus` holds two self-contained repositories and the
ground-truth resolution for each labeled reference, and the tests here compute
accuracy against that ground truth and gate on 0.85.

Two numbers are reported, and they answer different questions:

``accuracy``
    Fraction of hand-labeled references resolved the way a competent reader of
    the corpus would resolve them. This is the exit criterion. It counts an
    over-eager resolution (binding a reference that has no repository target) as
    a failure, which a resolution *rate* cannot do.
``operational rate``
    ``resolved / (resolved + unresolved)`` over intra-repo references, as
    computed in production by :class:`ResolutionStats`. It is what the indexer
    reports per run, and it needs the labeled accuracy beside it: a resolver that
    guessed on every reference would score 100% here and be worthless.

Run ``pytest tests/eval -s`` to see both, per language.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from app.domain.indexing.models import (
    EdgeKind,
    Language,
    ParsedFile,
    SymbolEdge,
    confidence,
)
from app.domain.indexing.resolution import ResolutionResult, Resolver
from tests.conftest import parse_python, parse_typescript
from tests.eval.corpus import EXTERNAL, UNRESOLVED, Expectation
from tests.eval.corpus.python_corpus import (
    PY_CORPUS,
    PY_EXPECTATIONS,
    PY_TEST_FILES,
)
from tests.eval.corpus.typescript_corpus import (
    TS_CORPUS,
    TS_EXPECTATIONS,
    TS_TEST_FILES,
)

RESOLUTION_TARGET = 0.85
"""ROADMAP M1 exit criterion. Deliberately below the measured value: the gap is
what a regression has to burn through before the build goes red."""

CORPUS_TARGET = 200
"""References per language the roadmap asks the corpus to reach."""


def _resolve(
    corpus: Mapping[str, str], test_files: frozenset[str], *, typescript: bool
) -> ResolutionResult:
    parse = parse_typescript if typescript else parse_python
    parsed: list[ParsedFile] = [
        parse(path, text, is_test=path in test_files) for path, text in corpus.items()
    ]
    return Resolver(parsed).resolve()


def _score(
    edges: Sequence[SymbolEdge], expectations: Sequence[Expectation]
) -> tuple[float, list[Expectation]]:
    """Accuracy against ground truth, plus the references that missed it."""
    resolved: set[tuple[EdgeKind, str, str | None]] = {
        (e.kind, e.src_fqn, e.dst_fqn) for e in edges if e.is_resolved
    }
    external: set[tuple[EdgeKind, str, str | None]] = {
        (e.kind, e.src_fqn, e.dst_unresolved_name)
        for e in edges
        if not e.is_resolved and e.confidence == confidence.EXTERNAL
    }
    unbound: set[tuple[EdgeKind, str, str | None]] = {
        (e.kind, e.src_fqn, e.dst_unresolved_name) for e in edges if not e.is_resolved
    }

    misses: list[Expectation] = []
    for kind, src, target in expectations:
        match target:
            case EXTERNAL(name=name):
                ok = (kind, src, name) in external
            case UNRESOLVED(name=name):
                ok = (kind, src, name) in unbound
            case _:
                ok = (kind, src, target) in resolved
        if not ok:
            misses.append((kind, src, target))
    return 1.0 - len(misses) / len(expectations), misses


def _report(
    language: Language, result: ResolutionResult, expectations: Sequence[Expectation]
) -> list[Expectation]:
    accuracy, misses = _score(result.edges, expectations)
    stats = result.stats
    print(
        f"\n[{language.value}] labeled accuracy {accuracy:.2%} "
        f"over {len(expectations)} references ({len(misses)} missed) | "
        f"operational rate {stats.rate_for(language):.2%} | "
        f"edges {stats.total_edges} "
        f"(resolved {stats.resolved}, external {stats.external}, "
        f"unresolved {stats.unresolved})"
    )
    for kind, src, target in misses:
        print(f"    miss: {kind.value:10} {src} -> {target}")
    assert accuracy >= RESOLUTION_TARGET, (
        f"{language.value} resolution accuracy {accuracy:.2%} is below the "
        f"{RESOLUTION_TARGET:.0%} M1 exit criterion; missed {misses}"
    )
    return misses


@pytest.fixture(scope="module")
def python_result() -> ResolutionResult:
    return _resolve(PY_CORPUS, PY_TEST_FILES, typescript=False)


@pytest.fixture(scope="module")
def typescript_result() -> ResolutionResult:
    return _resolve(TS_CORPUS, TS_TEST_FILES, typescript=True)


class TestPythonResolutionRate:
    def test_corpus_is_large_enough(self) -> None:
        assert len(PY_EXPECTATIONS) >= CORPUS_TARGET

    def test_labeled_accuracy_meets_target(
        self, python_result: ResolutionResult
    ) -> None:
        _report(Language.PYTHON, python_result, PY_EXPECTATIONS)

    def test_operational_rate_meets_target(
        self, python_result: ResolutionResult
    ) -> None:
        assert python_result.stats.rate_for(Language.PYTHON) >= RESOLUTION_TARGET


class TestTypeScriptResolutionRate:
    def test_corpus_is_large_enough(self) -> None:
        assert len(TS_EXPECTATIONS) >= CORPUS_TARGET

    def test_labeled_accuracy_meets_target(
        self, typescript_result: ResolutionResult
    ) -> None:
        _report(Language.TYPESCRIPT, typescript_result, TS_EXPECTATIONS)

    def test_operational_rate_meets_target(
        self, typescript_result: ResolutionResult
    ) -> None:
        assert (
            typescript_result.stats.rate_for(Language.TYPESCRIPT) >= RESOLUTION_TARGET
        )


class TestCorpusIntegrity:
    """The corpus is only ground truth if it stays hand-labeled and distinct.

    Duplicated expectations would inflate accuracy for free (the same edge
    scored twice), and a label that can never be satisfied -- an fqn no symbol
    in the corpus carries -- would quietly become permanent, unfixable headroom.
    """

    @pytest.mark.parametrize(
        ("name", "expectations"),
        [("python", PY_EXPECTATIONS), ("typescript", TS_EXPECTATIONS)],
    )
    def test_expectations_are_unique(
        self, name: str, expectations: Sequence[Expectation]
    ) -> None:
        seen = {
            (kind, src, target if isinstance(target, str) else target.name)
            for kind, src, target in expectations
        }
        assert len(seen) == len(expectations), f"{name} corpus has duplicate labels"

    @pytest.mark.parametrize(
        ("name", "corpus", "test_files", "expectations", "typescript"),
        [
            ("python", PY_CORPUS, PY_TEST_FILES, PY_EXPECTATIONS, False),
            ("typescript", TS_CORPUS, TS_TEST_FILES, TS_EXPECTATIONS, True),
        ],
    )
    def test_labeled_sources_exist_in_the_corpus(
        self,
        name: str,
        corpus: Mapping[str, str],
        test_files: frozenset[str],
        expectations: Sequence[Expectation],
        typescript: bool,
    ) -> None:
        parse = parse_typescript if typescript else parse_python
        known = {
            symbol.fqn
            for path, text in corpus.items()
            for symbol in parse(path, text, is_test=path in test_files).symbols
        }
        unknown = {src for _, src, _ in expectations if src not in known}
        assert not unknown, f"{name} labels reference symbols that do not exist: {unknown}"
