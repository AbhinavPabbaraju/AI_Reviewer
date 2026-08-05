"""20 seeded-defect pull requests, and a reviewer that lies about them.

M3's exit criterion is a zero: *on 20 seeded-defect PRs, zero findings escape the
gate with a nonexistent file, an out-of-range line, or an unparseable patch.*
A zero is only meaningful if something was genuinely trying to get through, so
the findings here do not come from a model -- they come from
:class:`AdversarialReviewer`, which fabricates on purpose.

That is a deliberate choice over calling a real model. A real model cannot be
asked to hallucinate on demand: on a good day it emits nothing invalid and the
gate is never exercised, on a bad day it emits something unpredictable, and
either way the run costs money and is not reproducible. The fabrications below
are the exact failure modes sec. 4.6 names, one per gate, produced deterministically
every run. This is also the shape M6's recorded ``LLMPort`` will replay.

The PRs themselves are real: real files from the retrieval corpora, mutated by
one line, diffed with ``difflib`` into genuine unified diffs, and parsed by the
same parser production uses. The line numbers the fabrications abuse are
therefore real line numbers in a real tree.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from app.domain.contracts import (
    Category,
    CodeSpan,
    Evidence,
    EvidenceRole,
    Finding,
    FindingSource,
    Severity,
)
from app.domain.review.diff import ParsedDiff, parse_unified_diff
from tests.eval.corpus.python_corpus import PY_CORPUS
from tests.eval.corpus.typescript_corpus import TS_CORPUS

__all__ = ["AdversarialReviewer", "SeededPR", "seeded_prs"]

RUN_ID = UUID("33333333-3333-3333-3333-333333333333")


@dataclass(frozen=True, slots=True)
class SeededPR:
    """One PR: the tree at head, the diff, and where the defect was seeded."""

    slug: str
    head_files: Mapping[str, str]
    diff: ParsedDiff
    path: str
    changed_line: int
    """A line this PR actually changed -- the anchor a *legitimate* finding uses,
    and the baseline the fabrications are measured against."""

    @property
    def line_count(self) -> int:
        return len(self.head_files[self.path].splitlines())


def _mutate(text: str, path: str, index: int) -> tuple[str, int] | None:
    """Change one indented, non-trivial line of a real file.

    The edit is a trailing comment in the file's own syntax, so the mutated tree
    still parses. That matters: these files are the *head* of the PR, and a head
    that did not parse would be testing the gate against a repository state that
    could not exist.
    """
    marker = "  // seeded defect" if path.endswith((".ts", ".tsx")) else "  # seeded defect"
    lines = text.splitlines()
    candidates = [
        n
        for n, line in enumerate(lines)
        if line.startswith((" ", "\t"))
        and line.strip()
        and not line.strip().startswith(("#", "//", "*", '"""', "'''"))
    ]
    if not candidates:
        # A package marker or a file of bare declarations. Not every file makes
        # a PR; skipped rather than forced, since a synthetic edit to an empty
        # `__init__.py` would not exercise anything.
        return None
    target = candidates[index % len(candidates)]
    lines[target] = lines[target] + marker
    return "\n".join(lines) + "\n", target + 1


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


def seeded_prs() -> tuple[SeededPR, ...]:
    """Twenty PRs, sixteen Python and four TypeScript.

    Drawn from both corpora because ``PATCH_PARSES`` dispatches on the file's
    grammar, and a corpus of one language would leave that dispatch unexercised.
    """
    sources: list[tuple[str, Mapping[str, str]]] = [
        *[(path, PY_CORPUS) for path in sorted(PY_CORPUS)],
        *[(path, TS_CORPUS) for path in sorted(TS_CORPUS)],
    ]
    prs: list[SeededPR] = []
    for index, (path, corpus) in enumerate(sources):
        if len(prs) == 20:
            break
        before = corpus[path]
        mutated = _mutate(before, path, index)
        if mutated is None:
            continue
        after, line = mutated
        head = {**corpus, path: after}
        prs.append(
            SeededPR(
                slug=f"pr-{len(prs) + 1:02d}-{path}",
                head_files=head,
                diff=parse_unified_diff(_unified(path, before, after)),
                path=path,
                changed_line=line,
            )
        )
    if len(prs) < 20:  # pragma: no cover - corpora are far larger than 20 files
        raise ValueError(f"only built {len(prs)} seeded PRs, need 20")
    return tuple(prs)


class AdversarialReviewer:
    """Emits findings for a PR, most of them fabricated.

    Every method below is one failure mode from ARCHITECTURE sec. 4.6, written
    to be as plausible as possible: the fabrications cite real-looking paths,
    real-looking line numbers and real-looking symbols, because a gate that only
    catches obvious garbage catches nothing that matters.
    """

    def review(self, pr: SeededPR) -> tuple[Finding, ...]:
        return (
            self._legitimate(pr),
            self._nonexistent_file(pr),
            self._line_past_end_of_file(pr),
            self._line_outside_the_diff(pr),
            self._unparseable_patch(pr),
            self._patch_outside_the_diff(pr),
            self._duplicate_of_the_legitimate_one(pr),
            self._invented_symbol(pr),
            self._below_confidence_floor(pr),
            self._unrelated_file(pr),
        )

    # -- the one that should survive ---------------------------------------- #

    def _legitimate(self, pr: SeededPR) -> Finding:
        return _finding(
            pr,
            title="Seeded defect changes behaviour on the modified line",
            line=pr.changed_line,
            confidence=0.9,
        )

    # -- fabrications, one per gate ----------------------------------------- #

    def _nonexistent_file(self, pr: SeededPR) -> Finding:
        """FILE_EXISTS. A plausible sibling path that is not in the tree."""
        return _finding(
            pr,
            title="Null dereference in the helper module",
            line=1,
            path=pr.path.rsplit(".", 1)[0] + "_helpers." + pr.path.rsplit(".", 1)[1],
        )

    def _line_past_end_of_file(self, pr: SeededPR) -> Finding:
        """LINE_IN_RANGE. Right file, line number past the end."""
        beyond = pr.line_count + 25
        return _finding(
            pr, title="Resource is leaked on the error path", line=beyond
        )

    def _line_outside_the_diff(self, pr: SeededPR) -> Finding:
        """LINE_IN_RANGE. A real line in a real file that this PR never touched."""
        line = 1 if pr.changed_line > 5 else pr.line_count
        return _finding(
            pr, title="Unvalidated input flows into the sink here", line=line
        )

    def _unparseable_patch(self, pr: SeededPR) -> Finding:
        """PATCH_PARSES. Prose is sound, patch is syntactic garbage."""
        return _finding(
            pr,
            title="Guard clause is missing before the call",
            line=pr.changed_line,
            suggested_fix="Add the guard before calling.",
            improved_code="if (((: return ,,, else ]]]",
        )

    def _patch_outside_the_diff(self, pr: SeededPR) -> Finding:
        """PATCH_APPLIES. Valid syntax, but anchored where GitHub cannot render
        a suggestion."""
        return _finding(
            pr,
            title="Return value is ignored by the caller",
            line=pr.changed_line,
            line_end=min(pr.changed_line + 40, pr.line_count),
            suggested_fix="Propagate the result.",
            improved_code="x = 1",
        )

    def _duplicate_of_the_legitimate_one(self, pr: SeededPR) -> Finding:
        """NOT_DUPLICATE. Same line, same words, lower confidence."""
        return _finding(
            pr,
            title="Seeded defect changes behaviour on the modified line",
            line=pr.changed_line,
            confidence=0.75,
        )

    def _invented_symbol(self, pr: SeededPR) -> Finding:
        """SYMBOL_RESOLVES. Cites a helper that does not exist anywhere."""
        return _finding(
            pr,
            title="Sanitizer is bypassed on this branch",
            line=pr.changed_line,
            explanation=(
                "The `Sanitizer.normalize_and_check` helper is skipped here, so "
                "the value reaches the sink without ever being validated."
            ),
        )

    def _below_confidence_floor(self, pr: SeededPR) -> Finding:
        """CONFIDENCE_FLOOR. Real location, admitted low confidence, low value."""
        return _finding(
            pr,
            title="Variable name does not describe its contents",
            line=pr.changed_line,
            severity=Severity.LOW,
            confidence=0.4,
        )

    def _unrelated_file(self, pr: SeededPR) -> Finding:
        """LINE_IN_RANGE, cross-file. A real file in the repo, untouched by this
        PR, with nothing tying it back to the change."""
        other = next(
            (p for p in sorted(pr.head_files) if p != pr.path), pr.path
        )
        return _finding(
            pr, title="Configuration default is unsafe for production", line=1,
            path=other,
        )


def _finding(
    pr: SeededPR,
    *,
    title: str,
    line: int,
    path: str | None = None,
    line_end: int | None = None,
    severity: Severity = Severity.HIGH,
    confidence: float = 0.85,
    explanation: str | None = None,
    suggested_fix: str | None = None,
    improved_code: str | None = None,
) -> Finding:
    target = path or pr.path
    span = CodeSpan(
        path=target, line_start=line, line_end=max(line_end or line, line)
    )
    return Finding(
        id=uuid4(),
        run_id=RUN_ID,
        severity=severity,
        category=Category.CORRECTNESS,
        source=FindingSource.LLM,
        title=title,
        explanation=explanation
        or (
            "The changed line alters control flow so the surrounding function "
            "returns before its invariant is re-established."
        ),
        location=span,
        evidence=(
            Evidence(
                span=span,
                role=EvidenceRole.DEFECT_SITE,
                excerpt="<excerpt as the model claimed to see it>",
            ),
        ),
        confidence=confidence,
        suggested_fix=suggested_fix,
        improved_code=improved_code,
        prompt_version="adversarial/v1",
    )


def known_symbols(paths: Sequence[str]) -> frozenset[str]:
    """A deliberately small vocabulary for ``SYMBOL_RESOLVES``.

    Built from the corpus module names rather than a full parse: the gate only
    needs *a* vocabulary to check against, and keeping it independent of the
    resolver means this gate is not silently testing the resolver instead.
    """
    names: set[str] = set()
    for path in paths:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        names.add(stem)
    return frozenset(names)
