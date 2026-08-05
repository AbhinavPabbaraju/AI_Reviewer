"""Stage VI -- the verification gate. The load-bearing component.

ARCHITECTURE sec. 4.6: *"Never hallucinate" is not a prompt instruction. It is a
mechanism.* Everything a model emits is treated as a claim about a repository
that this module then checks against the repository. Nothing reaches a pull
request without passing.

The seven gates, their checks and their outcomes are specified in sec. 4.6 and
implemented one method each below. Two properties of the design matter more than
any individual check:

**Hard gates reject; soft gates demote.** A finding citing a file that does not
exist is not salvageable and is dropped. A finding whose *patch* does not parse
still contains prose a human can act on, so the patch is stripped and the prose
survives. Conflating the two would either post fabrications or throw away real
bugs over a formatting detail.

**Order is load-bearing.** Cheap hard rejections run first, so a finding about a
nonexistent file never reaches the parser. Dedup runs after per-finding checks,
because merging a finding that is about to be dropped anyway wastes the survivor's
identity. ``CONFIDENCE_FLOOR`` runs last, because demotions lower confidence and a
finding demoted twice should be measured against the floor *after* both.

Every drop is counted per gate. ``verification_drop_rate`` is an unfakeable
measure of model reliability -- it is how a bad prompt change is detected before
users see it -- so the report carries counts rather than just survivors.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from app.domain.base import Frozen
from app.domain.contracts import (
    Finding,
    Severity,
    VerificationGate,
    VerificationResult,
    VerificationStatus,
)
from app.domain.indexing.models import Language
from app.domain.review.diff import ParsedDiff
from app.domain.review.ports import HeadFilePort, SyntaxCheckerPort

__all__ = [
    "DEFAULT_POLICY",
    "VerificationPolicy",
    "VerificationReport",
    "Verifier",
]


# Backticked spans in the explanation are the reliable signal for "the model is
# naming a symbol". Prose is not scanned: an unquoted word like "the store" is
# not a claim about an identifier, and treating it as one would demote every
# well-written explanation.
_BACKTICKED: Final = re.compile(r"`([^`\n]{1,200})`")

# What counts as a symbol-shaped token once the backticks are off: an identifier,
# optionally dotted or `::`-separated (the TypeScript fqn convention), optionally
# called. Anything else in backticks is prose, a flag, a literal or a path.
_SYMBOL_TOKEN: Final = re.compile(
    r"^[A-Za-z_$][A-Za-z0-9_$]*(?:(?:\.|::)[A-Za-z_$][A-Za-z0-9_$]*)*(?:\(\))?$"
)

# Names that resolve to a language, not to this repository. Checking them
# against the symbol table would demote correct findings for naming `len`.
_UNIVERSAL_NAMES: Final = frozenset(
    {
        "none", "true", "false", "null", "undefined", "self", "this", "super",
        "int", "str", "float", "bool", "bytes", "list", "dict", "set", "tuple",
        "object", "type", "len", "range", "print", "open", "map", "filter",
        "number", "string", "boolean", "array", "promise", "record", "partial",
        "any", "unknown", "void", "never", "await", "async", "return", "if",
        "else", "for", "while", "try", "except", "catch", "finally", "raise",
        "throw", "import", "export", "class", "def", "function", "const", "let",
        "var", "new", "delete", "typeof", "instanceof", "in", "is", "not",
        "and", "or", "assert", "yield", "lambda", "pass", "continue", "break",
        "todo", "fixme", "note", "warning", "error", "exception", "valueerror",
        "typeerror", "keyerror", "indexerror", "runtimeerror", "nameerror",
        "attributeerror", "notimplementederror", "zerodivisionerror",
    }
)


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Tunable thresholds. Defaults are precision-first, per ADR-001."""

    confidence_floor: Mapping[Severity, float] = field(
        default_factory=lambda: {
            # A low-severity comment has to be *very* likely correct to be worth
            # a reviewer's attention at all; a critical one is worth surfacing at
            # lower confidence because the cost of missing it is asymmetric.
            Severity.CRITICAL: 0.50,
            Severity.HIGH: 0.60,
            Severity.MEDIUM: 0.70,
            Severity.LOW: 0.75,
            Severity.INFO: 0.80,
        }
    )
    demotion_penalty: Mapping[VerificationGate, float] = field(
        default_factory=lambda: {
            # What each soft failure actually says about the finding's truth.
            # A model naming a symbol that exists nowhere in the repository is
            # evidence it invented part of its reasoning, so the penalty is
            # heavy. A patch that does not parse says the *patch* is wrong and
            # leaves the prose untouched. A patch that cannot be rendered as a
            # GitHub suggestion says nothing about correctness at all -- it is a
            # limitation of where the comment lands -- so it costs almost
            # nothing.
            #
            # A single flat penalty was the first implementation, and it made
            # sec. 4.6's "strip the patch, keep the prose" a fiction: one
            # demotion pushed a typical finding under its severity floor, so the
            # prose that was supposed to survive was suppressed a step later.
            VerificationGate.SYMBOL_RESOLVES: 0.30,
            VerificationGate.PATCH_PARSES: 0.10,
            VerificationGate.PATCH_APPLIES: 0.05,
        }
    )
    default_demotion_penalty: float = 0.30
    duplicate_similarity: float = 0.70
    """Normalized-title similarity above which two overlapping findings are the
    same finding. Paired with span overlap, never used alone -- two different
    bugs in one function often have similar titles."""

    allow_context_lines: bool = True
    """Whether a finding may anchor on an unchanged line inside a hunk. True
    because a reviewer *can* see those lines and a bug is often on the guard
    above the new call; the changed-line set alone would drop those."""


DEFAULT_POLICY: Final = VerificationPolicy()


class VerificationReport(Frozen):
    """The outcome for a whole run, survivors and casualties alike."""

    findings: tuple[Finding, ...]
    """Every finding, each carrying its own verification result. Rejected
    findings are retained (never posted, still stored) so the dashboard can show
    what was suppressed and why."""

    drops_by_gate: Mapping[VerificationGate, int] = {}

    @property
    def postable(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.verification.is_postable)

    @property
    def rejected(self) -> tuple[Finding, ...]:
        return tuple(
            f
            for f in self.findings
            if f.verification.status is VerificationStatus.REJECTED
        )

    @property
    def drop_rate(self) -> float:
        """Fraction of findings the gate refused to post. The metric sec. 4.6
        calls a direct, unfakeable measure of model reliability."""
        if not self.findings:
            return 0.0
        return len(self.rejected) / len(self.findings)


@dataclass(frozen=True, slots=True)
class VerificationContext:
    """Everything the gate checks findings *against*."""

    diff: ParsedDiff
    known_symbols: frozenset[str] = frozenset()
    """Symbol fqns and bare names from the symbol table and the context pack.
    Empty disables ``SYMBOL_RESOLVES`` rather than failing every finding -- a
    gate with nothing to check against must not manufacture verdicts."""


class Verifier:
    """Runs the seven gates. One instance per run; ``verify`` is the whole API."""

    def __init__(
        self,
        *,
        files: HeadFilePort,
        syntax: SyntaxCheckerPort | None = None,
        policy: VerificationPolicy = DEFAULT_POLICY,
    ) -> None:
        self._files = files
        self._syntax = syntax
        self._policy = policy

    async def verify(
        self, findings: Sequence[Finding], context: VerificationContext
    ) -> VerificationReport:
        drops: dict[VerificationGate, int] = {}
        cache: dict[str, str | None] = {}
        checked: list[Finding] = []

        for finding in findings:
            checked.append(await self._verify_one(finding, context, cache, drops))

        # Set-level gates, in order: dedup needs survivors, and the floor needs
        # the confidence left after every demotion above.
        deduped = self._gate_not_duplicate(checked, drops)
        floored = [self._gate_confidence_floor(f, drops) for f in deduped]
        return VerificationReport(findings=tuple(floored), drops_by_gate=drops)

    # -- per-finding gates ------------------------------------------------- #

    async def _verify_one(
        self,
        finding: Finding,
        context: VerificationContext,
        cache: dict[str, str | None],
        drops: dict[VerificationGate, int],
    ) -> Finding:
        source = await self._read(finding.location.path, cache)

        # FILE_EXISTS -- drop. Nothing downstream is meaningful without the file,
        # so this short-circuits rather than accumulating further gate failures
        # against a finding that is already gone.
        if source is None:
            return self._reject(
                finding,
                VerificationGate.FILE_EXISTS,
                drops,
                f"{finding.location.path} does not exist at head",
            )

        # LINE_IN_RANGE -- drop.
        problem = self._line_problem(finding, source, context)
        if problem is not None:
            return self._reject(finding, VerificationGate.LINE_IN_RANGE, drops, problem)

        verified = finding

        # SYMBOL_RESOLVES -- demote. Soft because the location has already been
        # proven real; an unrecognised name in the prose lowers trust in the
        # reasoning, it does not disprove the bug.
        unresolved = self._unresolved_symbols(verified, context)
        if unresolved:
            verified = self._demote(
                verified, VerificationGate.SYMBOL_RESOLVES, drops
            )

        # PATCH_PARSES -- strip the patch, keep the prose.
        if verified.improved_code is not None and not self._patch_parses(verified):
            verified = self._strip_patch(
                verified, VerificationGate.PATCH_PARSES, drops
            )

        # PATCH_APPLIES -- downgrade to a plain comment. Only meaningful if a
        # patch survived the previous gate.
        if verified.improved_code is not None and not self._patch_applies(
            verified, context
        ):
            verified = self._strip_patch(
                verified, VerificationGate.PATCH_APPLIES, drops
            )

        return _promote(verified)

    def _line_problem(
        self, finding: Finding, source: str, context: VerificationContext
    ) -> str | None:
        """``None`` when the cited span is postable, else why it is not."""
        location = finding.location
        line_count = len(source.splitlines())
        if location.line_end > line_count:
            return (
                f"cites {location} but {location.path} has {line_count} lines "
                "at head"
            )

        file_diff = context.diff.for_path(location.path)
        if file_diff is None:
            # A finding about a file this PR did not touch. Allowed only when it
            # connects back to the change through its own evidence -- "your new
            # caller breaks this" is a real review comment; "this unrelated file
            # has a bug" is noise the author did not cause and cannot act on.
            if self._cites_changed_file(finding, context):
                return None
            return (
                f"{location.path} is not part of this diff and the finding cites "
                "no evidence in a changed file"
            )

        allowed = (
            file_diff.touched_lines
            if self._policy.allow_context_lines
            else file_diff.changed_lines
        )
        cited = set(range(location.line_start, location.line_end + 1))
        if not cited & allowed:
            return (
                f"cites {location}, which touches no line this PR changed "
                f"in {location.path}"
            )
        return None

    def _cites_changed_file(
        self, finding: Finding, context: VerificationContext
    ) -> bool:
        changed = context.diff.paths
        return any(e.span.path in changed for e in finding.evidence)

    def _unresolved_symbols(
        self, finding: Finding, context: VerificationContext
    ) -> list[str]:
        if not context.known_symbols:
            return []
        unresolved: list[str] = []
        for token in _symbol_tokens(finding.explanation):
            if not _resolves(token, context.known_symbols):
                unresolved.append(token)
        return unresolved

    def _patch_parses(self, finding: Finding) -> bool:
        if self._syntax is None:
            return True
        language = Language.for_path(finding.location.path)
        if language is None:
            # Not a language we parse. The gate has no opinion rather than a
            # negative one; stripping every patch on a .md or .yaml file would be
            # a checker asserting something it cannot know.
            return True
        patch = finding.improved_code or ""
        # A suggestion is usually lifted out of an indented block, so it is
        # tried both as written and dedented. Accepting either is the difference
        # between checking syntax and checking indentation.
        return self._syntax.parses(language, patch) or self._syntax.parses(
            language, textwrap.dedent(patch)
        )

    def _patch_applies(self, finding: Finding, context: VerificationContext) -> bool:
        """Whether GitHub could render this as a suggested-change block.

        A suggestion replaces exactly the commented line range, and GitHub only
        accepts one on lines that appear in the diff. A patch anchored outside
        the diff is not wrong -- it just cannot be offered as one click, so it
        degrades to prose.
        """
        file_diff = context.diff.for_path(finding.location.path)
        if file_diff is None:
            return False
        cited = set(
            range(finding.location.line_start, finding.location.line_end + 1)
        )
        return cited <= file_diff.touched_lines

    # -- set-level gates ---------------------------------------------------- #

    def _gate_not_duplicate(
        self, findings: Sequence[Finding], drops: dict[VerificationGate, int]
    ) -> list[Finding]:
        """Merge near-duplicates: overlapping spans *and* similar titles.

        Both conditions are required. Span overlap alone merges two genuinely
        different bugs in one function; title similarity alone merges the same
        class of bug found in two unrelated places.
        """
        result = list(findings)
        # Highest priority first, so the survivor of each merge is the finding a
        # reviewer would rather have seen; ties go to deterministic sources,
        # which cannot have hallucinated their location in the first place.
        order = sorted(
            range(len(result)),
            key=lambda i: (
                -result[i].priority,
                not result[i].source.is_deterministic,
                str(result[i].id),
            ),
        )
        winners: list[int] = []
        for index in order:
            candidate = result[index]
            if candidate.verification.status is VerificationStatus.REJECTED:
                continue
            duplicate_of = next(
                (w for w in winners if _is_duplicate(result[w], candidate,
                                                     self._policy.duplicate_similarity)),
                None,
            )
            if duplicate_of is None:
                winners.append(index)
                continue
            group = result[duplicate_of].fingerprint
            result[duplicate_of] = result[duplicate_of].model_copy(
                update={"dedup_group": group}
            )
            result[index] = candidate.model_copy(
                update={"dedup_group": group}
            ).reject(
                VerificationGate.NOT_DUPLICATE,
                f"merged into {result[duplicate_of].title!r}",
            )
            drops[VerificationGate.NOT_DUPLICATE] = (
                drops.get(VerificationGate.NOT_DUPLICATE, 0) + 1
            )
        return result

    def _gate_confidence_floor(
        self, finding: Finding, drops: dict[VerificationGate, int]
    ) -> Finding:
        if finding.verification.status is VerificationStatus.REJECTED:
            return finding
        floor = self._policy.confidence_floor.get(finding.severity, 0.0)
        if finding.confidence >= floor:
            return finding
        return self._reject(
            finding,
            VerificationGate.CONFIDENCE_FLOOR,
            drops,
            f"confidence {finding.confidence:.2f} below the "
            f"{finding.severity.value} floor of {floor:.2f}",
        )

    # -- outcome helpers ---------------------------------------------------- #

    def _reject(
        self,
        finding: Finding,
        gate: VerificationGate,
        drops: dict[VerificationGate, int],
        note: str,
    ) -> Finding:
        drops[gate] = drops.get(gate, 0) + 1
        return finding.reject(gate, note)

    def _demote(
        self,
        finding: Finding,
        gate: VerificationGate,
        drops: dict[VerificationGate, int],
    ) -> Finding:
        drops[gate] = drops.get(gate, 0) + 1
        penalty = self._policy.demotion_penalty.get(
            gate, self._policy.default_demotion_penalty
        )
        return finding.demote(gate, penalty=penalty)

    def _strip_patch(
        self,
        finding: Finding,
        gate: VerificationGate,
        drops: dict[VerificationGate, int],
    ) -> Finding:
        """Remove the patch, keep the prose, record the demotion."""
        stripped = finding.model_copy(update={"improved_code": None})
        return self._demote(stripped, gate, drops)

    async def _read(self, path: str, cache: dict[str, str | None]) -> str | None:
        if path not in cache:
            cache[path] = await self._files.read(path)
        return cache[path]


# -- pure helpers ---------------------------------------------------------- #


def _promote(finding: Finding) -> Finding:
    """Mark a finding that cleared every gate as VERIFIED.

    Findings arrive PENDING, and ``VerificationResult.is_postable`` deliberately
    excludes PENDING -- "not yet checked" must never be mistaken for "checked and
    fine". Something therefore has to state the positive result explicitly, and
    it has to happen here rather than being assumed downstream: a publisher that
    treated PENDING as postable would post unverified model output, which is the
    one outcome this whole module exists to prevent.
    """
    if finding.verification.status is not VerificationStatus.PENDING:
        return finding
    return finding.model_copy(
        update={"verification": VerificationResult(status=VerificationStatus.VERIFIED)}
    )


def _symbol_tokens(explanation: str) -> list[str]:
    """Backticked tokens from the prose that claim to be symbols."""
    tokens: list[str] = []
    for raw in _BACKTICKED.findall(explanation):
        token = raw.strip().removesuffix("()")
        if not token or not _SYMBOL_TOKEN.match(raw.strip()):
            continue
        if token.lower() in _UNIVERSAL_NAMES:
            continue
        if token.lower().split(".")[-1] in _UNIVERSAL_NAMES:
            continue
        tokens.append(token)
    return tokens


def _resolves(token: str, known: frozenset[str]) -> bool:
    """A token resolves if the vocabulary knows it whole or knows its tail.

    The tail check is what lets an explanation say ``FileStore.read`` about a
    symbol whose fqn is ``app.files.FileStore.read``: writing the full dotted
    path in prose would be unreadable, and demanding it would punish good
    explanations.
    """
    if token in known:
        return True
    tail = re.split(r"\.|::", token)[-1]
    return tail in known


def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", title.lower()))


def _title_similarity(left: str, right: str) -> float:
    """Jaccard overlap of the title's word sets.

    Character-level similarity was the obvious choice here and is measurably
    wrong. "Missing null check on user lookup" and "Missing bounds check on
    index lookup" are two different bugs, and they score 0.78 by character
    ratio -- higher than genuine duplicates like "Unnormalized path join allows
    directory traversal" / "Unnormalized path join allows traversal" (0.54),
    because the words that differ happen to share most of their letters. No
    threshold separates those two cases.

    Word sets do separate them: 0.50 for the different bugs against 0.83 for the
    duplicate. The signal that two titles describe one finding is *which words
    they use*, not how the letters line up.
    """
    left_tokens = _title_tokens(left)
    right_tokens = _title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _is_duplicate(left: Finding, right: Finding, threshold: float) -> bool:
    if not left.location.overlaps(right.location):
        return False
    return _title_similarity(left.title, right.title) >= threshold
