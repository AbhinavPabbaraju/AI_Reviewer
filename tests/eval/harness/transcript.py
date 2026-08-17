"""What the reviewer says about each labeled PR, fixed in advance.

The harness needs model output that is deterministic, free, and *fallible*. A
transcript supplies all three: 31 hand-authored claims across the 30 cases,
rendered into the response envelope Stage V expects and replayed through
``RecordedLLM`` with no provider behind it.

**Why a transcript and not a live model.** A live model makes the harness
non-deterministic, non-free, and impossible to regression-test: a precision
change would be indistinguishable from the model having a different afternoon.
Fixing the transcript makes precision a function of the pipeline alone, which is
what the M6 CI gate needs it to be — the same reason ``RecordedLLM`` exists.

**Why the transcript is deliberately imperfect.** It misses six of the twenty
seeded defects, invents symbols, cites lines past the end of the file, repeats
itself, and says three confident wrong things that are anchored on genuinely
changed lines and name genuinely existing symbols. That last group is the
important one: the verification gate is a mechanism for catching *checkable*
falsehoods, and a claim that is merely wrong is not checkable. A transcript
whose every false claim was catchable would measure a gate that does not exist,
and would report a precision of 1.0 that no real system will ever see.

**What a claim may not do.** Nothing here declares whether it is true. Truth is
decided in ``metrics`` by comparing the *posted* comment against the label in
``labeled_prs``, which was written from the source. If a claim could assert its
own correctness the harness would be scoring the transcript against itself.

Symbols named in backticks are real symbols from the corpora, checked against
the indexed symbol table — with one deliberate exception, noted at the case that
carries it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.contracts import Category, Severity
from tests.eval.corpus.labeled_prs import LabeledPR

__all__ = [
    "REVIEWER_V1",
    "Claim",
    "Transcript",
    "render_response",
    "with_extra_speculation",
]


@dataclass(frozen=True, slots=True)
class Claim:
    """One finding the reviewer emits, exactly as far as a model is trusted.

    Mirrors ``DraftFinding``: no id, no source, no verification status. Those
    belong to the pipeline, and a transcript that could set them would be
    testing something other than the pipeline.
    """

    title: str
    explanation: str
    severity: Severity = Severity.HIGH
    category: Category = Category.CORRECTNESS
    confidence: float = 0.85
    line: int | None = None
    """``None`` anchors on the case's changed line."""

    line_end: int | None = None
    path: str | None = None
    """``None`` cites the case's changed file. Anything else is out of scope for
    the group under review and is dropped at decode."""

    suggested_fix: str | None = None
    improved_code: str | None = None
    excerpt: str | None = None
    """``None`` quotes the real line at head. A value is a fabricated quote --
    which is what a model that invented the line number would produce."""


@dataclass(frozen=True, slots=True)
class Transcript:
    """A whole reviewer's behaviour over the corpus, under one name."""

    name: str
    claims: Mapping[str, tuple[Claim, ...]]

    def for_case(self, slug: str) -> tuple[Claim, ...]:
        return self.claims.get(slug, ())

    @property
    def total_claims(self) -> int:
        return sum(len(c) for c in self.claims.values())


def render_response(case: LabeledPR, claims: Sequence[Claim]) -> str:
    """Render claims into the JSON envelope ``decode_findings`` parses.

    Evidence excerpts are read out of the head tree, so a claim about a real
    line quotes the real code. A claim about a line that does not exist cannot,
    and falls back to a fabricated quote -- the tell that a model made the line
    number up.
    """
    findings = []
    for claim in claims:
        line = claim.line if claim.line is not None else case.changed_line
        line_end = claim.line_end if claim.line_end is not None else line
        path = claim.path or case.path
        excerpt = claim.excerpt
        if excerpt is None:
            excerpt = case.line_at(line) if path == case.path else ""
        if not excerpt:
            excerpt = "<line as the reviewer claimed to see it>"
        entry: dict[str, object] = {
            "severity": claim.severity.value,
            "category": claim.category.value,
            "title": claim.title,
            "explanation": claim.explanation,
            "path": path,
            "line_start": line,
            "line_end": line_end,
            "confidence": claim.confidence,
            "evidence": [
                {
                    "path": path,
                    "line_start": line,
                    "line_end": line_end,
                    "role": "defect_site",
                    "excerpt": excerpt,
                }
            ],
        }
        if claim.suggested_fix is not None:
            entry["suggested_fix"] = claim.suggested_fix
        if claim.improved_code is not None:
            entry["improved_code"] = claim.improved_code
        findings.append(entry)
    return json.dumps({"findings": findings})


# --------------------------------------------------------------------------- #
# reviewer/v1 -- the transcript the gate is measured on
# --------------------------------------------------------------------------- #

_V1: Mapping[str, tuple[Claim, ...]] = {
    # -- Python ---------------------------------------------------------- #
    "py-01-shipping-boundary": (
        Claim(
            title="Orders at exactly the free-shipping threshold are now charged",
            explanation=(
                "`shipping` compares the subtotal with a strict `>`, so an order "
                "worth exactly `FREE_SHIPPING_THRESHOLD` falls into the 500 "
                "branch. The constant names the amount at which shipping becomes "
                "free, so the boundary case is now charged."
            ),
            confidence=0.88,
        ),
        # Same defect, said again, slightly worse. Models repeat themselves when
        # a hunk is small; NOT_DUPLICATE is what stops the PR getting both.
        Claim(
            title="Orders at exactly the free-shipping threshold are charged 500",
            explanation=(
                "The comparison in `shipping` excludes the threshold value "
                "itself, so an order of exactly that amount pays for shipping."
            ),
            confidence=0.80,
        ),
    ),
    "py-02-truncate-off-by-one": (
        Claim(
            title="Truncated value is one character longer than the limit",
            explanation=(
                "The `~` marker is appended after the slice, so slicing to "
                "`limit` produces limit + 1 characters. `summarize` passes the "
                "caller's limit straight through, so every caller now gets a "
                "string one character over the bound it asked for."
            ),
            confidence=0.90,
        ),
        Claim(
            title="Slice can split a surrogate pair and corrupt the output",
            explanation=(
                "Slicing by index rather than by grapheme can cut a multi-byte "
                "character in half."
            ),
            line=999,
            confidence=0.78,
        ),
    ),
    "py-03-put-skips-validation": (
        Claim(
            title="Entities are persisted without being validated",
            explanation=(
                "`MemoryRepository.put` no longer calls `Entity.validate`, and "
                "it was the only validation on the write path. An entity that "
                "fails its own checks is stored and returned by every later "
                "`get`."
            ),
            confidence=0.90,
        ),
        Claim(
            title="Key normalization is skipped before the dictionary write",
            explanation=(
                "`KeyNormalizer.canonicalize` is not applied to the key here, so "
                "two spellings of the same key produce two entries."
            ),
            confidence=0.80,
        ),
    ),
    "py-04-inverted-email-guard": (
        Claim(
            title="Email validation is inverted and rejects valid addresses",
            explanation=(
                "`AccountService.create` now raises `ValidationFailed` when the "
                "address contains an `@` and accepts it when it does not. The "
                "guard passes exactly the inputs it was written to reject."
            ),
            confidence=0.92,
            suggested_fix="Restore the negation so the guard rejects malformed addresses.",
            improved_code='if "@" not in email  raise ValidationFailed("email" "missing @"',
        ),
    ),
    "py-05-shipping-subtracted": (
        Claim(
            title="Shipping is subtracted from the order total instead of added",
            explanation=(
                "`total` returns `subtotal` minus `shipping`. Shipping is a "
                "charge, so this discounts the customer by the shipping cost and "
                "drives small orders negative."
            ),
            confidence=0.90,
            line_end=26,
            suggested_fix="Add the shipping charge rather than subtracting it.",
            improved_code="    return subtotal(order) + shipping(order)",
        ),
    ),
    "py-06-inverted-none-check": (
        Claim(
            title="require raises NotFound for the items it did find",
            explanation=(
                "`Repository.require` exists to convert a missing item into "
                "`NotFound`. With the condition inverted it raises for every "
                "item it successfully fetched and returns None otherwise."
            ),
            confidence=0.90,
        ),
        Claim(
            title="Docstring does not mention the NotFound path",
            explanation=(
                "The method raises but its documentation does not say so, which "
                "callers have to discover by reading the body."
            ),
            severity=Severity.LOW,
            category=Category.MAINTAINABILITY,
            confidence=0.50,
        ),
    ),
    "py-07-validate-wrong-field": (
        # The real defect is missed. What the reviewer says instead is about a
        # line the PR never touched.
        Claim(
            title="Module does not declare an explicit __all__",
            explanation=(
                "Without an export list, `from shop.models import *` pulls in "
                "the imported names as well as the entities."
            ),
            severity=Severity.MEDIUM,
            category=Category.MAINTAINABILITY,
            line=1,
            confidence=0.75,
        ),
    ),
    "py-08-inverted-membership": (
        Claim(
            title="get raises NotFound for keys that are present",
            explanation=(
                "The membership test in `MemoryRepository.get` is inverted: a "
                "stored key raises `NotFound`, and a missing key falls through "
                "to a raw KeyError on the line below."
            ),
            confidence=0.90,
        ),
    ),
    "py-09-deadline-accepts-zero": (
        Claim(
            title="deadline(0) returns an already-expired deadline",
            explanation=(
                "`deadline` used to reject a non-positive duration. Zero now "
                "passes the guard and returns `now`, a deadline that has already "
                "expired, so callers cannot tell a misconfiguration from a real "
                "timeout."
            ),
            confidence=0.85,
        ),
    ),
    "py-10-view-drops-shipping": (
        # The real defect is missed; what is said instead is a plausible opinion
        # that no gate can disprove. This is the shape of a surviving false
        # positive, and the reason measured precision is not 1.0.
        Claim(
            title="Monetary amount is exposed as a bare integer",
            explanation=(
                "`order_view` puts an int of minor units on the wire while "
                "`as_decimal` exists for exactly this conversion. Clients have "
                "to know the scale out of band."
            ),
            severity=Severity.MEDIUM,
            category=Category.API_CONTRACT,
            confidence=0.75,
        ),
    ),
    "py-11-discount-always-applies": (
        Claim(
            title="Discount now applies to every user, tagged or not",
            explanation=(
                "The tags attribute defaults to an empty list, which is never "
                "None, so `discount_for` returns 10 for every user. The previous "
                "truthiness test distinguished an empty list from a populated "
                "one; `is not None` does not."
            ),
            confidence=0.88,
        ),
    ),
    "py-12-broad-except": (
        Claim(
            title="Broad except swallows the Conflict raised by cancel",
            explanation=(
                "`OrderService.cancel` raises `Conflict` when a cancellation "
                "cannot proceed. Catching `Exception` here absorbs it, so "
                "`cancel_order` returns None and the caller sees a successful "
                "cancellation that did not happen."
            ),
            confidence=0.85,
        ),
        # A claim about a file this review was not looking at. Dropped at decode,
        # before the gate ever reads a file.
        Claim(
            title="cancel raises unconditionally after checking the invariant",
            explanation=(
                "`OrderService.cancel` raises `Conflict` on every call, so no "
                "cancellation can ever succeed."
            ),
            path="shop/services/orders.py",
            line=28,
            confidence=0.80,
        ),
    ),
    "py-13-clean-summarize-local": (),
    "py-14-clean-dict-clear": (
        Claim(
            title="Rebinding the dict is not the same as clearing it",
            explanation=(
                "Anything holding a reference to the previous dict keeps seeing "
                "the old contents after a rebind, so the two forms differ."
            ),
            severity=Severity.MEDIUM,
            confidence=0.62,
        ),
    ),
    "py-15-clean-explicit-decimal": (),
    "py-16-clean-named-repository": (
        # Confident, well-anchored, names only real symbols -- and wrong about
        # Python's evaluation order. Nothing in the gate can catch this.
        Claim(
            title="Extraction changes the construction order of the dependencies",
            explanation=(
                "Hoisting `make_repository` above the call means it now runs "
                "before `make_accounts`, so `OrderService` is handed a "
                "repository built earlier than the one it used to receive."
            ),
            confidence=0.82,
        ),
    ),
    "py-17-clean-docstring": (),
    "py-18-clean-loop-variable": (
        Claim(
            title="Loop variable name diverges from the parameter it iterates",
            explanation=(
                "`notify_all` takes `users`, so a reader expects the element to "
                "be called user."
            ),
            severity=Severity.LOW,
            category=Category.STYLE,
            confidence=0.60,
        ),
    ),
    # -- TypeScript ------------------------------------------------------ #
    "ts-01-truncate-off-by-one": (
        Claim(
            title="Truncated string exceeds the limit by one character",
            explanation=(
                "`truncate` slices to `limit` and then appends the marker, so "
                "the result is limit + 1 characters. `summarize` forwards the "
                "caller's limit unchanged."
            ),
            confidence=0.90,
        ),
        Claim(
            title="Marker is appended without going through the shared clamp",
            explanation=(
                "`TextGuard.clampLength` is bypassed here, so the marker is not "
                "counted against the budget the rest of the module enforces."
            ),
            confidence=0.80,
        ),
    ),
    "ts-02-deadline-unit": (
        Claim(
            title="deadline adds seconds to a millisecond clock",
            explanation=(
                "`now` returns `Date.now`, which is in milliseconds. Adding a "
                "value expressed in seconds makes every deadline a thousand "
                "times nearer than requested."
            ),
            confidence=0.90,
        ),
    ),
    "ts-03-shipping-boundary": (
        # Misses the real defect and says something confident, checkable-looking
        # and wrong on the same line.
        Claim(
            title="subtotal can return NaN and make this comparison false",
            explanation=(
                "`subtotal` reads the order's total straight through without a "
                "numeric check, so an `Order` constructed from untyped JSON "
                "yields NaN here and the comparison silently takes the paid "
                "branch."
            ),
            confidence=0.80,
        ),
    ),
    "ts-04-put-skips-validation": (
        Claim(
            title="Entities are stored without being validated",
            explanation=(
                "`MemoryRepository.put` no longer calls `Entity.validate` before "
                "writing to the map, so an entity that fails its own validation "
                "is stored and read back."
            ),
            confidence=0.90,
        ),
        Claim(
            title="Entities are stored without validation running first",
            explanation=(
                "The call to `validate` is gone from `put`, so nothing checks "
                "the entity before the map write."
            ),
            confidence=0.78,
        ),
    ),
    "ts-05-inverted-email-guard": (
        Claim(
            title="Email guard is inverted and rejects valid addresses",
            explanation=(
                "`AccountService.create` throws `ValidationFailed` when the "
                "address does contain an `@`, and constructs a `User` when it "
                "does not."
            ),
            confidence=0.92,
        ),
        Claim(
            title="Repository write is not awaited on the success path",
            explanation=(
                "The store call returns a promise that is never awaited, so a "
                "failure is reported as an unhandled rejection."
            ),
            line=999,
            confidence=0.80,
        ),
    ),
    # The real defect is missed and nothing is said at all. A quiet reviewer
    # costs recall and nothing else -- which is the trade ADR-001 accepts.
    "ts-06-inverted-falsy-check": (),
    "ts-07-stale-effect-deps": (
        Claim(
            title="Effect never re-runs when the key prop changes",
            explanation=(
                "The effect reads `key` but the dependency array is empty, so "
                "`useOrders` keeps the orders it fetched for the first key and "
                "never refetches."
            ),
            confidence=0.88,
        ),
    ),
    "ts-08-view-drops-shipping": (
        Claim(
            title="Object literal duplicates the shape of the interface",
            explanation=(
                "`orderView` restates the fields of `OrderView` by hand, so the "
                "two drift apart when a field is added."
            ),
            severity=Severity.LOW,
            category=Category.MAINTAINABILITY,
            confidence=0.60,
        ),
    ),
    "ts-09-clean-for-of": (),
    "ts-10-clean-summarize-local": (),
    "ts-11-clean-named-repository": (
        Claim(
            title="Hoisting the repository changes which instance is shared",
            explanation=(
                "`makeRepository` now runs before `makeAccounts`, so the two may "
                "no longer receive the same instance."
            ),
            severity=Severity.MEDIUM,
            confidence=0.65,
        ),
    ),
    "ts-12-clean-class-doc": (),
}

REVIEWER_V1 = Transcript(name="reviewer/v1", claims=_V1)
"""The transcript the reported numbers are measured on.

One case carries a deliberate wrinkle. ``py-01`` backticks
``FREE_SHIPPING_THRESHOLD`` -- a module-level constant, which the indexer does
not emit as a symbol. The claim is correct and the symbol is real code, but the
``SYMBOL_RESOLVES`` vocabulary cannot see it. It is left in rather than smoothed
out because it is exactly the kind of thing an eval harness exists to surface,
and the harness reports it as a true finding suppressed by the gate.
"""


def with_extra_speculation(transcript: Transcript, *, name: str) -> Transcript:
    """A more trigger-happy reviewer, for demonstrating metric sensitivity.

    The analogue of a prompt edit that tells the model to look harder: one extra
    confident, well-anchored, unfalsifiable comment on every case. Every claim it
    adds passes the gate, because there is nothing in a matter-of-taste comment
    for a mechanism to check -- so precision falls and the drop rate barely
    moves, which is precisely the regression a drop-rate-only alarm would miss.
    """
    speculative = Claim(
        title="Changed block would read better extracted into a helper",
        explanation=(
            "The logic on this line is doing two things at once and would be "
            "clearer split out, so a later reader does not have to hold both in "
            "their head at the same time."
        ),
        severity=Severity.MEDIUM,
        category=Category.MAINTAINABILITY,
        confidence=0.78,
    )
    return Transcript(
        name=name,
        claims={
            slug: (*claims, speculative) for slug, claims in transcript.claims.items()
        },
    )
