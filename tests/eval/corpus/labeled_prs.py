"""30 hand-labeled pull requests — the M6 harness's ground truth.

ROADMAP M6 asks for a corpus of *mutation-seeded defects, hand-labeled*. These
are those, at the partial-milestone size: 30 pull requests against the two
retrieval corpora, 20 carrying a real defect and 10 carrying none.

Three properties make this a corpus rather than a fixture.

**The edits are real, and the labels are read from the source.** Every case is an
exact substring replacement into a real corpus file, asserted unique at import
time, diffed with ``difflib`` and parsed by the same parser production uses. The
defect's line number is *derived* from where the replacement landed, never typed
by hand, so a later edit to the corpus cannot silently detach a label from the
code it describes.

**Ten pull requests contain no defect at all.** This is not padding: precision is
the SLO, and a corpus of nothing but defects cannot measure it — every comment
would be arguably on-target and the number would be meaningless. The clean cases
are deliberately the kind of change a reviewer is tempted to comment on anyway
(a named local, a loop rewritten, a docstring), so a trigger-happy reviewer pays
for it here and only here.

**A label is a claim about behaviour, not a keyword.** Each defect carries
``signals`` — terms any correct description of *that* defect must use, written by
reading the code and asking "what would a human reviewer have to say for me to
agree they found it?". The harness matches a posted comment to a defect by
location overlap *and* one of those signals. That is a proxy for human
adjudication, and an imperfect one: it cannot tell a right answer phrased oddly
from a wrong answer phrased luckily. It is written down here rather than hidden
in the scorer so the approximation is arguable.

The rationale on every case is the labeling argument. Where a clean case is
arguably not clean, the argument for the label is stated rather than omitted —
those are the cases where the label is doing work.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.review.diff import ParsedDiff, parse_unified_diff
from tests.eval.corpus.python_corpus import PY_CORPUS, PY_TEST_FILES
from tests.eval.corpus.typescript_corpus import TS_CORPUS, TS_TEST_FILES

__all__ = ["Defect", "LabeledPR", "labeled_prs"]


@dataclass(frozen=True, slots=True)
class Defect:
    """Ground truth: this pull request breaks something, here, in this way."""

    line: int
    """Line at head where a reviewer would leave the comment. Derived from the
    edit, not typed."""

    kind: str
    """Short taxonomy label, for reporting drift by defect class."""

    note: str
    """Why this is a defect, argued from the source."""

    signals: tuple[str, ...]
    """Lowercase terms, any one of which a correct description must contain."""

    def described_by(self, text: str) -> bool:
        lowered = text.lower()
        return any(signal in lowered for signal in self.signals)


@dataclass(frozen=True, slots=True)
class LabeledPR:
    """One pull request: the tree at head, the diff, and the verdict."""

    slug: str
    path: str
    typescript: bool
    base_files: Mapping[str, str]
    test_files: frozenset[str]
    head_files: Mapping[str, str]
    unified_diff: str
    diff: ParsedDiff
    changed_line: int
    """First line of the edited region at head. The anchor a comment about this
    change would use, defect or not."""

    defect: Defect | None
    rationale: str

    @property
    def is_clean(self) -> bool:
        return self.defect is None

    def line_at(self, line: int) -> str:
        """The source line at head, for evidence excerpts that are real."""
        lines = self.head_files[self.path].splitlines()
        if 1 <= line <= len(lines):
            return lines[line - 1].strip()
        return ""

    @property
    def line_count(self) -> int:
        return len(self.head_files[self.path].splitlines())


# --------------------------------------------------------------------------- #
# Authoring form
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Case:
    """One case as written by hand: an edit, and what it does or does not do."""

    slug: str
    path: str
    find: str
    replace: str
    rationale: str
    defect: tuple[str, str, tuple[str, ...]] | None = None
    """``(kind, note, signals)`` — the line is filled in from the edit."""


# --------------------------------------------------------------------------- #
# Python: 12 defects, 6 clean
# --------------------------------------------------------------------------- #

_PYTHON: tuple[_Case, ...] = (
    _Case(
        slug="py-01-shipping-boundary",
        path="shop/services/pricing.py",
        find="    return 0 if subtotal(order) >= FREE_SHIPPING_THRESHOLD else 500",
        replace="    return 0 if subtotal(order) > FREE_SHIPPING_THRESHOLD else 500",
        defect=(
            "off-by-one",
            "FREE_SHIPPING_THRESHOLD is the amount at which shipping becomes "
            "free. With `>` an order of exactly 5000 is charged 500 shipping, so "
            "the documented threshold is off by one cent at the boundary.",
            ("threshold", "boundary", "exactly", "free shipping"),
        ),
        rationale="A strict comparison excludes the boundary the constant names.",
    ),
    _Case(
        slug="py-02-truncate-off-by-one",
        path="shop/util/text.py",
        find='    return value if len(value) <= limit else value[: limit - 1] + "~"',
        replace='    return value if len(value) <= limit else value[:limit] + "~"',
        defect=(
            "off-by-one",
            "The `~` marker takes a character, which is why the slice stopped at "
            "limit - 1. Slicing to `limit` and appending returns limit + 1 "
            "characters, one over what the caller asked for.",
            ("limit", "one character", "longer", "off-by-one", "too long"),
        ),
        rationale="The function's contract is a maximum length; it now exceeds it.",
    ),
    _Case(
        slug="py-03-put-skips-validation",
        path="shop/store/memory.py",
        find=(
            "    def put(self, item: Entity) -> None:\n"
            "        item.validate()\n"
            "        self._items[item.key()] = item"
        ),
        replace=(
            "    def put(self, item: Entity) -> None:\n"
            "        self._items[item.key()] = item"
        ),
        defect=(
            "dropped-check",
            "MemoryRepository.put was the only place entities were validated "
            "before storage. Without the call an entity failing its own "
            "`validate` is persisted and every later `get` returns it.",
            ("validate", "validation", "unvalidated"),
        ),
        rationale="The store's one invariant check is gone; nothing else re-checks.",
    ),
    _Case(
        slug="py-04-inverted-email-guard",
        path="shop/services/accounts.py",
        find='        if "@" not in email:',
        replace='        if "@" in email:',
        defect=(
            "inverted-condition",
            "The guard now rejects every well-formed address and accepts every "
            "malformed one: `create` raises ValidationFailed exactly when the "
            "email is valid.",
            ("inverted", "backwards", "reversed", "valid email", "negat"),
        ),
        rationale="Dropping `not` inverts the validation, which is the opposite check.",
    ),
    _Case(
        slug="py-05-shipping-subtracted",
        path="shop/services/pricing.py",
        find="    return subtotal(order) + shipping(order)",
        replace="    return subtotal(order) - shipping(order)",
        defect=(
            "wrong-operator",
            "Shipping is a charge, so the order total must add it. Subtracting "
            "discounts the customer by the shipping cost and can drive a small "
            "order's total negative.",
            ("subtract", "minus", "added", "shipping cost", "sign"),
        ),
        rationale="A cost subtracted instead of added is a money bug, not a style one.",
    ),
    _Case(
        slug="py-06-inverted-none-check",
        path="shop/store/base.py",
        find="        if item is None:",
        replace="        if item is not None:",
        defect=(
            "inverted-condition",
            "`require` exists to turn a missing item into NotFound. Inverted, it "
            "raises NotFound for every item it successfully found and returns "
            "None when there is nothing there.",
            ("inverted", "backwards", "reversed", "not none", "raises when"),
        ),
        rationale="The guard's whole purpose is reversed by the added `not`.",
    ),
    _Case(
        slug="py-07-validate-wrong-field",
        path="shop/models.py",
        find='        if "@" not in self.email:',
        replace='        if "@" not in self.id:',
        defect=(
            "wrong-field",
            "User.validate now checks the id, which AccountService builds with "
            "`slugify(email)` and which therefore never contains an `@`. Every "
            "user fails validation, and the email is never checked at all.",
            ("id", "wrong field", "email is", "instead of"),
        ),
        rationale="The check moved to a field that is a slug by construction.",
    ),
    _Case(
        slug="py-08-inverted-membership",
        path="shop/store/memory.py",
        find="        if key not in self._items:",
        replace="        if key in self._items:",
        defect=(
            "inverted-condition",
            "`get` now raises NotFound for keys that are present and falls "
            "through to a raw KeyError for keys that are not.",
            ("inverted", "backwards", "reversed", "keyerror", "present"),
        ),
        rationale="Membership test inverted; both branches are now wrong.",
    ),
    _Case(
        slug="py-09-deadline-accepts-zero",
        path="shop/util/timing.py",
        find="    if seconds <= 0:",
        replace="    if seconds < 0:",
        defect=(
            "boundary",
            "`deadline(0)` now returns `now()`, a deadline that has already "
            "expired, instead of raising ShopError. Callers cannot distinguish "
            "'no time left' from a configuration mistake.",
            ("zero", "0", "already expired", "boundary", "immediately"),
        ),
        rationale="Zero was rejected deliberately; it now produces an expired deadline.",
    ),
    _Case(
        slug="py-10-view-drops-shipping",
        path="shop/api/schemas.py",
        find="        total=pricing.total(order),",
        replace="        total=pricing.subtotal(order),",
        defect=(
            "wrong-callee",
            "OrderView.total is what the customer is shown. `subtotal` excludes "
            "shipping, so the displayed total is now less than what "
            "`pricing.total` will charge.",
            ("shipping", "subtotal", "excludes", "charged"),
        ),
        rationale="The view and the charge disagree; only one of them includes shipping.",
    ),
    _Case(
        slug="py-11-discount-always-applies",
        path="shop/services/pricing.py",
        find="    return 10 if user.tags else 0",
        replace="    return 10 if user.tags is not None else 0",
        defect=(
            "truthiness",
            "`User.tags` is `field(default_factory=list)`, so it is an empty "
            "list, never None. Testing `is not None` grants the discount to "
            "every user including those with no tags at all.",
            ("empty list", "is not none", "every user", "always", "default"),
        ),
        rationale="An empty list is falsy but not None; the two tests are not equivalent.",
    ),
    _Case(
        slug="py-12-broad-except",
        path="shop/api/routes.py",
        find="    except NotFound:",
        replace="    except Exception:",
        defect=(
            "over-broad-except",
            "OrderService.cancel raises Conflict on a cancellation that cannot "
            "proceed. Catching Exception swallows that Conflict, so the handler "
            "reports success for a cancellation that did not happen.",
            ("conflict", "swallow", "broad", "bare except", "hides"),
        ),
        rationale=(
            "The widened clause catches the exception the callee actually raises, "
            "which is visible only by reading shop/services/orders.py."
        ),
    ),
    # -- clean ------------------------------------------------------------- #
    _Case(
        slug="py-13-clean-summarize-local",
        path="shop/util/text.py",
        find="    return truncate(slugify(value), limit)",
        replace="    slug = slugify(value)\n    return truncate(slug, limit)",
        rationale=(
            "A named intermediate. Same two calls, same order, same result; "
            "nothing observable changes."
        ),
    ),
    _Case(
        slug="py-14-clean-dict-clear",
        path="shop/store/memory.py",
        find="    def clear(self) -> None:\n        self._items = {}",
        replace="    def clear(self) -> None:\n        self._items.clear()",
        rationale=(
            "Arguably not clean, and that is why it is here. Rebinding versus "
            "mutating differs only if something else holds a reference to the "
            "old dict; `_items` is private and never handed out anywhere in the "
            "corpus, so the two are equivalent for this class. A comment here is "
            "a comment about a hazard that does not exist in this code."
        ),
    ),
    _Case(
        slug="py-15-clean-explicit-decimal",
        path="shop/services/pricing.py",
        find="    return Decimal(amount) / 100",
        replace="    return Decimal(amount) / Decimal(100)",
        rationale=(
            "Decimal already coerces an int operand exactly; the explicit form "
            "is the same arithmetic written out. No rounding behaviour changes."
        ),
    ),
    _Case(
        slug="py-16-clean-named-repository",
        path="shop/api/deps.py",
        find="    return OrderService(make_repository(settings), make_accounts(settings))",
        replace=(
            "    repository = make_repository(settings)\n"
            "    return OrderService(repository, make_accounts(settings))"
        ),
        rationale=(
            "Python evaluates arguments left to right, so `make_repository` ran "
            "first before this change and runs first after it. The refactor is "
            "an extraction, not a reordering — a reviewer claiming otherwise is "
            "wrong about the language."
        ),
    ),
    _Case(
        slug="py-17-clean-docstring",
        path="shop/config.py",
        find=(
            "def load_settings() -> Settings:\n"
            '    dsn = os.environ.get("SHOP_DSN", DEFAULT_DSN)'
        ),
        replace=(
            "def load_settings() -> Settings:\n"
            '    """Read settings from the environment, falling back to the default DSN."""\n'
            '    dsn = os.environ.get("SHOP_DSN", DEFAULT_DSN)'
        ),
        rationale="Documentation only. No executable line changed.",
    ),
    _Case(
        slug="py-18-clean-loop-variable",
        path="shop/services/notifications.py",
        find="    for user in users:\n        notifier.send(user, body)",
        replace="    for recipient in users:\n        notifier.send(recipient, body)",
        rationale=(
            "A local rename, consistently applied within the loop that binds it. "
            "The name is not read anywhere else in the module."
        ),
    ),
)


# --------------------------------------------------------------------------- #
# TypeScript: 8 defects, 4 clean
# --------------------------------------------------------------------------- #

_TYPESCRIPT: tuple[_Case, ...] = (
    _Case(
        slug="ts-01-truncate-off-by-one",
        path="src/util/text.ts",
        find='  return value.length <= limit ? value : value.slice(0, limit - 1) + "~";',
        replace='  return value.length <= limit ? value : value.slice(0, limit) + "~";',
        defect=(
            "off-by-one",
            "The appended `~` occupies a character. Slicing to `limit` and then "
            "appending returns a string of limit + 1 characters, breaking the "
            "bound the parameter names.",
            ("limit", "one character", "longer", "off-by-one", "too long"),
        ),
        rationale="Mirror of the Python case; the same contract is broken the same way.",
    ),
    _Case(
        slug="ts-02-deadline-unit",
        path="src/util/timing.ts",
        find="export const deadline = (seconds: number): number => now() + seconds * 1000;",
        replace="export const deadline = (seconds: number): number => now() + seconds;",
        defect=(
            "unit-mismatch",
            "`now()` returns `Date.now()`, which is milliseconds. Adding a value "
            "in seconds produces a deadline a thousand times nearer than asked "
            "for — `deadline(60)` expires 60 ms from now.",
            ("millisecond", "unit", "seconds", "1000", "date.now"),
        ),
        rationale="The conversion that reconciled two units was removed.",
    ),
    _Case(
        slug="ts-03-shipping-boundary",
        path="src/services/pricing.ts",
        find="  return subtotal(order) >= FREE_SHIPPING ? 0 : 500;",
        replace="  return subtotal(order) > FREE_SHIPPING ? 0 : 500;",
        defect=(
            "off-by-one",
            "An order of exactly FREE_SHIPPING is charged shipping, though the "
            "constant names the amount at which shipping becomes free.",
            ("threshold", "boundary", "exactly", "free shipping"),
        ),
        rationale="Strict comparison excludes the boundary the constant defines.",
    ),
    _Case(
        slug="ts-04-put-skips-validation",
        path="src/store/memory.ts",
        find=(
            "  put(item: Entity): void {\n"
            "    item.validate();\n"
            "    this.items.set(item.key(), item);\n"
            "  }"
        ),
        replace=(
            "  put(item: Entity): void {\n"
            "    this.items.set(item.key(), item);\n"
            "  }"
        ),
        defect=(
            "dropped-check",
            "MemoryRepository.put was where Entity.validate ran. Without it an "
            "entity that fails its own validation is stored and read back.",
            ("validate", "validation", "unvalidated"),
        ),
        rationale="The only validation call on the write path is gone.",
    ),
    _Case(
        slug="ts-05-inverted-email-guard",
        path="src/services/accounts.ts",
        find='    if (!email.includes("@")) {',
        replace='    if (email.includes("@")) {',
        defect=(
            "inverted-condition",
            "The guard now throws ValidationFailed for every valid address and "
            "lets every invalid one through to `new User(...)`.",
            ("inverted", "backwards", "reversed", "valid email", "negat"),
        ),
        rationale="Dropping `!` inverts the validation.",
    ),
    _Case(
        slug="ts-06-inverted-falsy-check",
        path="src/store/base.ts",
        find="    if (!item) {",
        replace="    if (item) {",
        defect=(
            "inverted-condition",
            "`require` now throws NotFound whenever it did find the item, and "
            "returns the falsy value when it did not.",
            ("inverted", "backwards", "reversed", "throws when", "found"),
        ),
        rationale="The negation that made this a guard was removed.",
    ),
    _Case(
        slug="ts-07-stale-effect-deps",
        path="src/hooks/useOrders.ts",
        find="  }, [key]);",
        replace="  }, []);",
        defect=(
            "stale-closure",
            "The effect closes over `key`. With an empty dependency array it "
            "runs once with the first key and never re-runs, so the hook keeps "
            "returning orders for a key the caller has already changed.",
            ("dependency", "deps", "stale", "re-run", "rerun", "closure"),
        ),
        rationale="A React effect reading a prop must list it; the array no longer does.",
    ),
    _Case(
        slug="ts-08-view-drops-shipping",
        path="src/api/schemas.ts",
        find="    total: pricing.total(order),",
        replace="    total: pricing.subtotal(order),",
        defect=(
            "wrong-callee",
            "OrderView.total is the number rendered to the customer. `subtotal` "
            "omits shipping, so the view under-reports what `pricing.total` "
            "charges.",
            ("shipping", "subtotal", "excludes", "charged"),
        ),
        rationale="The wire shape and the charged amount no longer agree.",
    ),
    # -- clean ------------------------------------------------------------- #
    _Case(
        slug="ts-09-clean-for-of",
        path="src/services/notifications.ts",
        find="  users.forEach((user) => notifier.send(user, body));",
        replace=(
            "  for (const user of users) {\n"
            "    notifier.send(user, body);\n"
            "  }"
        ),
        rationale=(
            "`forEach` and `for...of` differ on sparse arrays and on `await` "
            "inside the body. Neither applies: `users` is a `User[]` built by "
            "callers with array literals, and the body is synchronous. "
            "Equivalent here."
        ),
    ),
    _Case(
        slug="ts-10-clean-summarize-local",
        path="src/util/text.ts",
        find="  return truncate(slugify(value), limit);",
        replace=(
            "  const slug = slugify(value);\n  return truncate(slug, limit);"
        ),
        rationale="A named intermediate; same calls in the same order.",
    ),
    _Case(
        slug="ts-11-clean-named-repository",
        path="src/api/deps.ts",
        find="  return new OrderService(makeRepository(), makeAccounts());",
        replace=(
            "  const repository = makeRepository();\n"
            "  return new OrderService(repository, makeAccounts());"
        ),
        rationale=(
            "JavaScript evaluates arguments left to right, so `makeRepository` "
            "ran first both before and after. An extraction, not a reordering."
        ),
    ),
    _Case(
        slug="ts-12-clean-class-doc",
        path="src/store/cache.ts",
        find="export class CacheRepository extends BaseRepository {",
        replace=(
            "/** Wraps another repository, stamping every write with its time. */\n"
            "export class CacheRepository extends BaseRepository {"
        ),
        rationale="Documentation only.",
    ),
)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


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


def _build(
    case: _Case,
    corpus: Mapping[str, str],
    test_files: frozenset[str],
    *,
    typescript: bool,
) -> LabeledPR:
    before = corpus.get(case.path)
    if before is None:
        raise ValueError(f"{case.slug}: {case.path} is not in the corpus")

    occurrences = before.count(case.find)
    if occurrences != 1:
        # Ambiguity here would silently relabel a case: `replace` would land on
        # a line the rationale is not about, and the defect's line number --
        # which every downstream metric is anchored on -- would be wrong.
        raise ValueError(
            f"{case.slug}: {case.find!r} occurs {occurrences} times in "
            f"{case.path}, expected exactly 1"
        )

    offset = before.index(case.find)
    changed_line = before.count("\n", 0, offset) + 1
    after = before.replace(case.find, case.replace)
    if after == before:
        raise ValueError(f"{case.slug}: the edit changed nothing")

    diff_text = _unified(case.path, before, after)
    defect = None
    if case.defect is not None:
        kind, note, signals = case.defect
        defect = Defect(line=changed_line, kind=kind, note=note, signals=signals)

    return LabeledPR(
        slug=case.slug,
        path=case.path,
        typescript=typescript,
        base_files=corpus,
        test_files=test_files,
        head_files={**corpus, case.path: after},
        unified_diff=diff_text,
        diff=parse_unified_diff(diff_text),
        changed_line=changed_line,
        defect=defect,
        rationale=case.rationale,
    )


def labeled_prs() -> tuple[LabeledPR, ...]:
    """The corpus, built and validated. Deterministic in content and order."""
    built = [
        *(_build(c, PY_CORPUS, PY_TEST_FILES, typescript=False) for c in _PYTHON),
        *(_build(c, TS_CORPUS, TS_TEST_FILES, typescript=True) for c in _TYPESCRIPT),
    ]
    _check(built)
    return tuple(built)


def _check(cases: Sequence[LabeledPR]) -> None:
    """Invariants that would otherwise fail as a confusing metric, not an error."""
    slugs = [case.slug for case in cases]
    if len(set(slugs)) != len(slugs):
        raise ValueError("duplicate case slugs")

    for case in cases:
        touched = case.diff.touched_lines(case.path)
        if not touched:
            raise ValueError(f"{case.slug}: the diff touches no line")
        if case.defect is not None and case.defect.line not in touched:
            # A defect outside the diff is unreachable: the LINE_IN_RANGE gate
            # would reject any comment on it, so the case could only ever score
            # as a miss and would quietly depress recall for a reason that has
            # nothing to do with the reviewer.
            raise ValueError(
                f"{case.slug}: defect line {case.defect.line} is outside the "
                f"lines this PR touches"
            )
