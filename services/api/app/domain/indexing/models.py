"""Domain models for Stage I/II of the pipeline: indexing and the symbol graph.

These mirror the ``symbols``, ``symbol_edges``, ``chunks`` and ``files`` tables
in ``db/migrations/0001_init.sql`` and are the language-agnostic currency the
rest of the pipeline speaks. Nothing in this module knows what tree-sitter is:
parsers (infra) *produce* these; the resolver and chunker (domain) *consume*
them. That boundary is what lets the resolver be tested against a hand-built
``ParsedFile`` with no native grammar in the loop.

Confidence is a first-class field on every edge. ARCHITECTURE.md sec. 4.2 is
explicit that Python dynamic dispatch and TypeScript structural typing defeat
exact resolution, and that "pretending otherwise would be the first
hallucination in the pipeline." So an unresolved reference is *kept*, tagged
with its textual name and a low confidence, never silently dropped.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Self

from pydantic import Field, model_validator

from app.domain.base import Frozen
from app.domain.contracts import CodeSpan

__all__ = [
    "Chunk",
    "EdgeKind",
    "Import",
    "Language",
    "ParsedFile",
    "ParsedUnit",
    "Reference",
    "SourceFile",
    "Symbol",
    "SymbolEdge",
    "SymbolKind",
    "confidence",
]


class Language(StrEnum):
    """A language Argus can parse. Scope is deliberately two (ARCHITECTURE sec.
    2): each additional language multiplies grammar, resolver, and corpus work."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"

    @classmethod
    def for_path(cls, path: str) -> Language | None:
        """Best-effort language detection by extension. ``None`` means "not a
        source file we index" -- the caller filters it out before parsing."""
        suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        return _EXTENSION_LANGUAGE.get(suffix)


# TS-family extensions map to TYPESCRIPT; the parser picks the ts vs tsx grammar
# dialect. `.js`/`.jsx` are parsed with the (superset) tsx grammar rather than
# dropped, because a TS project's JS files still define symbols worth indexing.
_EXTENSION_LANGUAGE: Final[dict[str, Language]] = {
    "py": Language.PYTHON,
    "pyi": Language.PYTHON,
    "ts": Language.TYPESCRIPT,
    "mts": Language.TYPESCRIPT,
    "cts": Language.TYPESCRIPT,
    "tsx": Language.TYPESCRIPT,
    "js": Language.TYPESCRIPT,
    "jsx": Language.TYPESCRIPT,
    "mjs": Language.TYPESCRIPT,
    "cjs": Language.TYPESCRIPT,
}


class SymbolKind(StrEnum):
    """Mirrors the ``symbol_kind`` enum in the DDL."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    VARIABLE = "variable"
    MODULE = "module"
    ENUM = "enum"


class EdgeKind(StrEnum):
    """Mirrors the ``edge_kind`` enum in the DDL (ARCHITECTURE sec. 4.2)."""

    DEFINES = "defines"
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    REFERENCES = "references"
    TESTS = "tests"


class confidence:
    """Calibrated-by-construction confidence levels for resolved edges.

    These are not model outputs; they encode how much the *resolver* trusts a
    given resolution, and they are what the retrieval expansion (M2) down-weights
    unresolved neighbours by. Kept as named constants so a change to the
    resolver's trust model is a one-line, reviewable diff.
    """

    EXACT: Final = 1.0
    """Unique resolution: FQN match, imported internal symbol, or `self.method`
    bound to the enclosing class."""

    HEURISTIC_UNIQUE: Final = 0.8
    """Resolved by name and exactly one repo symbol carries that name."""

    HEURISTIC_AMBIGUOUS: Final = 0.5
    """Resolved by name but several symbols share it; the arbitrary pick is
    flagged so retrieval treats it as weak evidence."""

    TEST_HEURISTIC: Final = 0.6
    """`TESTS` edge inferred from a `test_*` name plus import evidence."""

    EXTERNAL: Final = 0.9
    """An import that resolves to a module outside the repo (stdlib / vendor).
    High confidence that it is *external*, which is itself useful signal."""

    UNRESOLVED: Final = 0.3
    """Kept for graph completeness but not resolved to any symbol."""


class SourceFile(Frozen):
    """A file selected for indexing, with the metadata the ``files`` table needs.

    ``blob_sha`` is the git object id of the file's content and is the pivot of
    incremental indexing (ARCHITECTURE sec. 4.1): a file whose blob sha is
    unchanged since the last snapshot is not re-parsed at all.
    """

    path: str = Field(min_length=1, max_length=1024)
    language: Language
    blob_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    size_bytes: int = Field(ge=0)
    line_count: int = Field(ge=0)
    is_test: bool = False
    is_generated: bool = False


class Symbol(Frozen):
    """A definition site: function, method, class, interface, etc.

    ``fqn`` is the stable identity used throughout the graph. Python uses dotted
    module paths (``app.files.FileStore.read``); TypeScript uses
    ``<module-path>::<Nested.Name>`` because TS module ids are filesystem paths,
    not dotted names. The convention differs per language but is unique within a
    repository, which is all the resolver requires.
    """

    fqn: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: SymbolKind
    span: CodeSpan
    language: Language
    signature: str | None = None
    docstring: str | None = None
    is_exported: bool = True
    parent_fqn: str | None = Field(
        default=None,
        description="Enclosing symbol fqn (the class for a method), else None.",
    )

    @property
    def path(self) -> str:
        return self.span.path


class Import(Frozen):
    """A single name brought into a module's namespace.

    Imports do double duty: each yields an ``IMPORTS`` edge, *and* it seeds the
    per-file name-resolution scope the resolver uses to turn a bare ``read()``
    into a real target. That is why ``local_name`` (what the code writes) and the
    target (``module`` + optional ``imported_symbol``) are both retained.
    """

    local_name: str = Field(min_length=1)
    module: str = Field(
        min_length=0,
        description="Module specifier. May be empty for a bare Python relative "
        "import (`from . import x`), where the package comes from `level`.",
    )
    imported_symbol: str | None = None
    is_relative: bool = False
    level: int = Field(default=0, ge=0, description="Leading-dot count for Python "
                       "relative imports; 0 for absolute.")
    line: int = Field(ge=1)

    @model_validator(mode="after")
    def _absolute_import_names_a_module(self) -> Self:
        if not self.is_relative and not self.module:
            raise ValueError("a non-relative import must name a module")
        return self

    @property
    def target_name(self) -> str:
        """The fully-qualified textual target this binding points at, used both
        as the edge's ``dst_unresolved_name`` and as a resolution key."""
        if self.imported_symbol is not None:
            return (
                f"{self.module}.{self.imported_symbol}"
                if self.module
                else self.imported_symbol
            )
        return self.module


class Reference(Frozen):
    """A use site emitted by the parser, *before* resolution.

    A reference names its target textually as written in the source
    (``read``, ``self.read``, ``mod.func``, ``Base``). Turning that into a
    ``SymbolEdge`` with a resolved ``dst_fqn`` is the resolver's job, and is
    where confidence is assigned.
    """

    kind: EdgeKind
    target_name: str = Field(min_length=1)
    from_fqn: str = Field(min_length=1, description="Enclosing symbol, or the "
                          "module fqn for a module-level reference.")
    line: int = Field(ge=1)
    receiver: str | None = Field(
        default=None,
        description="Receiver text for an attribute access: 'self', a module "
        "alias, or a variable name. None for a bare name.",
    )


class ParsedFile(Frozen):
    """Everything a parser extracts from one file: the unit the resolver and
    chunker consume. Language-agnostic by construction."""

    file: SourceFile
    module_fqn: str = Field(min_length=1)
    symbols: tuple[Symbol, ...] = ()
    imports: tuple[Import, ...] = ()
    references: tuple[Reference, ...] = ()


class SymbolEdge(Frozen):
    """A resolved edge in the symbol graph (mirrors ``symbol_edges``).

    Either ``dst_fqn`` (resolved to a repo symbol) or ``dst_unresolved_name``
    (kept textually) is set -- the same invariant the DDL enforces with
    ``edge_has_a_target``. An edge is never both fully anonymous and dropped.
    """

    kind: EdgeKind
    src_fqn: str = Field(min_length=1)
    dst_fqn: str | None = None
    dst_unresolved_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _has_a_target(self) -> Self:
        if self.dst_fqn is None and self.dst_unresolved_name is None:
            raise ValueError(
                "edge must resolve to a symbol (dst_fqn) or retain its textual "
                "target (dst_unresolved_name); a targetless edge is a dropped edge"
            )
        return self

    @property
    def is_resolved(self) -> bool:
        return self.dst_fqn is not None


class Chunk(Frozen):
    """A symbol-boundary chunk of code, content-addressed for incremental reuse.

    ``content_hash`` = sha256(repo_id | path | symbol_fqn | normalized_body).
    Two indexing runs that produce a byte-identical (modulo normalization) chunk
    at the same location produce the same hash, so re-embedding is skipped
    (ARCHITECTURE sec. 4.1). The embedding itself is M2; M1 produces the chunks
    and proves reuse via the hash.
    """

    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    symbol_fqn: str | None = None
    span: CodeSpan
    language: Language
    token_count: int = Field(gt=0)
    content: str = Field(min_length=1)

    @property
    def path(self) -> str:
        return self.span.path


class ParsedUnit(Frozen):
    """The complete, content-addressed indexing output for one file: its parsed
    structure plus its chunks. Both derive from a single blob, so the two are
    cached and reused together -- an unchanged file yields its symbols *and* its
    chunks from cache, and neither is recomputed (ARCHITECTURE sec. 4.1)."""

    parsed: ParsedFile
    chunks: tuple[Chunk, ...] = ()

    @property
    def blob_sha(self) -> str:
        return self.parsed.file.blob_sha
