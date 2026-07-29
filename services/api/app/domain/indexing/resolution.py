"""Edge resolution: turning textual references into a symbol graph (sec. 4.2).

This is the component ROADMAP.md calls "the hardest correctness work in the
project," and its accuracy is the retrieval ceiling. The design commitment from
ARCHITECTURE sec. 4.2 governs every choice here:

> Resolution is best-effort and explicitly confidence-tagged. Python dynamic
> dispatch and TypeScript structural typing both defeat exact resolution;
> pretending otherwise would be the first hallucination in the pipeline.

So a reference always yields an edge. If it resolves uniquely to a repo symbol,
the edge carries ``dst_fqn`` at full confidence. If it resolves only by name, the
edge is kept at a heuristic confidence. If it does not resolve at all, the edge
keeps its textual target and a floor confidence -- it is *down-weighted, never
dropped*.

The resolver is deliberately language-agnostic: it indexes symbols by
``(module, name)`` and ``(parent, name)`` and never reconstructs a fully-qualified
name by string surgery, so the same code resolves Python and TypeScript. The only
language-specific logic is import-module resolution (dotted packages vs relative
paths), which is isolated in :meth:`_resolve_module`.
"""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.domain.indexing.models import (
    STAR_IMPORT,
    EdgeKind,
    Import,
    Language,
    ParsedFile,
    Reference,
    Symbol,
    SymbolEdge,
    SymbolKind,
    confidence,
)

__all__ = ["ResolutionResult", "ResolutionStats", "Resolver"]

_SELF_RECEIVERS: frozenset[str] = frozenset({"self", "this", "cls"})
_MAX_INHERITANCE_DEPTH = 5
_MAX_REEXPORT_DEPTH = 5
_PACKAGE_INIT = "__init__.py"


@dataclass(frozen=True, slots=True)
class _Resolved:
    """Internal resolution outcome before it becomes a :class:`SymbolEdge`."""

    dst_fqn: str | None
    dst_unresolved_name: str | None
    confidence: float


@dataclass(slots=True)
class _Scope:
    """A module's resolution context: its imports and where it lives on disk.

    Wildcard bindings are kept in a separate list rather than the by-local-name
    map: they all share the local name ``*``, so a barrel with two
    ``export * from`` lines would otherwise keep only the last one.
    """

    module_fqn: str
    language: Language
    path: str
    imports: dict[str, Import] = field(default_factory=dict)
    stars: tuple[Import, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolutionStats:
    """Operational resolution metrics, per the M1 exit criterion.

    ``resolution_rate`` deliberately excludes edges classified *external* (an
    import of a stdlib/vendor module is correctly unresolved, not a failure). It
    is ``resolved / (resolved + unresolved)`` over intra-repo references.
    """

    total_edges: int
    resolved: int
    external: int
    unresolved: int
    by_language: dict[Language, tuple[int, int]]
    """language -> (resolved, unresolved) for intra-repo references."""

    @property
    def resolution_rate(self) -> float:
        denom = self.resolved + self.unresolved
        return self.resolved / denom if denom else 1.0

    def rate_for(self, language: Language) -> float:
        resolved, unresolved = self.by_language.get(language, (0, 0))
        denom = resolved + unresolved
        return resolved / denom if denom else 1.0


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    edges: tuple[SymbolEdge, ...]
    stats: ResolutionStats


class Resolver:
    """Builds a global symbol index from parsed files and resolves references
    against it. Construct once per indexing run; it is stateless after ``__init__``
    apart from the memoized inheritance map."""

    def __init__(self, parsed_files: Sequence[ParsedFile]) -> None:
        self._by_fqn: dict[str, Symbol] = {}
        self._top_level: dict[str, dict[str, Symbol]] = {}
        self._members: dict[str, dict[str, Symbol]] = {}
        self._by_name: dict[str, list[Symbol]] = {}
        self._modules: set[str] = set()
        self._scopes: dict[str, _Scope] = {}
        self._parsed = tuple(parsed_files)

        for parsed in self._parsed:
            self._modules.add(parsed.module_fqn)
            self._scopes[parsed.module_fqn] = _Scope(
                module_fqn=parsed.module_fqn,
                language=parsed.file.language,
                path=parsed.file.path,
                imports={
                    imp.local_name: imp
                    for imp in parsed.imports
                    if imp.local_name != STAR_IMPORT
                },
                stars=tuple(
                    imp for imp in parsed.imports if imp.local_name == STAR_IMPORT
                ),
            )
            for symbol in parsed.symbols:
                self._index_symbol(parsed.module_fqn, symbol)

        # Inheritance is resolved eagerly so `self.method()` can climb to a base
        # class defined in another file.
        self._bases: dict[str, list[str]] = self._resolve_inheritance()

    def _index_symbol(self, module_fqn: str, symbol: Symbol) -> None:
        self._by_fqn[symbol.fqn] = symbol
        self._by_name.setdefault(symbol.name, []).append(symbol)
        if symbol.kind is SymbolKind.MODULE:
            return
        if symbol.parent_fqn is None:
            self._top_level.setdefault(module_fqn, {})[symbol.name] = symbol
        else:
            self._members.setdefault(symbol.parent_fqn, {})[symbol.name] = symbol

    # -- public API ------------------------------------------------------ #

    def resolve(self) -> ResolutionResult:
        """Resolve every import and reference across all files into a deduplicated
        edge set, with resolution statistics."""
        seen: dict[tuple[str, str, str | None, str | None], SymbolEdge] = {}
        for parsed in self._parsed:
            edges = (
                *(self.resolve_import(imp, parsed.module_fqn) for imp in parsed.imports),
                *(
                    self.resolve_reference(ref, parsed.module_fqn)
                    for ref in parsed.references
                ),
            )
            for edge in edges:
                key = (
                    edge.kind.value,
                    edge.src_fqn,
                    edge.dst_fqn,
                    edge.dst_unresolved_name,
                )
                current = seen.get(key)
                if current is None or edge.confidence > current.confidence:
                    seen[key] = edge
        ordered = tuple(seen.values())
        return ResolutionResult(edges=ordered, stats=self._stats(ordered))

    def resolve_reference(self, ref: Reference, module_fqn: str) -> SymbolEdge:
        """Resolve a single use site. Public so the eval harness can score
        resolution accuracy reference-by-reference against a labeled corpus."""
        scope = self._scopes[module_fqn]
        if ref.kind is EdgeKind.TESTS:
            outcome = self._resolve_test(scope, ref)
        else:
            outcome = self._resolve_use(scope, ref)
        return self._edge(ref.kind, ref.from_fqn, outcome)

    def resolve_import(self, imp: Import, module_fqn: str) -> SymbolEdge:
        scope = self._scopes[module_fqn]
        return self._edge(
            EdgeKind.IMPORTS, module_fqn, self._resolve_binding(scope, imp)
        )

    def _resolve_binding(
        self, scope: _Scope, imp: Import, _depth: int = 0
    ) -> _Resolved:
        """Resolve an import binding to a repo symbol, a repo *submodule*, or an
        external target. Shared by import-edge creation and by bare-name
        resolution, so `from . import util` and a later bare `util.x` agree."""
        if imp.imported_symbol is None:
            module = self._resolve_module(scope, imp)  # whole-module import
            if module is not None:
                return _Resolved(module, None, confidence.EXACT)
            return _Resolved(None, imp.target_name, confidence.EXTERNAL)

        base = self._from_module_path(scope, imp)
        if base is not None:
            symbol = self._top_level.get(base, {}).get(imp.imported_symbol)
            if symbol is not None:
                return _Resolved(symbol.fqn, None, confidence.EXACT)
            # `from pkg import submodule` -> the target is the submodule itself.
            submodule = f"{base}.{imp.imported_symbol}"
            if submodule in self._modules:
                return _Resolved(submodule, None, confidence.EXACT)
            if base in self._modules:
                # The module exists but does not define the name: it is almost
                # always re-exporting it (a package `__init__` or a TS barrel).
                # Follow the chain rather than declaring a repo-internal import
                # unresolved -- barrels are how real code is imported.
                reexport = self._follow_reexport(base, imp.imported_symbol, _depth)
                if reexport is not None:
                    return reexport
                return _Resolved(None, imp.target_name, confidence.UNRESOLVED)
        return _Resolved(None, imp.target_name, confidence.EXTERNAL)

    def _follow_reexport(
        self, module_fqn: str, name: str, _depth: int
    ) -> _Resolved | None:
        """Resolve ``name`` through ``module_fqn``'s own imports (one hop of a
        re-export chain), or None if it does not re-export that name.

        Depth-capped: a cyclic barrel (``a`` re-exports from ``b``, ``b`` from
        ``a``) must terminate rather than recurse forever.
        """
        if _depth >= _MAX_REEXPORT_DEPTH:
            return None
        via = self._scopes.get(module_fqn)
        if via is None:
            return None
        imp = via.imports.get(name)
        if imp is not None:
            resolved = self._resolve_binding(via, imp, _depth + 1)
            if resolved.dst_fqn is not None:
                return resolved
        return self._follow_wildcard(via, name, _depth)

    def _follow_wildcard(
        self, via: _Scope, name: str, _depth: int
    ) -> _Resolved | None:
        """Search the modules a wildcard binding pulls in (``export * from``,
        ``from m import *``) for ``name``. Only wildcards are searched: a named
        binding says exactly which name it carries, and rummaging through
        unrelated imported modules for a name they were never asked for is how a
        resolver starts inventing edges."""
        for star in via.stars:
            target = self._resolve_module(via, star)
            if target is None:
                continue
            symbol = self._top_level.get(target, {}).get(name)
            if symbol is not None:
                return _Resolved(symbol.fqn, None, confidence.EXACT)
            chained = self._follow_reexport(target, name, _depth + 1)
            if chained is not None:
                return chained
        return None

    # -- reference resolution -------------------------------------------- #

    def _resolve_use(self, scope: _Scope, ref: Reference) -> _Resolved:
        name = ref.target_name
        receiver = ref.receiver

        if receiver in _SELF_RECEIVERS:
            enclosing = self._enclosing_class(ref.from_fqn)
            if enclosing is not None:
                member = self._lookup_member(enclosing, name)
                if member is not None:
                    return _Resolved(member.fqn, None, confidence.EXACT)
            return self._heuristic(name, prefer_methods=True)

        if receiver is not None:
            typed = self._resolve_typed_receiver(scope, ref.receiver_type, name)
            if typed is not None:
                return typed
            module = self._imported_module(scope, receiver)
            if module is not None:
                symbol = self._top_level.get(module, {}).get(name)
                if symbol is not None:
                    return _Resolved(symbol.fqn, None, confidence.EXACT)
                return _Resolved(None, f"{module}.{name}", confidence.UNRESOLVED)
            if self._is_external_receiver(scope, receiver):
                return _Resolved(None, f"{receiver}.{name}", confidence.EXTERNAL)
            return self._heuristic(
                name,
                prefer_methods=True,
                exclude_members_of=self._self_attribute_owner(ref, receiver),
            )

        return self._resolve_bare(scope, name)

    def _resolve_typed_receiver(
        self, scope: _Scope, type_name: str | None, member: str
    ) -> _Resolved | None:
        """Resolve ``member`` against the class a local receiver was constructed
        from, or None if that is not what the receiver is.

        This is where the parser's ``receiver_type`` hint is *validated*: the
        name must resolve, it must resolve to a class in this repository, and
        that class (or a base) must actually declare the member. Any of those
        failing falls through to the ordinary name heuristics, so a hint that
        names a factory function or an external type costs nothing.
        """
        if type_name is None:
            return None
        candidate = self._resolve_bare(scope, type_name)
        if candidate.dst_fqn is None:
            return None
        symbol = self._by_fqn.get(candidate.dst_fqn)
        if symbol is None or symbol.kind is not SymbolKind.CLASS:
            return None
        found = self._lookup_member(symbol.fqn, member)
        if found is None:
            return None
        return _Resolved(found.fqn, None, confidence.INFERRED_LOCAL)

    def _self_attribute_owner(self, ref: Reference, receiver: str) -> str | None:
        """For a call through an *attribute of self* (``self._conn.execute()``),
        the class whose own members must be excluded from name matching.

        The receiver is a collaborator held by the enclosing instance, so it is
        emphatically not the enclosing instance. Without this, a method that
        happens to share a name with the collaborator's -- ``self._conn.close()``
        inside ``close()`` -- resolves to itself, and the graph grows a
        confident, entirely fabricated self-edge. Fabrication is the failure
        mode this project exists to prevent, so the ambiguity is preserved
        instead (ARCHITECTURE sec. 4.2).
        """
        root, _, rest = receiver.partition(".")
        if not rest or root not in _SELF_RECEIVERS:
            return None
        return self._enclosing_class(ref.from_fqn)

    def _resolve_bare(self, scope: _Scope, name: str) -> _Resolved:
        imp = scope.imports.get(name)
        if imp is not None:
            return self._resolve_binding(scope, imp)

        local = self._top_level.get(scope.module_fqn, {}).get(name)
        if local is not None:
            return _Resolved(local.fqn, None, confidence.EXACT)

        return self._heuristic(name, prefer_methods=False)

    def _resolve_test(self, scope: _Scope, ref: Reference) -> _Resolved:
        """A `TESTS` edge is inferred: the target name (already stripped of its
        `test_` prefix by the parser) is resolved by import evidence or a unique
        global name, at a capped confidence."""
        candidate = ref.target_name
        bare = self._resolve_bare(scope, candidate)
        if bare.dst_fqn is not None:
            return _Resolved(bare.dst_fqn, None, confidence.TEST_HEURISTIC)
        return _Resolved(None, candidate, confidence.UNRESOLVED)

    def _heuristic(
        self,
        name: str,
        *,
        prefer_methods: bool,
        exclude_members_of: str | None = None,
    ) -> _Resolved:
        candidates = self._by_name.get(name, [])
        if exclude_members_of is not None:
            candidates = [s for s in candidates if s.parent_fqn != exclude_members_of]
        if prefer_methods:
            methods = [s for s in candidates if s.kind is SymbolKind.METHOD]
            candidates = methods or candidates
        if not candidates:
            return _Resolved(None, name, confidence.UNRESOLVED)
        if len(candidates) == 1:
            return _Resolved(candidates[0].fqn, None, confidence.HEURISTIC_UNIQUE)
        chosen = min(candidates, key=lambda s: s.fqn)
        return _Resolved(chosen.fqn, None, confidence.HEURISTIC_AMBIGUOUS)

    # -- symbol-table helpers -------------------------------------------- #

    def _enclosing_class(self, from_fqn: str) -> str | None:
        symbol = self._by_fqn.get(from_fqn)
        if symbol is None:
            return None
        if symbol.kind is SymbolKind.CLASS:
            return symbol.fqn
        if symbol.kind is SymbolKind.METHOD and symbol.parent_fqn is not None:
            return symbol.parent_fqn
        return None

    def _lookup_member(
        self, class_fqn: str, name: str, _depth: int = 0
    ) -> Symbol | None:
        member = self._members.get(class_fqn, {}).get(name)
        if member is not None:
            return member
        if _depth >= _MAX_INHERITANCE_DEPTH:
            return None
        for base_fqn in self._bases.get(class_fqn, ()):
            inherited = self._lookup_member(base_fqn, name, _depth + 1)
            if inherited is not None:
                return inherited
        return None

    def _imported_module(self, scope: _Scope, receiver: str) -> str | None:
        """If ``receiver`` is an alias bound to an *internal* module, return that
        module's fqn; otherwise None."""
        imp = scope.imports.get(receiver)
        if imp is not None and imp.imported_symbol is None:
            return self._resolve_module(scope, imp)
        return None

    def _is_external_receiver(self, scope: _Scope, receiver: str) -> bool:
        """True when the receiver chain is *rooted* at an external module.

        The root matters, not just the whole chain: ``os.environ.get()`` has
        receiver ``os.environ``, which is bound to nothing, but its root ``os``
        is an external import -- so the call leaves the repository and must not
        be name-matched against repo symbols (there is a ``get`` in every
        repository class).
        """
        root = receiver.partition(".")[0]
        for candidate in (receiver, root):
            imp = scope.imports.get(candidate)
            if imp is not None and imp.imported_symbol is None:
                return self._resolve_module(scope, imp) is None
        return False

    def _resolve_module(self, scope: _Scope, imp: Import) -> str | None:
        """Map an import's module specifier to an internal module fqn, or None if
        it points outside the repository. The one language-specific step."""
        if scope.language is Language.PYTHON:
            if imp.is_relative:
                package = self._python_package(scope, imp.level)
                if package is None:
                    return None
                module = f"{package}.{imp.module}" if imp.module else package
            else:
                module = imp.module
            return module if module in self._modules else None

        # TypeScript: only relative specifiers can be intra-repo; bare specifiers
        # ('react') are always external packages.
        if imp.is_relative:
            return self._ts_resolve_relative(scope.path, imp.module)
        return None

    def _from_module_path(self, scope: _Scope, imp: Import) -> str | None:
        """The dotted/relative path of a ``from X import y`` clause, whether or
        not ``X`` is itself an indexed module -- so `from . import submodule`
        yields the package path used to look ``submodule`` up as a module."""
        if scope.language is Language.PYTHON:
            if imp.is_relative:
                package = self._python_package(scope, imp.level)
                if package is None:
                    return None
                return f"{package}.{imp.module}" if imp.module else package
            return imp.module
        if imp.is_relative:
            return self._ts_resolve_relative(scope.path, imp.module)
        return None

    @staticmethod
    def _python_package(scope: _Scope, level: int) -> str | None:
        """The package a relative import of ``level`` dots resolves against.

        A module's own name is not part of its package, but a package's
        ``__init__.py`` *is* the package (``__package__`` is the package itself,
        not its parent), so one leading dot there means "this package". Getting
        this wrong makes every re-export in every ``__init__.py`` resolve to a
        phantom external module -- and package inits are where real projects put
        their public API.
        """
        parts = scope.module_fqn.split(".")
        strip = level - 1 if scope.path.endswith(_PACKAGE_INIT) else level
        if strip > len(parts):
            return None
        base = parts[: len(parts) - strip] if strip else parts
        return ".".join(base) if base else None

    def _ts_resolve_relative(self, importer_path: str, spec: str) -> str | None:
        base_dir = posixpath.dirname(importer_path)
        target = posixpath.normpath(posixpath.join(base_dir, spec))
        for candidate in (target, f"{target}/index"):
            if candidate in self._modules:
                return candidate
        return None

    # -- inheritance map ------------------------------------------------- #

    def _resolve_inheritance(self) -> dict[str, list[str]]:
        bases: dict[str, list[str]] = {}
        for parsed in self._parsed:
            scope = self._scopes[parsed.module_fqn]
            for ref in parsed.references:
                if ref.kind is not EdgeKind.INHERITS:
                    continue
                outcome = self._resolve_use(scope, ref)
                if outcome.dst_fqn is not None:
                    bases.setdefault(ref.from_fqn, []).append(outcome.dst_fqn)
        return bases

    # -- edge construction & stats --------------------------------------- #

    @staticmethod
    def _edge(kind: EdgeKind, src_fqn: str, outcome: _Resolved) -> SymbolEdge:
        return SymbolEdge(
            kind=kind,
            src_fqn=src_fqn,
            dst_fqn=outcome.dst_fqn,
            dst_unresolved_name=outcome.dst_unresolved_name,
            confidence=outcome.confidence,
        )

    def _stats(self, edges: Sequence[SymbolEdge]) -> ResolutionStats:
        resolved = external = unresolved = 0
        by_language: dict[Language, list[int]] = {}
        for edge in edges:
            language = self._edge_language(edge)
            bucket = by_language.setdefault(language, [0, 0]) if language else None
            if edge.is_resolved:
                resolved += 1
                if bucket is not None:
                    bucket[0] += 1
            elif edge.confidence == confidence.EXTERNAL:
                external += 1
            else:
                unresolved += 1
                if bucket is not None:
                    bucket[1] += 1
        return ResolutionStats(
            total_edges=len(edges),
            resolved=resolved,
            external=external,
            unresolved=unresolved,
            by_language={lang: (r, u) for lang, (r, u) in by_language.items()},
        )

    def _edge_language(self, edge: SymbolEdge) -> Language | None:
        symbol = self._by_fqn.get(edge.src_fqn)
        return symbol.language if symbol is not None else None
