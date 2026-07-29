"""TypeScript symbol and reference extraction (ARCHITECTURE.md sec. 4.2).

The counterpart to ``python.py``, producing the same language-agnostic
:class:`ParsedFile`. TypeScript fqns use ``<module-path>::<Nested.Name>`` because
a TS module id is a filesystem path, not a dotted name; the resolver never parses
these strings (it keys on ``(module, name)``), so the differing convention costs
nothing.

Scope notes for M1:

* ``extends`` and ``implements`` both become ``INHERITS`` edges -- for the symbol
  graph, "shares a supertype's surface" is the useful relation regardless of
  which keyword expressed it.
* No ``TESTS`` edges are inferred for TypeScript. Python's ``test_*`` convention
  is a strong signal; a Jest ``describe``/``it`` block is not, and a wrong
  ``TESTS`` edge is worse than a missing one. The calls a test makes are still
  captured as ordinary ``CALLS`` edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tree_sitter as ts

from app.domain.indexing.models import (
    STAR_IMPORT,
    EdgeKind,
    Import,
    Language,
    ParsedFile,
    Reference,
    SourceFile,
    Symbol,
    SymbolKind,
)
from app.infra.parsing.base import make_parser, node_text, span_of

__all__ = ["TypeScriptParser", "module_id_for_path"]

_WHITESPACE = re.compile(r"\s+")

# Objects whose members are runtime, never repo symbols: `console.log`,
# `Math.max`, `JSON.parse` are dropped by *receiver*. Deliberately excludes names
# a project might legitimately import (e.g. `fetch`), which are resolved normally.
_TS_MEMBER_GLOBALS: frozenset[str] = frozenset(
    {
        "console", "window", "document", "globalThis", "process", "Math", "JSON",
        "Object", "Reflect", "Number", "String", "Boolean", "Symbol", "Date",
        "Array", "Promise", "localStorage", "sessionStorage", "navigator",
    }
)

# Free-function globals dropped in *bare* call position. Kept narrow: only names
# that are never imported, so an imported binding is never mistaken for a global.
_TS_BARE_GLOBALS: frozenset[str] = frozenset(
    {
        "parseInt", "parseFloat", "isNaN", "isFinite", "setTimeout",
        "setInterval", "clearTimeout", "clearInterval", "structuredClone",
        "queueMicrotask", "require", "super",
    }
)

# Built-in type names, filtered out of annotation references.
_TS_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "string", "number", "boolean", "void", "any", "unknown", "never", "null",
        "undefined", "object", "bigint", "symbol", "this", "Array", "Promise",
        "Record", "Partial", "Readonly", "Pick", "Omit", "Map", "Set", "Date",
        "Function", "Object", "String", "Number", "Boolean", "RegExp", "Error",
        "Iterable", "Iterator", "ReadonlyArray",
    }
)

_TS_EXTENSIONS: tuple[str, ...] = (
    ".d.ts", ".tsx", ".ts", ".mts", ".cts", ".jsx", ".js", ".mjs", ".cjs",
)
_CALLABLE_VALUE_TYPES = frozenset({"arrow_function", "function_expression", "function"})
_SYMBOL_BOUNDARIES = frozenset(
    {
        "function_declaration", "generator_function_declaration",
        "class_declaration", "abstract_class_declaration", "method_definition",
    }
)


def module_id_for_path(path: str) -> str:
    """``src/store.ts`` -> ``src/store``. Slashes are kept; only the extension is
    stripped, because a TS module id *is* its path."""
    for ext in _TS_EXTENSIONS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


@dataclass(slots=True)
class _Extractor:
    source: bytes
    path: str
    module_id: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    scopes: list[dict[str, str]] = field(default_factory=list)
    """Stack of ``local name -> constructed class name`` maps, innermost last."""

    # -- fqn helpers ----------------------------------------------------- #

    def _fqn(self, name_stack: list[str], name: str) -> str:
        return f"{self.module_id}::" + ".".join([*name_stack, name])

    def _parent(self, name_stack: list[str]) -> str | None:
        return f"{self.module_id}::" + ".".join(name_stack) if name_stack else None

    # -- traversal ------------------------------------------------------- #

    def run(self, root: ts.Node) -> None:
        self.symbols.append(
            Symbol(
                fqn=self.module_id,
                name=self.module_id.rsplit("/", 1)[-1],
                kind=SymbolKind.MODULE,
                span=span_of(root, self.path),
                language=Language.TYPESCRIPT,
            )
        )
        self.scopes.append(_local_constructors(root, self.source))
        for child in root.named_children:
            self._handle(child, [], exported=False)
        self.scopes.pop()

    def _handle(self, node: ts.Node, name_stack: list[str], *, exported: bool) -> None:
        kind = node.type
        if kind == "export_statement":
            source = node.child_by_field_name("source")
            if source is not None:
                self._emit_reexport(node, source)
                return
            decl = node.child_by_field_name("declaration")
            if decl is not None:
                self._handle(decl, name_stack, exported=True)
            return
        if kind in ("class_declaration", "abstract_class_declaration"):
            self._emit_class(node, name_stack, exported=exported)
        elif kind in ("function_declaration", "generator_function_declaration"):
            self._emit_callable(node, name_stack, SymbolKind.FUNCTION, exported=exported)
        elif kind in ("lexical_declaration", "variable_declaration"):
            self._emit_variables(node, name_stack, exported=exported)
        elif kind == "interface_declaration":
            self._emit_interface(node, name_stack, exported=exported)
        elif kind == "type_alias_declaration":
            self._emit_named(node, name_stack, SymbolKind.TYPE_ALIAS, exported=exported)
        elif kind == "enum_declaration":
            self._emit_named(node, name_stack, SymbolKind.ENUM, exported=exported)
        elif kind == "import_statement":
            self._emit_import(node)
        else:
            self._collect_calls(node, self.module_id)

    # -- declarations ---------------------------------------------------- #

    def _emit_class(
        self, node: ts.Node, name_stack: list[str], *, exported: bool
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = node_text(name_node, self.source)
        fqn = self._fqn(name_stack, name)
        self.symbols.append(
            Symbol(
                fqn=fqn,
                name=name,
                kind=SymbolKind.CLASS,
                span=span_of(node, self.path),
                language=Language.TYPESCRIPT,
                signature=self._signature(node, node.child_by_field_name("body")),
                is_exported=exported,
                parent_fqn=self._parent(name_stack),
            )
        )
        self._emit_heritage(node, fqn)
        body = node.child_by_field_name("body")
        if body is None:
            return
        for member in body.named_children:
            if member.type == "method_definition":
                self._emit_callable(
                    member, [*name_stack, name], SymbolKind.METHOD, exported=exported
                )
            elif member.type in ("public_field_definition", "field_definition"):
                self._emit_annotation_refs(member, fqn)

    def _emit_interface(
        self, node: ts.Node, name_stack: list[str], *, exported: bool
    ) -> None:
        self._emit_named(node, name_stack, SymbolKind.INTERFACE, exported=exported)
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        fqn = self._fqn(name_stack, node_text(name_node, self.source))
        # `interface A extends B` — the extends target is an extends_clause child.
        for child in node.named_children:
            if child.type == "extends_type_clause" or child.type == "extends_clause":
                for base in child.named_children:
                    self._emit_type_reference(base, fqn, EdgeKind.INHERITS)

    def _emit_named(
        self,
        node: ts.Node,
        name_stack: list[str],
        kind: SymbolKind,
        *,
        exported: bool,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = node_text(name_node, self.source)
        fqn = self._fqn(name_stack, name)
        self.symbols.append(
            Symbol(
                fqn=fqn,
                name=name,
                kind=kind,
                span=span_of(node, self.path),
                language=Language.TYPESCRIPT,
                signature=self._first_line(node),
                is_exported=exported,
                parent_fqn=self._parent(name_stack),
            )
        )

    def _emit_callable(
        self,
        node: ts.Node,
        name_stack: list[str],
        kind: SymbolKind,
        *,
        exported: bool,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = node_text(name_node, self.source)
        fqn = self._fqn(name_stack, name)
        body = node.child_by_field_name("body")
        self.symbols.append(
            Symbol(
                fqn=fqn,
                name=name,
                kind=kind,
                span=span_of(node, self.path),
                language=Language.TYPESCRIPT,
                signature=self._signature(node, body),
                is_exported=exported,
                parent_fqn=self._parent(name_stack),
            )
        )
        self._emit_annotation_refs(node, fqn)
        if body is not None:
            self.scopes.append(_local_constructors(body, self.source))
            self._collect_calls(body, fqn)
            self.scopes.pop()

    def _emit_variables(
        self, node: ts.Node, name_stack: list[str], *, exported: bool
    ) -> None:
        for declarator in node.named_children:
            if declarator.type != "variable_declarator":
                continue
            value = declarator.child_by_field_name("value")
            name_node = declarator.child_by_field_name("name")
            if name_node is None or name_node.type != "identifier":
                continue
            if value is None or value.type not in _CALLABLE_VALUE_TYPES:
                continue
            name = node_text(name_node, self.source)
            fqn = self._fqn(name_stack, name)
            self.symbols.append(
                Symbol(
                    fqn=fqn,
                    name=name,
                    kind=SymbolKind.FUNCTION,
                    span=span_of(declarator, self.path),
                    language=Language.TYPESCRIPT,
                    signature=self._first_line(declarator),
                    is_exported=exported,
                    parent_fqn=self._parent(name_stack),
                )
            )
            self._emit_annotation_refs(value, fqn)
            body = value.child_by_field_name("body")
            if body is not None:
                self.scopes.append(_local_constructors(body, self.source))
                self._collect_calls(body, fqn)
                self.scopes.pop()

    # -- imports --------------------------------------------------------- #

    def _emit_import(self, node: ts.Node) -> None:
        source_node = node.child_by_field_name("source")
        if source_node is None:
            return
        module = _string_value(node_text(source_node, self.source))
        is_relative = module.startswith(".")
        line = node.start_point[0] + 1
        clause = _first_child_of_type(node, "import_clause")
        if clause is None:  # side-effect import: `import "./x"`
            self.imports.append(
                Import(
                    local_name=module,
                    module=module,
                    imported_symbol=None,
                    is_relative=is_relative,
                    line=line,
                )
            )
            return
        for member in clause.named_children:
            if member.type == "named_imports":
                for spec in member.named_children:
                    if spec.type != "import_specifier":
                        continue
                    name = node_text(spec.child_by_field_name("name"), self.source)  # type: ignore[arg-type]
                    alias_node = spec.child_by_field_name("alias")
                    local = node_text(alias_node, self.source) if alias_node else name
                    self.imports.append(
                        Import(
                            local_name=local,
                            module=module,
                            imported_symbol=name,
                            is_relative=is_relative,
                            line=line,
                        )
                    )
            elif member.type == "namespace_import":
                alias = member.named_children[-1]
                self.imports.append(
                    Import(
                        local_name=node_text(alias, self.source),
                        module=module,
                        imported_symbol=None,
                        is_relative=is_relative,
                        line=line,
                    )
                )
            elif member.type == "identifier":  # default import
                self.imports.append(
                    Import(
                        local_name=node_text(member, self.source),
                        module=module,
                        imported_symbol="default",
                        is_relative=is_relative,
                        line=line,
                    )
                )

    def _emit_reexport(self, node: ts.Node, source_node: ts.Node) -> None:
        """``export { X } from "./m"`` / ``export * from "./m"``.

        Modeled as *imports*, because that is what they are: the name is bound
        into this module's namespace (and re-exposed from it). Emitting them as
        imports means the resolver's re-export following works on TS barrels
        with no TS-specific code -- and a barrel ``index.ts`` is how most
        TypeScript code is imported, so dropping these statements (as the parser
        previously did) silently disconnected whole packages from the graph.
        """
        module = _string_value(node_text(source_node, self.source))
        is_relative = module.startswith(".")
        line = node.start_point[0] + 1
        clause = _first_child_of_type(node, "export_clause")
        if clause is not None:
            for spec in clause.named_children:
                if spec.type != "export_specifier":
                    continue
                name_node = spec.child_by_field_name("name")
                if name_node is None:
                    continue
                name = node_text(name_node, self.source)
                alias_node = spec.child_by_field_name("alias")
                local = node_text(alias_node, self.source) if alias_node else name
                self.imports.append(
                    Import(
                        local_name=local,
                        module=module,
                        imported_symbol=name,
                        is_relative=is_relative,
                        line=line,
                    )
                )
            return
        namespace = _first_child_of_type(node, "namespace_export")
        local = (
            node_text(namespace.named_children[-1], self.source)
            if namespace is not None and namespace.named_children
            else STAR_IMPORT
        )
        self.imports.append(
            Import(
                local_name=local,
                module=module,
                imported_symbol=None,
                is_relative=is_relative,
                line=line,
            )
        )

    # -- references ------------------------------------------------------ #

    def _emit_heritage(self, class_node: ts.Node, class_fqn: str) -> None:
        heritage = _first_child_of_type(class_node, "class_heritage")
        if heritage is None:
            return
        for clause in heritage.named_children:
            if clause.type == "extends_clause":
                value = clause.child_by_field_name("value")
                targets = [value] if value is not None else clause.named_children
            elif clause.type == "implements_clause":
                targets = list(clause.named_children)
            else:
                continue
            for target in targets:
                self._emit_type_reference(target, class_fqn, EdgeKind.INHERITS)

    def _emit_annotation_refs(self, node: ts.Node, owner_fqn: str) -> None:
        for annotation in _descendants_of_type(node, "type_annotation"):
            for type_node in _type_identifiers(annotation):
                self._emit_type_reference(type_node, owner_fqn, EdgeKind.REFERENCES)

    def _emit_type_reference(
        self, node: ts.Node, owner_fqn: str, kind: EdgeKind
    ) -> None:
        receiver, target = self._dotted(node)
        if target is None or target in _TS_BUILTIN_TYPES:
            return
        self.references.append(
            Reference(
                kind=kind,
                target_name=target,
                from_fqn=owner_fqn,
                line=node.start_point[0] + 1,
                receiver=receiver,
            )
        )

    def _collect_calls(self, node: ts.Node, enclosing_fqn: str) -> None:
        if node.type in _SYMBOL_BOUNDARIES:
            return
        if node.type == "call_expression":
            self._emit_call(node.child_by_field_name("function"), node, enclosing_fqn)
        elif node.type == "new_expression":
            self._emit_call(
                node.child_by_field_name("constructor"), node, enclosing_fqn
            )
        for child in node.named_children:
            self._collect_calls(child, enclosing_fqn)

    def _emit_call(
        self, target_node: ts.Node | None, call_node: ts.Node, enclosing_fqn: str
    ) -> None:
        if target_node is None:
            return
        receiver, target = self._dotted(target_node)
        if target is None:
            return
        if receiver is None and target in _TS_BARE_GLOBALS:
            return
        if receiver is not None and receiver.split(".", 1)[0] in _TS_MEMBER_GLOBALS:
            return
        self.references.append(
            Reference(
                kind=EdgeKind.CALLS,
                target_name=target,
                from_fqn=enclosing_fqn,
                line=call_node.start_point[0] + 1,
                receiver=receiver,
                receiver_type=self._local_type(receiver),
            )
        )

    def _local_type(self, receiver: str | None) -> str | None:
        """The class a plain-name receiver was constructed from, innermost scope
        first. Dotted receivers are member chains, not locals."""
        if receiver is None or "." in receiver:
            return None
        for scope in reversed(self.scopes):
            found = scope.get(receiver)
            if found is not None:
                return found
        return None

    def _dotted(self, node: ts.Node) -> tuple[str | None, str | None]:
        if node.type in ("identifier", "type_identifier", "property_identifier"):
            return None, node_text(node, self.source)
        if node.type in ("member_expression", "nested_type_identifier"):
            prop = node.child_by_field_name("property")
            obj = node.child_by_field_name("object")
            if prop is None or obj is None:
                # nested_type_identifier: module '.' name without named fields
                parts = list(node.named_children)
                if len(parts) >= 2:
                    return self._receiver(parts[0]), node_text(parts[-1], self.source)
                return None, None
            return self._receiver(obj), node_text(prop, self.source)
        return None, None

    def _receiver(self, node: ts.Node) -> str | None:
        if node.type == "this":
            return "this"
        if node.type in ("identifier", "type_identifier"):
            return node_text(node, self.source)
        if node.type == "member_expression":
            obj = node.child_by_field_name("object")
            prop = node.child_by_field_name("property")
            if obj is None or prop is None:
                return None
            base = self._receiver(obj)
            return f"{base}.{node_text(prop, self.source)}" if base else None
        return None

    # -- text helpers ---------------------------------------------------- #

    def _signature(self, node: ts.Node, body: ts.Node | None) -> str:
        end = body.start_byte if body is not None else node.end_byte
        text = self.source[node.start_byte : end].decode("utf-8", "replace").strip()
        return _WHITESPACE.sub(" ", text).rstrip("{").strip()

    def _first_line(self, node: ts.Node) -> str:
        text = node_text(node, self.source).split("\n", 1)[0].strip()
        return _WHITESPACE.sub(" ", text).rstrip("{").strip()


def _local_constructors(body: ts.Node, source: bytes) -> dict[str, str]:
    """``{local name: class name}`` for every ``const x = new Thing(...)``
    directly in this scope.

    TypeScript states construction explicitly with ``new``, so unlike Python
    there is no ambiguity about whether the right-hand side is a constructor
    call -- but the class name is still only *recorded* here. The resolver
    decides whether it names a class in this repository (see
    ``confidence.INFERRED_LOCAL``).
    """
    found: dict[str, str] = {}
    stack = list(body.named_children)
    while stack:
        node = stack.pop()
        if node.type in _SYMBOL_BOUNDARIES:
            continue
        if node.type == "variable_declarator":
            binding = _constructor_binding(node, source)
            if binding is not None:
                found[binding[0]] = binding[1]
            continue
        stack.extend(node.named_children)
    return found


def _constructor_binding(
    declarator: ts.Node, source: bytes
) -> tuple[str, str] | None:
    name_node = declarator.child_by_field_name("name")
    value = declarator.child_by_field_name("value")
    if name_node is None or value is None or name_node.type != "identifier":
        return None
    if value.type != "new_expression":
        return None
    constructor = value.child_by_field_name("constructor")
    if constructor is None or constructor.type != "identifier":
        return None
    return node_text(name_node, source), node_text(constructor, source)


def _string_value(raw: str) -> str:
    return raw.strip().strip("'\"`")


def _first_child_of_type(node: ts.Node, type_name: str) -> ts.Node | None:
    for child in node.named_children:
        if child.type == type_name:
            return child
    return None


def _descendants_of_type(node: ts.Node, type_name: str) -> list[ts.Node]:
    found: list[ts.Node] = []
    stack = list(node.named_children)
    while stack:
        current = stack.pop()
        if current.type in _SYMBOL_BOUNDARIES:
            continue
        if current.type == type_name:
            found.append(current)
        stack.extend(current.named_children)
    return found


def _type_identifiers(node: ts.Node) -> list[ts.Node]:
    found: list[ts.Node] = []
    stack = list(node.named_children)
    while stack:
        current = stack.pop()
        if current.type in ("type_identifier", "nested_type_identifier"):
            found.append(current)
        else:
            stack.extend(current.named_children)
    return found


class TypeScriptParser:
    """Implements :class:`app.domain.indexing.ports.ParserPort` for TypeScript."""

    @property
    def language(self) -> Language:
        return Language.TYPESCRIPT

    def parse(self, file: SourceFile, source: bytes) -> ParsedFile:
        grammar = "tsx" if file.path.endswith((".tsx", ".jsx")) else "typescript"
        tree = make_parser(grammar).parse(source)
        module_id = module_id_for_path(file.path)
        extractor = _Extractor(source=source, path=file.path, module_id=module_id)
        extractor.run(tree.root_node)
        return ParsedFile(
            file=file,
            module_fqn=module_id,
            symbols=tuple(extractor.symbols),
            imports=tuple(extractor.imports),
            references=tuple(extractor.references),
        )
