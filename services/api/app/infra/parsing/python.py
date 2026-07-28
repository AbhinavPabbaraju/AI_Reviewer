"""Python symbol and reference extraction (ARCHITECTURE.md sec. 4.2).

Produces a language-agnostic :class:`ParsedFile` from Python source: a MODULE
symbol per file, functions/methods/classes with spans and signatures, import
bindings, and *unresolved* references (calls, inheritance, decorator/annotation
uses, and test-target guesses). Resolution is emphatically not done here -- the
parser records what the code *says*; the resolver decides what it *means*.

Two modeling choices worth stating:

* A MODULE symbol is emitted per file so that ``IMPORTS`` edges have a real
  ``src`` symbol (the DDL's ``symbol_kind = 'module'``), rather than dangling.
* Language builtins (``len``, ``print``, ``str`` ...) are **not** emitted as
  references. They are not repo symbols and never will be; an edge to ``len`` is
  noise that would only depress the resolution rate. The symbol graph is about
  the repository and its imports, not the language runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tree_sitter as ts

from app.domain.indexing.models import (
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

__all__ = ["PythonParser", "module_fqn_for_path"]

_WHITESPACE = re.compile(r"\s+")
_TEST_PREFIX = "test_"

# Builtins and common globals that are not repository symbols. Excluded from
# reference extraction to keep the graph (and the resolution rate) honest.
_PY_BUILTINS: frozenset[str] = frozenset(
    {
        "print", "len", "range", "enumerate", "zip", "map", "filter", "open",
        "isinstance", "issubclass", "getattr", "setattr", "hasattr", "super",
        "str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple",
        "frozenset", "object", "type", "repr", "hash", "id", "iter", "next",
        "sorted", "reversed", "sum", "min", "max", "abs", "round", "any", "all",
        "callable", "vars", "dir", "format", "input", "chr", "ord", "hex", "oct",
        "bin", "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "RuntimeError", "StopIteration", "None", "True", "False", "self", "cls",
        "property", "staticmethod", "classmethod", "NotImplementedError",
        "AttributeError", "OSError", "IOError",
    }
)


def module_fqn_for_path(path: str) -> str:
    """``app/files.py`` -> ``app.files``; ``pkg/__init__.py`` -> ``pkg``."""
    stem = path
    for ext in (".pyi", ".py"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    parts = [seg for seg in stem.split("/") if seg]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else stem


@dataclass(slots=True)
class _Extractor:
    source: bytes
    path: str
    module_fqn: str
    is_test: bool
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)

    # -- traversal ------------------------------------------------------- #

    def run(self, root: ts.Node) -> None:
        self.symbols.append(
            Symbol(
                fqn=self.module_fqn,
                name=self.module_fqn.rsplit(".", 1)[-1] or self.module_fqn,
                kind=SymbolKind.MODULE,
                span=span_of(root, self.path),
                language=Language.PYTHON,
                is_exported=True,
            )
        )
        self._visit(root, self.module_fqn, [], in_class=False)

    def _visit(
        self, node: ts.Node, enclosing_fqn: str, name_stack: list[str], *, in_class: bool
    ) -> None:
        for child in node.named_children:
            kind = child.type
            if kind == "decorated_definition":
                inner = child.child_by_field_name("definition")
                if inner is None:
                    continue
                decorators = [c for c in child.named_children if c.type == "decorator"]
                self._emit_def(inner, child, decorators, name_stack, in_class=in_class)
            elif kind in ("function_definition", "class_definition"):
                self._emit_def(child, child, [], name_stack, in_class=in_class)
            elif kind in ("import_statement", "import_from_statement"):
                self._emit_imports(child)
            else:
                self._collect_calls(child, enclosing_fqn)

    def _emit_def(
        self,
        node: ts.Node,
        span_node: ts.Node,
        decorators: list[ts.Node],
        name_stack: list[str],
        *,
        in_class: bool,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = node_text(name_node, self.source)
        fqn = ".".join([self.module_fqn, *name_stack, name])
        parent_fqn = (
            ".".join([self.module_fqn, *name_stack]) if name_stack else None
        )
        body = node.child_by_field_name("body")
        is_class = node.type == "class_definition"
        kind = (
            SymbolKind.CLASS
            if is_class
            else (SymbolKind.METHOD if in_class else SymbolKind.FUNCTION)
        )
        self.symbols.append(
            Symbol(
                fqn=fqn,
                name=name,
                kind=kind,
                span=span_of(span_node, self.path),
                language=Language.PYTHON,
                signature=_signature(node, body, self.source),
                docstring=_docstring(body),
                is_exported=not name.startswith("_"),
                parent_fqn=parent_fqn,
            )
        )

        for decorator in decorators:
            self._emit_decorator(decorator, fqn)
        self._emit_annotation_refs(node, fqn)

        if is_class:
            self._emit_inherits(node, fqn)
        elif (
            self.is_test
            and not in_class
            and name.startswith(_TEST_PREFIX)
            and len(name) > len(_TEST_PREFIX)
        ):
            self.references.append(
                Reference(
                    kind=EdgeKind.TESTS,
                    target_name=name[len(_TEST_PREFIX) :],
                    from_fqn=fqn,
                    line=span_node.start_point[0] + 1,
                )
            )

        if body is not None:
            self._visit(body, fqn, [*name_stack, name], in_class=is_class)

    # -- imports --------------------------------------------------------- #

    def _emit_imports(self, node: ts.Node) -> None:
        line = node.start_point[0] + 1
        if node.type == "import_statement":
            for name_node in node.children_by_field_name("name"):
                self._emit_plain_import(name_node, line)
            return
        self._emit_from_import(node, line)

    def _emit_plain_import(self, name_node: ts.Node, line: int) -> None:
        if name_node.type == "aliased_import":
            module = node_text(name_node.named_children[0], self.source)
            local = node_text(name_node.named_children[1], self.source)
        elif name_node.type == "dotted_name":
            module = node_text(name_node, self.source)
            local = module
        else:
            return
        self.imports.append(
            Import(local_name=local, module=module, imported_symbol=None, line=line)
        )

    def _emit_from_import(self, node: ts.Node, line: int) -> None:
        module_node = node.child_by_field_name("module_name")
        is_relative = False
        level = 0
        base_module = ""
        if module_node is None:
            return
        if module_node.type == "relative_import":
            is_relative = True
            prefix = module_node.named_children[0]
            level = len(node_text(prefix, self.source))
            if len(module_node.named_children) > 1:
                base_module = node_text(module_node.named_children[1], self.source)
        elif module_node.type == "dotted_name":
            base_module = node_text(module_node, self.source)

        name_nodes = node.children_by_field_name("name")
        if not name_nodes:  # `from x import *`
            self.imports.append(
                Import(
                    local_name=base_module or "*",
                    module=base_module,
                    imported_symbol=None,
                    is_relative=is_relative,
                    level=level,
                    line=line,
                )
            )
            return
        for name_node in name_nodes:
            if name_node.type == "aliased_import":
                symbol = node_text(name_node.named_children[0], self.source)
                local = node_text(name_node.named_children[1], self.source)
            elif name_node.type == "dotted_name":
                symbol = node_text(name_node, self.source)
                local = symbol
            else:
                continue
            self.imports.append(
                Import(
                    local_name=local,
                    module=base_module,
                    imported_symbol=symbol,
                    is_relative=is_relative,
                    level=level,
                    line=line,
                )
            )

    # -- references ------------------------------------------------------ #

    def _emit_inherits(self, class_node: ts.Node, class_fqn: str) -> None:
        supers = class_node.child_by_field_name("superclasses")
        if supers is None:
            return
        for base in supers.named_children:
            receiver, target = self._dotted_reference(base)
            if target is not None and target not in _PY_BUILTINS:
                self.references.append(
                    Reference(
                        kind=EdgeKind.INHERITS,
                        target_name=target,
                        from_fqn=class_fqn,
                        line=base.start_point[0] + 1,
                        receiver=receiver,
                    )
                )

    def _emit_decorator(self, decorator: ts.Node, owner_fqn: str) -> None:
        expr = decorator.named_children[0] if decorator.named_children else None
        if expr is None:
            return
        if expr.type == "call":
            fn = expr.child_by_field_name("function")
            expr = fn if fn is not None else expr
        receiver, target = self._dotted_reference(expr)
        if target is not None and target not in _PY_BUILTINS:
            self.references.append(
                Reference(
                    kind=EdgeKind.REFERENCES,
                    target_name=target,
                    from_fqn=owner_fqn,
                    line=decorator.start_point[0] + 1,
                    receiver=receiver,
                )
            )

    def _emit_annotation_refs(self, def_node: ts.Node, owner_fqn: str) -> None:
        params = def_node.child_by_field_name("parameters")
        type_nodes: list[ts.Node] = []
        if params is not None:
            for param in params.named_children:
                annotation = param.child_by_field_name("type")
                if annotation is not None:
                    type_nodes.append(annotation)
        return_type = def_node.child_by_field_name("return_type")
        if return_type is not None:
            type_nodes.append(return_type)
        for type_node in type_nodes:
            for name in self._type_names(type_node):
                receiver, target = name
                if target and target not in _PY_BUILTINS:
                    self.references.append(
                        Reference(
                            kind=EdgeKind.REFERENCES,
                            target_name=target,
                            from_fqn=owner_fqn,
                            line=type_node.start_point[0] + 1,
                            receiver=receiver,
                        )
                    )

    def _collect_calls(self, node: ts.Node, enclosing_fqn: str) -> None:
        if node.type in (
            "function_definition",
            "class_definition",
            "decorated_definition",
        ):
            return
        if node.type == "call":
            self._emit_call(node, enclosing_fqn)
        for child in node.named_children:
            self._collect_calls(child, enclosing_fqn)

    def _emit_call(self, call_node: ts.Node, enclosing_fqn: str) -> None:
        fn = call_node.child_by_field_name("function")
        if fn is None:
            return
        receiver, target = self._dotted_reference(fn)
        if target is None or target in _PY_BUILTINS:
            return
        self.references.append(
            Reference(
                kind=EdgeKind.CALLS,
                target_name=target,
                from_fqn=enclosing_fqn,
                line=call_node.start_point[0] + 1,
                receiver=receiver,
            )
        )

    # -- shared node helpers --------------------------------------------- #

    def _dotted_reference(self, node: ts.Node) -> tuple[str | None, str | None]:
        """Split an identifier or attribute chain into ``(receiver, name)``.

        ``foo`` -> ``(None, 'foo')``; ``self.foo`` -> ``('self', 'foo')``;
        ``a.b.foo`` -> ``('a.b', 'foo')``. Anything with a call or subscript in
        the receiver position yields ``(None, name)`` so it resolves by name."""
        if node.type == "identifier":
            return None, node_text(node, self.source)
        if node.type == "attribute":
            attr = node.child_by_field_name("attribute")
            obj = node.child_by_field_name("object")
            if attr is None or obj is None:
                return None, None
            receiver = self._receiver_text(obj)
            return receiver, node_text(attr, self.source)
        return None, None

    def _receiver_text(self, node: ts.Node) -> str | None:
        if node.type == "identifier":
            return node_text(node, self.source)
        if node.type == "attribute":
            obj = node.child_by_field_name("object")
            attr = node.child_by_field_name("attribute")
            if obj is None or attr is None:
                return None
            base = self._receiver_text(obj)
            return f"{base}.{node_text(attr, self.source)}" if base else None
        return None

    def _type_names(self, node: ts.Node) -> list[tuple[str | None, str | None]]:
        """Every identifier/attribute name mentioned in a type annotation, so
        ``dict[str, Foo]`` yields a reference to ``Foo`` (``str`` is a builtin and
        filtered by the caller)."""
        results: list[tuple[str | None, str | None]] = []
        if node.type in ("identifier", "attribute"):
            results.append(self._dotted_reference(node))
            return results
        for child in node.named_children:
            results.extend(self._type_names(child))
        return results


def _signature(node: ts.Node, body: ts.Node | None, source: bytes) -> str:
    end = body.start_byte if body is not None else node.end_byte
    text = source[node.start_byte : end].decode("utf-8", "replace").strip()
    return _WHITESPACE.sub(" ", text).rstrip(":").strip()


def _docstring(body: ts.Node | None) -> str | None:
    if body is None:
        return None
    for child in body.named_children:
        if child.type != "expression_statement":
            return None
        inner = child.named_children[0] if child.named_children else None
        if inner is not None and inner.type == "string" and inner.text is not None:
            return _strip_string_quotes(inner.text.decode("utf-8", "replace"))
        return None
    return None


def _strip_string_quotes(raw: str) -> str:
    text = raw
    if text[:1] in "rbfuRBFU":
        text = text.lstrip("rbfuRBFU")
    for quote in ('"""', "'''", '"', "'"):
        if text.startswith(quote) and text.endswith(quote) and len(text) >= 2 * len(quote):
            return text[len(quote) : -len(quote)].strip()
    return text.strip()


class PythonParser:
    """Implements :class:`app.domain.indexing.ports.ParserPort` for Python."""

    @property
    def language(self) -> Language:
        return Language.PYTHON

    def parse(self, file: SourceFile, source: bytes) -> ParsedFile:
        parser = make_parser("python")
        tree = parser.parse(source)
        extractor = _Extractor(
            source=source,
            path=file.path,
            module_fqn=module_fqn_for_path(file.path),
            is_test=file.is_test,
        )
        extractor.run(tree.root_node)
        return ParsedFile(
            file=file,
            module_fqn=extractor.module_fqn,
            symbols=tuple(extractor.symbols),
            imports=tuple(extractor.imports),
            references=tuple(extractor.references),
        )
