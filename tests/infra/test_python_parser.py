"""Python extraction: symbols, imports, and unresolved references."""

from __future__ import annotations

from app.domain.indexing.models import EdgeKind, SymbolKind
from app.infra.parsing.python import module_fqn_for_path
from tests.conftest import parse_python


class TestModuleFqn:
    def test_regular_module(self) -> None:
        assert module_fqn_for_path("app/files.py") == "app.files"

    def test_package_init_collapses(self) -> None:
        assert module_fqn_for_path("app/pkg/__init__.py") == "app.pkg"

    def test_stub_file(self) -> None:
        assert module_fqn_for_path("app/types.pyi") == "app.types"


_SRC = '''"""Module doc."""
import os
import a.b.c as abc
from a.b import c, d as e
from .util import helper


@decorator
class Store(Base):
    """Store docstring."""

    def _private(self):
        return 1

    def save(self, item: Item) -> None:
        return self.load(item)

    def load(self, key: str) -> Item:
        return c(key)


def top(s: Store) -> int:
    return len(s)
'''


def _parsed():
    return parse_python("app/store.py", _SRC)


class TestSymbols:
    def test_module_symbol_present(self) -> None:
        pf = _parsed()
        module = next(s for s in pf.symbols if s.kind is SymbolKind.MODULE)
        assert module.fqn == "app.store"

    def test_symbol_kinds_and_fqns(self) -> None:
        by_fqn = {s.fqn: s for s in _parsed().symbols}
        assert by_fqn["app.store.Store"].kind is SymbolKind.CLASS
        assert by_fqn["app.store.Store.save"].kind is SymbolKind.METHOD
        assert by_fqn["app.store.Store.save"].parent_fqn == "app.store.Store"
        assert by_fqn["app.store.top"].kind is SymbolKind.FUNCTION
        assert by_fqn["app.store.top"].parent_fqn is None

    def test_signature_and_docstring(self) -> None:
        by_fqn = {s.fqn: s for s in _parsed().symbols}
        assert by_fqn["app.store.Store.save"].signature == (
            "def save(self, item: Item) -> None"
        )
        assert by_fqn["app.store.Store"].signature == "class Store(Base)"
        assert by_fqn["app.store.Store"].docstring == "Store docstring."

    def test_export_visibility_from_underscore(self) -> None:
        by_fqn = {s.fqn: s for s in _parsed().symbols}
        assert by_fqn["app.store.Store.save"].is_exported
        assert not by_fqn["app.store.Store._private"].is_exported

    def test_decorated_class_span_includes_decorator(self) -> None:
        store = next(s for s in _parsed().symbols if s.fqn == "app.store.Store")
        # `@decorator` is line 8; the class keyword is line 9
        assert store.span.line_start == 8


class TestImports:
    def test_all_import_forms(self) -> None:
        imports = {imp.local_name: imp for imp in _parsed().imports}
        assert imports["os"].module == "os" and imports["os"].imported_symbol is None
        assert imports["abc"].module == "a.b.c"
        assert imports["c"].module == "a.b" and imports["c"].imported_symbol == "c"
        assert imports["e"].imported_symbol == "d"  # aliased
        helper = imports["helper"]
        assert helper.is_relative and helper.level == 1 and helper.module == "util"


class TestReferences:
    def _refs(self):
        return _parsed().references

    def test_inherits_reference(self) -> None:
        inherits = [r for r in self._refs() if r.kind is EdgeKind.INHERITS]
        assert any(r.target_name == "Base" for r in inherits)

    def test_self_call_has_receiver(self) -> None:
        call = next(
            r for r in self._refs()
            if r.kind is EdgeKind.CALLS and r.target_name == "load"
        )
        assert call.receiver == "self"
        assert call.from_fqn == "app.store.Store.save"

    def test_decorator_becomes_reference(self) -> None:
        assert any(
            r.kind is EdgeKind.REFERENCES and r.target_name == "decorator"
            for r in self._refs()
        )

    def test_type_annotation_becomes_reference(self) -> None:
        assert any(
            r.kind is EdgeKind.REFERENCES and r.target_name == "Item"
            for r in self._refs()
        )

    def test_builtins_are_not_referenced(self) -> None:
        # `len(s)` is a builtin call and must not create an edge
        assert not any(r.target_name == "len" for r in self._refs())

    def test_test_edge_only_in_test_files(self) -> None:
        prod = parse_python("app/x.py", "def test_thing():\n    pass\n")
        assert not any(r.kind is EdgeKind.TESTS for r in prod.references)
        test = parse_python(
            "tests/test_x.py", "def test_thing():\n    pass\n", is_test=True
        )
        tests_refs = [r for r in test.references if r.kind is EdgeKind.TESTS]
        assert tests_refs and tests_refs[0].target_name == "thing"


class TestRobustness:
    def test_syntactically_broken_file_does_not_crash(self) -> None:
        pf = parse_python("app/broken.py", "def f(:\n    return\nclass ")
        # tree-sitter is error-tolerant; we still get a module symbol and no crash
        assert any(s.kind is SymbolKind.MODULE for s in pf.symbols)

    def test_empty_file(self) -> None:
        pf = parse_python("app/empty.py", "")
        assert [s.kind for s in pf.symbols] == [SymbolKind.MODULE]
