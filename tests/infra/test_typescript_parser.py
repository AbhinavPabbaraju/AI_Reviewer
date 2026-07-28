"""TypeScript extraction: declarations, imports, heritage, and calls."""

from __future__ import annotations

from app.domain.indexing.models import EdgeKind, SymbolKind
from app.infra.parsing.typescript import module_id_for_path
from tests.conftest import parse_typescript


class TestModuleId:
    def test_strips_extension_keeps_path(self) -> None:
        assert module_id_for_path("src/app/store.ts") == "src/app/store"
        assert module_id_for_path("src/app/view.tsx") == "src/app/view"


_SRC = '''import { Base, Helper as H } from "./base";
import * as util from "./util";
import Default from "react";

export class Store extends Base implements IStore {
  name: string;

  constructor(private repo: Repo) { super(); }

  async save(item: Item): Promise<void> {
    H(item);
    this.load(item.id);
    util.fetch(item.id);
    console.log("noise");
  }

  load(key: string): Item {
    return new Base();
  }
}

export function top(s: Store): void {
  new Store(r).save(null);
}

export const arrow = (x: number): Store => new Store(x);

export interface IStore { load(k: string): Item; }
export type Alias = Store | null;
export enum Color { Red, Green }

function internalHelper(): void {}
'''


def _parsed():
    return parse_typescript("src/store.ts", _SRC)


class TestSymbols:
    def test_module_and_declaration_kinds(self) -> None:
        by_fqn = {s.fqn: s for s in _parsed().symbols}
        assert by_fqn["src/store"].kind is SymbolKind.MODULE
        assert by_fqn["src/store::Store"].kind is SymbolKind.CLASS
        assert by_fqn["src/store::Store.save"].kind is SymbolKind.METHOD
        assert by_fqn["src/store::Store.save"].parent_fqn == "src/store::Store"
        assert by_fqn["src/store::top"].kind is SymbolKind.FUNCTION
        assert by_fqn["src/store::IStore"].kind is SymbolKind.INTERFACE
        assert by_fqn["src/store::Alias"].kind is SymbolKind.TYPE_ALIAS
        assert by_fqn["src/store::Color"].kind is SymbolKind.ENUM

    def test_arrow_const_is_a_function(self) -> None:
        by_fqn = {s.fqn: s for s in _parsed().symbols}
        assert by_fqn["src/store::arrow"].kind is SymbolKind.FUNCTION

    def test_export_visibility(self) -> None:
        by_fqn = {s.fqn: s for s in _parsed().symbols}
        assert by_fqn["src/store::top"].is_exported
        assert not by_fqn["src/store::internalHelper"].is_exported


class TestImports:
    def test_named_namespace_and_default(self) -> None:
        imports = {imp.local_name: imp for imp in _parsed().imports}
        assert imports["Base"].imported_symbol == "Base"
        assert imports["H"].imported_symbol == "Helper"  # aliased
        assert imports["util"].imported_symbol is None  # namespace import
        assert imports["Default"].imported_symbol == "default"
        assert imports["Base"].is_relative
        assert not imports["Default"].is_relative

    def test_side_effect_import(self) -> None:
        pf = parse_typescript("src/x.ts", 'import "./styles.css";\n')
        assert pf.imports[0].module == "./styles.css"
        assert pf.imports[0].imported_symbol is None


class TestReferences:
    def _refs(self):
        return _parsed().references

    def test_extends_and_implements_are_inherits(self) -> None:
        inherits = {r.target_name for r in self._refs() if r.kind is EdgeKind.INHERITS}
        assert {"Base", "IStore"} <= inherits

    def test_this_call_receiver(self) -> None:
        call = next(
            r for r in self._refs()
            if r.kind is EdgeKind.CALLS and r.target_name == "load"
        )
        assert call.receiver == "this"

    def test_namespace_member_call_captured(self) -> None:
        assert any(
            r.target_name == "fetch" and r.receiver == "util" for r in self._refs()
        )

    def test_console_member_is_skipped(self) -> None:
        assert not any(r.target_name == "log" for r in self._refs())

    def test_new_expression_is_a_call(self) -> None:
        assert any(
            r.kind is EdgeKind.CALLS and r.target_name == "Store" for r in self._refs()
        )


class TestRobustness:
    def test_broken_source_does_not_crash(self) -> None:
        pf = parse_typescript("src/broken.ts", "export class {{{ ")
        assert any(s.kind is SymbolKind.MODULE for s in pf.symbols)
