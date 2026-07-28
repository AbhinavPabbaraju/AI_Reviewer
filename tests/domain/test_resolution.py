"""Edge resolution: the hardest correctness work, tested case by case (sec. 4.2).

These run the *real* tree-sitter parsers into the resolver, so a passing test
means the whole extract-then-resolve path works, not just the resolver in
isolation.
"""

from __future__ import annotations

from app.domain.indexing.models import EdgeKind, SymbolEdge, confidence
from app.domain.indexing.resolution import Resolver
from tests.conftest import parse_python, parse_typescript


def _edges(*parsed: object) -> list[SymbolEdge]:
    return list(Resolver(list(parsed)).resolve().edges)  # type: ignore[arg-type]


def _find(
    edges: list[SymbolEdge], kind: EdgeKind, src: str, *, dst: str | None = None
) -> SymbolEdge:
    for edge in edges:
        if edge.kind is kind and edge.src_fqn == src and (dst is None or edge.dst_fqn == dst):
            return edge
    raise AssertionError(f"no {kind.value} edge from {src} to {dst}")


class TestPythonResolution:
    def test_self_call_resolves_within_class(self) -> None:
        src = (
            "class Store:\n"
            "    def save(self, x):\n"
            "        return self.load(x)\n"
            "    def load(self, x):\n"
            "        return x\n"
        )
        edge = _find(_edges(parse_python("app/s.py", src)), EdgeKind.CALLS, "app.s.Store.save")
        assert edge.dst_fqn == "app.s.Store.load"
        assert edge.confidence == confidence.EXACT

    def test_self_call_resolves_to_inherited_method(self) -> None:
        base = "class Base:\n    def read(self):\n        return 1\n"
        derived = (
            "from app.base import Base\n"
            "class Store(Base):\n"
            "    def go(self):\n"
            "        return self.read()\n"
        )
        edges = _edges(parse_python("app/base.py", base), parse_python("app/store.py", derived))
        edge = _find(edges, EdgeKind.CALLS, "app.store.Store.go")
        assert edge.dst_fqn == "app.base.Base.read"

    def test_imported_symbol_call_resolves(self) -> None:
        util = "def helper(x):\n    return x\n"
        main = (
            "from app.util import helper\n"
            "def run():\n"
            "    return helper(1)\n"
        )
        edges = _edges(parse_python("app/util.py", util), parse_python("app/main.py", main))
        edge = _find(edges, EdgeKind.CALLS, "app.main.run")
        assert edge.dst_fqn == "app.util.helper"
        assert edge.confidence == confidence.EXACT

    def test_relative_import_resolves(self) -> None:
        util = "def helper(x):\n    return x\n"
        main = "from .util import helper\ndef run():\n    return helper(1)\n"
        edges = _edges(parse_python("app/util.py", util), parse_python("app/main.py", main))
        imp = _find(edges, EdgeKind.IMPORTS, "app.main")
        assert imp.dst_fqn == "app.util.helper"

    def test_module_alias_attribute_call(self) -> None:
        util = "def helper(x):\n    return x\n"
        main = "import app.util as u\ndef run():\n    return u.helper(1)\n"
        edges = _edges(parse_python("app/util.py", util), parse_python("app/main.py", main))
        edge = _find(edges, EdgeKind.CALLS, "app.main.run")
        assert edge.dst_fqn == "app.util.helper"

    def test_external_import_is_classified_external(self) -> None:
        main = "import os\ndef run():\n    return os.getcwd()\n"
        edges = _edges(parse_python("app/main.py", main))
        imp = _find(edges, EdgeKind.IMPORTS, "app.main")
        assert not imp.is_resolved
        assert imp.confidence == confidence.EXTERNAL
        call = _find(edges, EdgeKind.CALLS, "app.main.run")
        assert call.dst_unresolved_name == "os.getcwd"
        assert call.confidence == confidence.EXTERNAL

    def test_ambiguous_name_is_low_confidence(self) -> None:
        a = "def handle():\n    return 1\n"
        b = "def handle():\n    return 2\n"
        caller = "def run():\n    return handle()\n"  # no import: pure heuristic
        edges = _edges(
            parse_python("app/a.py", a),
            parse_python("app/b.py", b),
            parse_python("app/c.py", caller),
        )
        edge = _find(edges, EdgeKind.CALLS, "app.c.run")
        assert edge.confidence == confidence.HEURISTIC_AMBIGUOUS
        assert edge.is_resolved  # a target is still chosen deterministically

    def test_unknown_bare_name_is_unresolved_not_dropped(self) -> None:
        main = "def run():\n    return nonexistent_helper()\n"
        edge = _find(_edges(parse_python("app/main.py", main)), EdgeKind.CALLS, "app.main.run")
        assert not edge.is_resolved
        assert edge.dst_unresolved_name == "nonexistent_helper"
        assert edge.confidence == confidence.UNRESOLVED

    def test_test_edge_inferred_from_name_and_import(self) -> None:
        util = "def helper(x):\n    return x\n"
        test = (
            "from app.util import helper\n"
            "def test_helper():\n"
            "    assert helper(1) == 1\n"
        )
        edges = _edges(
            parse_python("app/util.py", util),
            parse_python("tests/test_util.py", test, is_test=True),
        )
        edge = _find(edges, EdgeKind.TESTS, "tests.test_util.test_helper")
        assert edge.dst_fqn == "app.util.helper"
        assert edge.confidence == confidence.TEST_HEURISTIC


class TestTypeScriptResolution:
    def test_relative_named_import_resolves(self) -> None:
        base = "export function helper(x: number): number { return x; }\n"
        main = (
            'import { helper } from "./helper";\n'
            "export function run(): number { return helper(1); }\n"
        )
        edges = _edges(
            parse_typescript("src/helper.ts", base),
            parse_typescript("src/run.ts", main),
        )
        edge = _find(edges, EdgeKind.CALLS, "src/run::run")
        assert edge.dst_fqn == "src/helper::helper"

    def test_extends_resolves_across_files(self) -> None:
        base = "export class Base {}\n"
        derived = (
            'import { Base } from "./base";\n'
            "export class Store extends Base {}\n"
        )
        edges = _edges(
            parse_typescript("src/base.ts", base),
            parse_typescript("src/store.ts", derived),
        )
        edge = _find(edges, EdgeKind.INHERITS, "src/store::Store")
        assert edge.dst_fqn == "src/base::Base"

    def test_namespace_import_member_call(self) -> None:
        base = "export function fetchIt(k: string): void {}\n"
        main = (
            'import * as api from "./api";\n'
            "export function run(): void { api.fetchIt('x'); }\n"
        )
        edges = _edges(
            parse_typescript("src/api.ts", base),
            parse_typescript("src/run.ts", main),
        )
        edge = _find(edges, EdgeKind.CALLS, "src/run::run")
        assert edge.dst_fqn == "src/api::fetchIt"

    def test_bare_specifier_is_external(self) -> None:
        main = 'import { useState } from "react";\nexport const x = () => useState();\n'
        edges = _edges(parse_typescript("src/c.ts", main))
        imp = _find(edges, EdgeKind.IMPORTS, "src/c")
        assert not imp.is_resolved
        assert imp.confidence == confidence.EXTERNAL


class TestEdgeSetHygiene:
    def test_duplicate_calls_are_deduplicated_keeping_best_confidence(self) -> None:
        util = "def helper(x):\n    return x\n"
        main = (
            "from app.util import helper\n"
            "def run():\n"
            "    helper(1)\n"
            "    helper(2)\n"
        )
        edges = _edges(parse_python("app/util.py", util), parse_python("app/main.py", main))
        calls = [
            e for e in edges
            if e.kind is EdgeKind.CALLS and e.src_fqn == "app.main.run"
            and e.dst_fqn == "app.util.helper"
        ]
        assert len(calls) == 1

    def test_stats_rate_excludes_external(self) -> None:
        util = "def helper(x):\n    return x\n"
        main = (
            "import os\n"
            "from app.util import helper\n"
            "def run():\n"
            "    helper(1)\n"
            "    os.getcwd()\n"
        )
        result = Resolver(
            [parse_python("app/util.py", util), parse_python("app/main.py", main)]
        ).resolve()
        # os import + os.getcwd are external and must not drag the rate down
        assert result.stats.external >= 2
        assert result.stats.resolution_rate == 1.0
