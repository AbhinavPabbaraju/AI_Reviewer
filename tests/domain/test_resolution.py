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


class TestPackageAndReExportResolution:
    """Re-exports: how real code is imported, and where a naive resolver quits.

    A package `__init__` (or a TS barrel) defines almost nothing itself; it
    re-exposes names from its submodules. Treating those imports as unresolved
    disconnects the package's entire public surface from the graph.
    """

    def test_relative_import_inside_package_init_resolves(self) -> None:
        # In `pkg/__init__.py`, one dot means *this* package, not its parent.
        config = "class Settings:\n    pass\n"
        init = "from .config import Settings\n"
        edges = _edges(
            parse_python("pkg/config.py", config),
            parse_python("pkg/__init__.py", init),
        )
        imp = _find(edges, EdgeKind.IMPORTS, "pkg")
        assert imp.dst_fqn == "pkg.config.Settings"
        assert imp.confidence == confidence.EXACT

    def test_import_through_package_reexport_resolves_to_definition(self) -> None:
        base = "class Repository:\n    pass\n"
        init = "from .base import Repository\n"
        user = (
            "from pkg import Repository\n"
            "def build():\n"
            "    return Repository()\n"
        )
        edges = _edges(
            parse_python("pkg/base.py", base),
            parse_python("pkg/__init__.py", init),
            parse_python("app/user.py", user),
        )
        imp = _find(edges, EdgeKind.IMPORTS, "app.user")
        assert imp.dst_fqn == "pkg.base.Repository"
        call = _find(edges, EdgeKind.CALLS, "app.user.build")
        assert call.dst_fqn == "pkg.base.Repository"

    def test_star_reexport_is_followed(self) -> None:
        text = "def slugify(v):\n    return v\n"
        init = "from .text import *\n"
        user = "from pkg import slugify\ndef run():\n    return slugify('x')\n"
        edges = _edges(
            parse_python("pkg/text.py", text),
            parse_python("pkg/__init__.py", init),
            parse_python("app/user.py", user),
        )
        assert _find(edges, EdgeKind.IMPORTS, "app.user").dst_fqn == "pkg.text.slugify"

    def test_typescript_barrel_reexport_resolves(self) -> None:
        base = "export class Repository {\n  get(): void {}\n}\n"
        barrel = 'export { Repository } from "./base";\n'
        user = (
            'import { Repository } from "../store";\n'
            "export function build(): Repository {\n"
            "  return new Repository();\n"
            "}\n"
        )
        edges = _edges(
            parse_typescript("src/store/base.ts", base),
            parse_typescript("src/store/index.ts", barrel),
            parse_typescript("src/app/user.ts", user),
        )
        imp = _find(edges, EdgeKind.IMPORTS, "src/app/user")
        assert imp.dst_fqn == "src/store/base::Repository"

    def test_typescript_barrel_keeps_every_wildcard(self) -> None:
        # Two `export *` lines share the local name `*`; a map keyed by local
        # name would silently keep only the last one.
        text = "export function slugify(v: string): string {\n  return v;\n}\n"
        timing = "export function now(): number {\n  return 0;\n}\n"
        barrel = 'export * from "./text";\nexport * from "./timing";\n'
        user = (
            'import { slugify, now } from "../util";\n'
            "export function run(): void {\n"
            "  slugify(`${now()}`);\n"
            "}\n"
        )
        edges = _edges(
            parse_typescript("src/util/text.ts", text),
            parse_typescript("src/util/timing.ts", timing),
            parse_typescript("src/util/index.ts", barrel),
            parse_typescript("src/app/user.ts", user),
        )
        targets = {
            e.dst_fqn
            for e in edges
            if e.kind is EdgeKind.IMPORTS and e.src_fqn == "src/app/user"
        }
        assert targets == {"src/util/text::slugify", "src/util/timing::now"}


class TestReceiverResolutionDoesNotFabricate:
    """Precision guards: cases where the honest answer is "I don't know"."""

    def test_call_through_external_module_attribute_stays_external(self) -> None:
        # `os.environ.get()` must not match the `get` of some repository class.
        store = "class Store:\n    def get(self, k):\n        return k\n"
        main = "import os\ndef run():\n    return os.environ.get('X')\n"
        edges = _edges(
            parse_python("app/store.py", store), parse_python("app/main.py", main)
        )
        edge = _find(edges, EdgeKind.CALLS, "app.main.run")
        assert not edge.is_resolved
        assert edge.dst_unresolved_name == "os.environ.get"
        assert edge.confidence == confidence.EXTERNAL

    def test_self_attribute_call_does_not_bind_to_the_enclosing_class(self) -> None:
        # `self._conn.close()` inside `close()` is a *different* object's method.
        src = (
            "import sqlite3\n"
            "class Store:\n"
            "    def __init__(self, dsn):\n"
            "        self._conn = sqlite3.connect(dsn)\n"
            "    def close(self):\n"
            "        return self._conn.close()\n"
        )
        edges = _edges(parse_python("app/store.py", src))
        edge = _find(edges, EdgeKind.CALLS, "app.store.Store.close")
        assert not edge.is_resolved
        assert edge.dst_unresolved_name == "close"

    def test_self_attribute_call_still_reaches_a_collaborator(self) -> None:
        # Excluding the enclosing class must not blind the resolver entirely:
        # the collaborator's method is a different class and still resolves.
        repo = "class Repo:\n    def put(self, x):\n        return x\n"
        service = (
            "from app.repo import Repo\n"
            "class Service:\n"
            "    def __init__(self, repo):\n"
            "        self._repo = repo\n"
            "    def save(self, x):\n"
            "        return self._repo.put(x)\n"
        )
        edges = _edges(
            parse_python("app/repo.py", repo), parse_python("app/service.py", service)
        )
        edge = _find(edges, EdgeKind.CALLS, "app.service.Service.save")
        assert edge.dst_fqn == "app.repo.Repo.put"


class TestLocalConstructorInference:
    """`x = Concrete()` then `x.m()` resolves to `Concrete.m`, not to whichever
    same-named method sorts first."""

    def test_python_local_construction_selects_the_concrete_class(self) -> None:
        base = "class Base:\n    def put(self, x):\n        return x\n"
        memory = (
            "from app.base import Base\n"
            "class Memory(Base):\n"
            "    def put(self, x):\n"
            "        return x\n"
        )
        test = (
            "from app.memory import Memory\n"
            "def run():\n"
            "    repo = Memory()\n"
            "    return repo.put(1)\n"
        )
        edges = _edges(
            parse_python("app/base.py", base),
            parse_python("app/memory.py", memory),
            parse_python("app/run.py", test),
        )
        edge = _find(edges, EdgeKind.CALLS, "app.run.run", dst="app.memory.Memory.put")
        assert edge.confidence == confidence.INFERRED_LOCAL

    def test_python_inherited_member_is_found_through_the_local_type(self) -> None:
        base = "class Base:\n    def flush(self):\n        return 1\n"
        memory = "from app.base import Base\nclass Memory(Base):\n    pass\n"
        run = (
            "from app.memory import Memory\n"
            "def run():\n"
            "    repo = Memory()\n"
            "    return repo.flush()\n"
        )
        edges = _edges(
            parse_python("app/base.py", base),
            parse_python("app/memory.py", memory),
            parse_python("app/run.py", run),
        )
        edge = _find(edges, EdgeKind.CALLS, "app.run.run", dst="app.base.Base.flush")
        assert edge.confidence == confidence.INFERRED_LOCAL

    def test_factory_function_hint_is_ignored(self) -> None:
        # `service = make_service()` names a function, not a class: the hint
        # must be discarded rather than used to invent a type.
        module = (
            "class Service:\n"
            "    def run(self):\n"
            "        return 1\n"
            "def make_service():\n"
            "    return Service()\n"
            "def main():\n"
            "    service = make_service()\n"
            "    return service.run()\n"
        )
        edges = _edges(parse_python("app/m.py", module))
        edge = _find(edges, EdgeKind.CALLS, "app.m.main", dst="app.m.Service.run")
        assert edge.confidence == confidence.HEURISTIC_UNIQUE

    def test_typescript_local_construction_selects_the_concrete_class(self) -> None:
        base = "export class Base {\n  put(x: number): void {}\n}\n"
        memory = (
            'import { Base } from "./base";\n'
            "export class Memory extends Base {\n"
            "  put(x: number): void {}\n"
            "}\n"
        )
        test = (
            'import { Memory } from "../src/memory";\n'
            "export function run(): void {\n"
            "  const repo = new Memory();\n"
            "  repo.put(1);\n"
            "}\n"
        )
        edges = _edges(
            parse_typescript("src/base.ts", base),
            parse_typescript("src/memory.ts", memory),
            parse_typescript("tests/run.test.ts", test),
        )
        edge = _find(
            edges, EdgeKind.CALLS, "tests/run.test::run", dst="src/memory::Memory.put"
        )
        assert edge.confidence == confidence.INFERRED_LOCAL


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
