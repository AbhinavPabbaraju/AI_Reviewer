"""The M1 exit gate: symbol-resolution accuracy on a hand-labeled corpus.

ROADMAP.md M1 requires "symbol resolution rate >= 85% on a hand-labeled sample
... measured, not asserted." This module *measures* it: each corpus below is a
small, self-contained, correctly-imported repository, and each expectation is a
hand-labeled ground-truth edge -- the resolution a competent reader would make.
The test computes accuracy against that ground truth and gates on 0.85.

This is a representative seed, not yet the full 200-reference target the roadmap
sets for closing M1 -- but it clears the bar and, crucially, it is the mechanism
that will gate every future resolver change. Growing the corpus is adding entries
to these two lists.

An expectation is ``(kind, src_fqn, target)`` where ``target`` is a resolved
destination fqn, or ``EXTERNAL(name)`` for a reference that should be recognized
as leaving the repository (stdlib/vendor) rather than mis-resolved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.domain.indexing.models import EdgeKind, Language, ParsedFile, confidence
from app.domain.indexing.resolution import Resolver
from tests.conftest import parse_python, parse_typescript

RESOLUTION_TARGET = 0.85


@dataclass(frozen=True)
class EXTERNAL:
    name: str


type Target = str | EXTERNAL
type Expectation = tuple[EdgeKind, str, Target]


# --------------------------------------------------------------------------- #
# Python corpus
# --------------------------------------------------------------------------- #

PY_CORPUS: Mapping[str, str] = {
    "app/util.py": (
        "def greeting(name):\n"
        "    return 'hi ' + name\n"
        "\n"
        "def shout(text):\n"
        "    return text.upper()\n"
    ),
    "app/models.py": (
        "from app.util import greeting\n"
        "\n"
        "class Entity:\n"
        "    def identity(self):\n"
        "        return self._id\n"
        "\n"
        "class User(Entity):\n"
        "    def greet(self):\n"
        "        return greeting(self.identity())\n"
    ),
    "app/service.py": (
        "import os\n"
        "from app.models import User\n"
        "from app.util import shout\n"
        "\n"
        "class Service:\n"
        "    def run(self, name):\n"
        "        u = User(name)\n"
        "        return shout(u.greet())\n"
        "    def path(self):\n"
        "        return os.getcwd()\n"
    ),
    "app/handlers.py": (
        "from . import util as u\n"
        "from .models import User\n"
        "\n"
        "def handle(name):\n"
        "    user = User(name)\n"
        "    return u.greeting(name)\n"
    ),
    "tests/test_models.py": (
        "from app.models import User\n"
        "\n"
        "def test_greet():\n"
        "    return User('x').greet()\n"
    ),
}

PY_TEST_FILES = {"tests/test_models.py"}

PY_EXPECTATIONS: list[Expectation] = [
    # imports
    (EdgeKind.IMPORTS, "app.models", "app.util.greeting"),
    (EdgeKind.IMPORTS, "app.service", "app.models.User"),
    (EdgeKind.IMPORTS, "app.service", "app.util.shout"),
    (EdgeKind.IMPORTS, "app.service", EXTERNAL("os")),
    (EdgeKind.IMPORTS, "app.handlers", "app.util"),        # relative whole-module
    (EdgeKind.IMPORTS, "app.handlers", "app.models.User"),  # relative from-import
    (EdgeKind.IMPORTS, "tests.test_models", "app.models.User"),
    # inheritance
    (EdgeKind.INHERITS, "app.models.User", "app.models.Entity"),
    # calls that resolve within a class / to an imported symbol
    (EdgeKind.CALLS, "app.models.User.greet", "app.util.greeting"),
    (EdgeKind.CALLS, "app.models.User.greet", "app.models.Entity.identity"),  # self, inherited
    (EdgeKind.CALLS, "app.service.Service.run", "app.models.User"),
    (EdgeKind.CALLS, "app.service.Service.run", "app.util.shout"),
    (EdgeKind.CALLS, "app.service.Service.run", "app.models.User.greet"),  # u.greet heuristic
    (EdgeKind.CALLS, "app.service.Service.path", EXTERNAL("os.getcwd")),
    (EdgeKind.CALLS, "app.handlers.handle", "app.models.User"),
    (EdgeKind.CALLS, "app.handlers.handle", "app.util.greeting"),  # module alias u.greeting
    (EdgeKind.CALLS, "tests.test_models.test_greet", "app.models.User"),
    (EdgeKind.CALLS, "tests.test_models.test_greet", "app.models.User.greet"),
    # test edge
    (EdgeKind.TESTS, "tests.test_models.test_greet", "app.models.User.greet"),
]


# --------------------------------------------------------------------------- #
# TypeScript corpus
# --------------------------------------------------------------------------- #

TS_CORPUS: Mapping[str, str] = {
    "src/util.ts": (
        "export function greeting(name: string): string { return 'hi ' + name; }\n"
        "export function shout(text: string): string { return text.toUpperCase(); }\n"
    ),
    "src/models.ts": (
        'import { greeting } from "./util";\n'
        "export class Entity {\n"
        "  identity(): string { return this.id; }\n"
        "}\n"
        "export class User extends Entity {\n"
        "  greet(): string { return greeting(this.identity()); }\n"
        "}\n"
    ),
    "src/service.ts": (
        'import { User } from "./models";\n'
        'import { shout } from "./util";\n'
        'import * as util from "./util";\n'
        "export class Service {\n"
        "  run(name: string): string {\n"
        "    const u = new User(name);\n"
        "    return shout(u.greet());\n"
        "  }\n"
        "  viaNs(name: string): string { return util.greeting(name); }\n"
        "}\n"
    ),
    "src/external.ts": (
        'import { useState } from "react";\n'
        "export const useThing = () => useState();\n"
    ),
}

TS_EXPECTATIONS: list[Expectation] = [
    (EdgeKind.IMPORTS, "src/models", "src/util.greeting"),
    (EdgeKind.IMPORTS, "src/service", "src/models.User"),
    (EdgeKind.IMPORTS, "src/service", "src/util.shout"),
    (EdgeKind.IMPORTS, "src/service", "src/util"),  # namespace import -> module
    (EdgeKind.IMPORTS, "src/external", EXTERNAL("react.useState")),
    (EdgeKind.INHERITS, "src/models.User", "src/models.Entity"),
    (EdgeKind.CALLS, "src/models.User.greet", "src/util.greeting"),
    (EdgeKind.CALLS, "src/models.User.greet", "src/models.Entity.identity"),  # this, inherited
    (EdgeKind.CALLS, "src/service.Service.run", "src/models.User"),  # new User
    (EdgeKind.CALLS, "src/service.Service.run", "src/util.shout"),
    (EdgeKind.CALLS, "src/service.Service.run", "src/models.User.greet"),  # u.greet heuristic
    (EdgeKind.CALLS, "src/service.Service.viaNs", "src/util.greeting"),  # namespace member
    (EdgeKind.CALLS, "src/external.useThing", EXTERNAL("react.useState")),
]

# TS fqns use `::` between module id and nested name; the expectations above are
# written with `.` for readability and normalized here.
def _ts_str(value: str) -> str:
    if "/" in value and "::" not in value:
        module, _, rest = value.partition(".")
        return f"{module}::{rest}" if rest else module
    return value


def _ts(value: Target) -> Target:
    return value if isinstance(value, EXTERNAL) else _ts_str(value)


_TS_EXPECTATIONS_NORMALIZED: list[Expectation] = [
    (kind, _ts_str(src), _ts(target)) for kind, src, target in TS_EXPECTATIONS
]


def _resolve(corpus: Mapping[str, str], test_files: set[str], is_ts: bool):
    parsed: list[ParsedFile] = []
    for path, text in corpus.items():
        if is_ts:
            parsed.append(parse_typescript(path, text))
        else:
            parsed.append(parse_python(path, text, is_test=path in test_files))
    return Resolver(parsed).resolve()


def _accuracy(edges: object, expectations: list[Expectation]) -> tuple[float, list[Expectation]]:
    resolved = {
        (e.kind, e.src_fqn, e.dst_fqn) for e in edges if e.is_resolved  # type: ignore[attr-defined]
    }
    external = {
        (e.kind, e.src_fqn, e.dst_unresolved_name)
        for e in edges  # type: ignore[attr-defined]
        if not e.is_resolved and e.confidence == confidence.EXTERNAL
    }
    misses: list[Expectation] = []
    for kind, src, target in expectations:
        if isinstance(target, EXTERNAL):
            ok = (kind, src, target.name) in external
        else:
            ok = (kind, src, target) in resolved
        if not ok:
            misses.append((kind, src, target))
    accuracy = 1.0 - len(misses) / len(expectations)
    return accuracy, misses


class TestPythonResolutionRate:
    def test_labeled_accuracy_meets_target(self) -> None:
        result = _resolve(PY_CORPUS, PY_TEST_FILES, is_ts=False)
        accuracy, misses = _accuracy(result.edges, PY_EXPECTATIONS)
        assert accuracy >= RESOLUTION_TARGET, f"accuracy {accuracy:.2%}; missed {misses}"

    def test_operational_rate_meets_target(self) -> None:
        result = _resolve(PY_CORPUS, PY_TEST_FILES, is_ts=False)
        assert result.stats.rate_for(Language.PYTHON) >= RESOLUTION_TARGET


class TestTypeScriptResolutionRate:
    def test_labeled_accuracy_meets_target(self) -> None:
        result = _resolve(TS_CORPUS, set(), is_ts=True)
        accuracy, misses = _accuracy(result.edges, _TS_EXPECTATIONS_NORMALIZED)
        assert accuracy >= RESOLUTION_TARGET, f"accuracy {accuracy:.2%}; missed {misses}"

    def test_operational_rate_meets_target(self) -> None:
        result = _resolve(TS_CORPUS, set(), is_ts=True)
        assert result.stats.rate_for(Language.TYPESCRIPT) >= RESOLUTION_TARGET
