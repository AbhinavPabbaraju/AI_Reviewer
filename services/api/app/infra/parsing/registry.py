"""The default language -> parser mapping the indexer is wired with.

Kept trivially small on purpose: adding a language is adding a parser here plus a
grammar, and ARCHITECTURE sec. 2 is explicit that this should stay a deliberate,
two-entry decision rather than a sprawling plugin system.
"""

from __future__ import annotations

from app.domain.indexing.models import Language
from app.domain.indexing.ports import ParserPort
from app.infra.parsing.python import PythonParser
from app.infra.parsing.typescript import TypeScriptParser

__all__ = ["default_parsers"]


def default_parsers() -> dict[Language, ParserPort]:
    return {
        Language.PYTHON: PythonParser(),
        Language.TYPESCRIPT: TypeScriptParser(),
    }
