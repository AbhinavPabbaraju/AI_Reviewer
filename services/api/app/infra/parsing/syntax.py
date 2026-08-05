"""Tree-sitter :class:`SyntaxCheckerPort` for the ``PATCH_PARSES`` gate.

Answers one question -- does this fragment contain a syntax error? -- which is
narrower than parsing and therefore needs care in one specific way: tree-sitter
is *error-tolerant by design*. It always returns a tree, marking the parts it
could not understand with ERROR and MISSING nodes rather than raising. Checking
"did we get a tree back" would accept everything; the check has to be for those
markers.

The other subtlety is that a suggested fix is rarely a whole file. It is a few
lines lifted out of a method body, so it arrives indented and, in Python,
indentation is syntax. The verifier retries dedented for that reason; this module
additionally treats an empty or whitespace-only fragment as *not* parsing, since
a patch that replaces code with nothing is a suggestion no reviewer asked for.
"""

from __future__ import annotations

import tree_sitter as ts

from app.domain.indexing.models import Language
from app.infra.parsing.base import make_parser

__all__ = ["TreeSitterSyntaxChecker"]

_GRAMMARS: dict[Language, str] = {
    Language.PYTHON: "python",
    Language.TYPESCRIPT: "typescript",
}


class TreeSitterSyntaxChecker:
    """Implements :class:`app.domain.review.ports.SyntaxCheckerPort`."""

    def parses(self, language: Language, source: str) -> bool:
        if not source.strip():
            return False
        grammar = _GRAMMARS.get(language)
        if grammar is None:  # pragma: no cover - both languages are mapped
            return True
        tree = make_parser(grammar).parse(source.encode("utf-8"))
        if not _has_error(tree.root_node):
            return True
        # TSX is a superset of TypeScript, and a suggestion touching JSX parses
        # only under that dialect. Retrying is cheaper than asking the caller to
        # know which dialect the file was indexed with.
        if language is Language.TYPESCRIPT:
            retry = make_parser("tsx").parse(source.encode("utf-8"))
            return not _has_error(retry.root_node)
        return False


def _has_error(root: ts.Node) -> bool:
    """Whether the tree contains an ERROR or MISSING node.

    ``Node.has_error`` covers the subtree, so this is a single check at the root
    rather than a walk -- but it is the *root's* flag that matters, and reading
    only ``root.type == "ERROR"`` would miss an error nested inside an otherwise
    well-formed statement.
    """
    return bool(root.has_error)
