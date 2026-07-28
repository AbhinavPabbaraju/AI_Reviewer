"""Shared tree-sitter plumbing for the language parsers.

Grammars are loaded once at import and reused; ``tree_sitter.Parser`` is cheap to
construct per call and not thread-safe to share, so parsers build one per parse.
Everything here is deliberately thin -- the language-specific extraction lives in
``python.py`` / ``typescript.py``; this module only wraps the byte-oriented
tree-sitter API in line-oriented, ``str``-oriented helpers the extractors want.
"""

from __future__ import annotations

from functools import cache

import tree_sitter as ts
import tree_sitter_python as tsp
import tree_sitter_typescript as tst

from app.domain.contracts import CodeSpan

__all__ = [
    "child_text",
    "make_parser",
    "node_text",
    "span_of",
]


@cache
def _language(name: str) -> ts.Language:
    if name == "python":
        return ts.Language(tsp.language())
    if name == "typescript":
        return ts.Language(tst.language_typescript())
    if name == "tsx":
        return ts.Language(tst.language_tsx())
    raise ValueError(f"unknown grammar: {name}")


def make_parser(grammar: str) -> ts.Parser:
    """A fresh parser bound to the named grammar ('python', 'typescript', 'tsx')."""
    return ts.Parser(_language(grammar))


def node_text(node: ts.Node, source: bytes) -> str:
    """The source text a node spans, decoded leniently (cloned code may not be
    valid UTF-8, and a decode error must never crash indexing)."""
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


def child_text(node: ts.Node, field: str, source: bytes) -> str | None:
    child = node.child_by_field_name(field)
    return node_text(child, source) if child is not None else None


def span_of(node: ts.Node, path: str) -> CodeSpan:
    """A 1-indexed inclusive line span. tree-sitter points are 0-indexed (row,
    column); the graph, git and GitHub are all 1-indexed, so convert at the edge."""
    return CodeSpan(
        path=path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
    )
