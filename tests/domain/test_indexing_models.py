"""Invariants of the indexing value objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.contracts import CodeSpan
from app.domain.indexing.models import (
    Chunk,
    EdgeKind,
    Import,
    Language,
    SymbolEdge,
    confidence,
)


class TestLanguageDetection:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("app/main.py", Language.PYTHON),
            ("app/types.pyi", Language.PYTHON),
            ("src/app.ts", Language.TYPESCRIPT),
            ("src/app.tsx", Language.TYPESCRIPT),
            ("src/app.mts", Language.TYPESCRIPT),
            ("legacy/app.js", Language.TYPESCRIPT),
            ("README.md", None),
            ("Makefile", None),
            ("data.json", None),
        ],
    )
    def test_for_path(self, path: str, expected: Language | None) -> None:
        assert Language.for_path(path) is expected


class TestSymbolEdge:
    def test_resolved_edge_reports_resolved(self) -> None:
        edge = SymbolEdge(
            kind=EdgeKind.CALLS, src_fqn="a.f", dst_fqn="a.g", confidence=1.0
        )
        assert edge.is_resolved

    def test_unresolved_edge_keeps_textual_target(self) -> None:
        edge = SymbolEdge(
            kind=EdgeKind.CALLS,
            src_fqn="a.f",
            dst_unresolved_name="os.getcwd",
            confidence=confidence.EXTERNAL,
        )
        assert not edge.is_resolved
        assert edge.dst_unresolved_name == "os.getcwd"

    def test_targetless_edge_is_rejected(self) -> None:
        """A targetless edge is a *dropped* edge; the model forbids it, mirroring
        the DDL's edge_has_a_target constraint."""
        with pytest.raises(ValidationError, match="targetless edge"):
            SymbolEdge(kind=EdgeKind.CALLS, src_fqn="a.f", confidence=0.3)

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            SymbolEdge(
                kind=EdgeKind.CALLS, src_fqn="a", dst_fqn="b", confidence=1.5
            )


class TestImport:
    def test_absolute_import_target_name(self) -> None:
        imp = Import(local_name="c", module="a.b", imported_symbol="c", line=1)
        assert imp.target_name == "a.b.c"

    def test_whole_module_target_name(self) -> None:
        imp = Import(local_name="os", module="os", line=1)
        assert imp.target_name == "os"

    def test_bare_relative_import_allows_empty_module(self) -> None:
        imp = Import(
            local_name="x", module="", imported_symbol="x", is_relative=True,
            level=1, line=1,
        )
        assert imp.target_name == "x"

    def test_absolute_import_must_name_a_module(self) -> None:
        with pytest.raises(ValidationError, match="must name a module"):
            Import(local_name="x", module="", line=1)


class TestChunk:
    def test_hash_must_be_64_hex(self) -> None:
        span = CodeSpan(path="a.py", line_start=1, line_end=2)
        with pytest.raises(ValidationError):
            Chunk(
                content_hash="tooshort",
                span=span,
                language=Language.PYTHON,
                token_count=3,
                content="x = 1",
            )

    def test_token_count_must_be_positive(self) -> None:
        span = CodeSpan(path="a.py", line_start=1, line_end=1)
        with pytest.raises(ValidationError):
            Chunk(
                content_hash="a" * 64,
                span=span,
                language=Language.PYTHON,
                token_count=0,
                content="x",
            )
