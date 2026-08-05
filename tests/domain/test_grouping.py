"""Hunk grouping: one review unit per symbol.

Uses the real ``InMemorySymbolIndex`` over real parsed files rather than a stub
returning canned symbols. The interesting behaviour here is entirely about which
symbol encloses which line, and a fake that answered that question would be the
thing under test.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.domain.indexing.chunking import chunk_parsed_file
from app.domain.indexing.ports import SnapshotWrite
from app.domain.review.diff import parse_unified_diff
from app.domain.review.grouping import group_hunks
from app.infra.retrieval.memory import InMemorySymbolIndex
from tests.conftest import parse_python

SOURCE = '''"""Module docstring."""

CONSTANT = 1


class Store:
    """A store."""

    def read(self, name):
        value = self._lookup(name)
        return value

    def write(self, name, value):
        self._cache[name] = value
        return True


def helper(x):
    return x + 1
'''


@pytest.fixture
def index() -> tuple[InMemorySymbolIndex, str]:
    repository_id = uuid4()
    parsed = parse_python("app/store.py", SOURCE)
    chunks = chunk_parsed_file(repository_id, parsed, SOURCE)
    snapshot = SnapshotWrite(
        id=uuid4(),
        repository_id=repository_id,
        commit_sha="0" * 40,
        parent_snapshot_id=None,
        parser_version="treesitter/1",
        embedding_model="none",
        files=(parsed.file,),
        symbols=parsed.symbols,
        edges=(),
        chunks=tuple(chunks),
    )
    return InMemorySymbolIndex(snapshot), str(repository_id)


def _diff(*hunks: str) -> str:
    return "diff --git a/app/store.py b/app/store.py\n" + "".join(hunks)


class TestGrouping:
    async def test_hunks_in_one_function_become_one_unit(
        self, index: tuple[InMemorySymbolIndex, str]
    ) -> None:
        """Two edits inside `read` are one change and deserve one question."""
        symbol_index, repository_id = index
        diff = parse_unified_diff(
            _diff(
                """--- a/app/store.py
+++ b/app/store.py
@@ -10,1 +10,1 @@
-        value = self._lookup(name)
+        value = self._lookup(name.strip())
@@ -11,1 +11,1 @@
-        return value
+        return value or None
"""
            )
        )
        groups = await group_hunks(diff, symbol_index, repository_id)
        assert len(groups) == 1
        [group] = groups
        assert group.symbol_fqn is not None
        assert group.symbol_fqn.endswith("Store.read")
        assert len(group.hunks) == 2

    async def test_edits_to_two_functions_are_two_units(
        self, index: tuple[InMemorySymbolIndex, str]
    ) -> None:
        symbol_index, repository_id = index
        diff = parse_unified_diff(
            _diff(
                """--- a/app/store.py
+++ b/app/store.py
@@ -10,1 +10,1 @@
-        value = self._lookup(name)
+        value = self._lookup(name.strip())
@@ -14,1 +14,1 @@
-        self._cache[name] = value
+        self._cache[name] = value or ""
"""
            )
        )
        groups = await group_hunks(diff, symbol_index, repository_id)
        assert len(groups) == 2
        assert [g.symbol_fqn.rsplit(".", 1)[-1] for g in groups if g.symbol_fqn] == [
            "read",
            "write",
        ]

    async def test_anchor_span_is_the_whole_symbol_not_the_hunk(
        self, index: tuple[InMemorySymbolIndex, str]
    ) -> None:
        """Retrieval anchors on this span, and a model judging one line of a
        function without seeing the rest is guessing."""
        symbol_index, repository_id = index
        diff = parse_unified_diff(
            _diff(
                """--- a/app/store.py
+++ b/app/store.py
@@ -10,1 +10,1 @@
-        value = self._lookup(name)
+        value = self._lookup(name.strip())
"""
            )
        )
        [group] = await group_hunks(diff, symbol_index, repository_id)
        assert group.span.line_count > 1
        assert group.span.line_start <= 10 <= group.span.line_end

    async def test_module_level_change_is_grouped_without_a_symbol(
        self, index: tuple[InMemorySymbolIndex, str]
    ) -> None:
        """A tweak to a module-level constant has no enclosing function. It is
        still reviewed; it just anchors on the file."""
        symbol_index, repository_id = index
        diff = parse_unified_diff(
            _diff(
                """--- a/app/store.py
+++ b/app/store.py
@@ -3,1 +3,1 @@
-CONSTANT = 1
+CONSTANT = 2
"""
            )
        )
        [group] = await group_hunks(diff, symbol_index, repository_id)
        assert group.symbol_fqn is None
        assert not group.is_anchored
        assert group.path == "app/store.py"

    async def test_the_innermost_symbol_wins_over_its_class(
        self, index: tuple[InMemorySymbolIndex, str]
    ) -> None:
        """A hunk inside a method belongs to the method. Anchoring on `Store`
        would retrieve the class's neighbours rather than the method's."""
        symbol_index, repository_id = index
        diff = parse_unified_diff(
            _diff(
                """--- a/app/store.py
+++ b/app/store.py
@@ -14,1 +14,1 @@
-        self._cache[name] = value
+        self._cache[name] = value or ""
"""
            )
        )
        [group] = await group_hunks(diff, symbol_index, repository_id)
        assert group.symbol_fqn is not None
        assert group.symbol_fqn.endswith("write")
        assert not group.symbol_fqn.endswith("Store")

    async def test_deleted_and_binary_files_are_not_reviewed(
        self, index: tuple[InMemorySymbolIndex, str]
    ) -> None:
        symbol_index, repository_id = index
        diff = parse_unified_diff(
            """diff --git a/gone.py b/gone.py
--- a/gone.py
+++ /dev/null
@@ -1,1 +0,0 @@
-x = 1
diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
"""
        )
        assert await group_hunks(diff, symbol_index, repository_id) == ()

    async def test_empty_diff_produces_no_units(
        self, index: tuple[InMemorySymbolIndex, str]
    ) -> None:
        symbol_index, repository_id = index
        assert await group_hunks(
            parse_unified_diff(""), symbol_index, repository_id
        ) == ()

    async def test_grouping_makes_one_index_round_trip(
        self, index: tuple[InMemorySymbolIndex, str]
    ) -> None:
        """The retrieval ports are set-at-a-time; a 40-file PR must not become
        40 queries before the review has started."""
        symbol_index, repository_id = index
        calls = 0
        original = symbol_index.symbols_in_spans

        async def counting(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return await original(*args, **kwargs)  # type: ignore[arg-type]

        symbol_index.symbols_in_spans = counting  # type: ignore[method-assign]
        diff = parse_unified_diff(
            _diff(
                """--- a/app/store.py
+++ b/app/store.py
@@ -10,1 +10,1 @@
-        value = self._lookup(name)
+        value = self._lookup(name.strip())
@@ -14,1 +14,1 @@
-        self._cache[name] = value
+        self._cache[name] = value or ""
@@ -19,1 +19,1 @@
-    return x + 1
+    return x + 2
"""
            )
        )
        groups = await group_hunks(diff, symbol_index, repository_id)
        assert len(groups) == 3
        assert calls == 1
