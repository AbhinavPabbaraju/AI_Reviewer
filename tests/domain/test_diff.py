"""Unified-diff parsing.

Two kinds of test here, deliberately. Hand-written diffs pin down the edge cases
(omitted counts, empty sides, renames, binaries, "\\ No newline") because those
are hard to provoke on demand from git. Then the same parser is run over **real
``git diff`` output** from a throwaway repository, because a hand-written corpus
only ever contains the shapes its author thought of, and this parser's line
numbers decide where review comments land.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

import pytest

from app.domain.review.diff import FileStatus, parse_unified_diff
from tests.conftest import GitRepo


def _diff(repo: GitRepo, *args: str) -> str:
    return subprocess.run(
        ["git", "diff", *args],
        cwd=repo.path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class TestHunkGeometry:
    def test_added_lines_are_numbered_in_the_new_file(self) -> None:
        diff = parse_unified_diff(
            """diff --git a/app/store.py b/app/store.py
--- a/app/store.py
+++ b/app/store.py
@@ -10,3 +10,4 @@ def save(value):
 context one
-old line
+new line
+another new line
 context two
"""
        )
        [file] = diff.files
        assert file.path == "app/store.py"
        assert file.status is FileStatus.MODIFIED
        # Line 10 is context, 11 and 12 are the additions.
        assert file.changed_lines == {11, 12}
        [hunk] = file.hunks
        assert hunk.removed_lines == (11,)
        assert hunk.heading == "def save(value):"

    def test_deletions_contribute_no_changed_lines(self) -> None:
        """A deleted line does not exist at head, so nothing can be anchored to
        it -- and GitHub would have nowhere to render the comment."""
        diff = parse_unified_diff(
            """diff --git a/app/store.py b/app/store.py
--- a/app/store.py
+++ b/app/store.py
@@ -10,3 +10,2 @@
 context
-removed line
 trailing
"""
        )
        [file] = diff.files
        assert file.changed_lines == frozenset()
        assert file.hunks[0].removed_lines == (11,)

    def test_omitted_counts_mean_one_line(self) -> None:
        diff = parse_unified_diff(
            """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -3 +3 @@
-old
+new
"""
        )
        assert diff.files[0].changed_lines == {3}

    def test_multiple_hunks_accumulate(self) -> None:
        diff = parse_unified_diff(
            """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,3 @@
 one
+two
 three
@@ -20,2 +21,3 @@
 twenty
+twenty one
 twenty two
"""
        )
        [file] = diff.files
        assert file.changed_lines == {2, 22}
        assert len(file.hunks) == 2

    def test_no_newline_marker_is_ignored(self) -> None:
        diff = parse_unified_diff(
            """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
\\ No newline at end of file
+new
\\ No newline at end of file
"""
        )
        assert diff.files[0].changed_lines == {1}

    def test_touched_lines_include_context(self) -> None:
        """The lenient yardstick: a finding may point at an unchanged guard
        inside the hunk, which is on the reviewer's screen."""
        diff = parse_unified_diff(
            """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -10,3 +10,4 @@
 guard
+added
 tail
 more
"""
        )
        [file] = diff.files
        assert file.changed_lines == {11}
        assert file.touched_lines == {10, 11, 12, 13}


class TestFileStatus:
    def test_added_file(self) -> None:
        diff = parse_unified_diff(
            """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+first
+second
"""
        )
        [file] = diff.files
        assert file.status is FileStatus.ADDED
        assert file.path == "new.py"
        assert file.changed_lines == {1, 2}

    def test_deleted_file_is_excluded_from_head_paths(self) -> None:
        diff = parse_unified_diff(
            """diff --git a/gone.py b/gone.py
deleted file mode 100644
--- a/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-first
-second
"""
        )
        [file] = diff.files
        assert file.status is FileStatus.DELETED
        assert file.path == "gone.py"
        assert diff.paths == frozenset()

    def test_rename_keeps_both_paths(self) -> None:
        diff = parse_unified_diff(
            """diff --git a/old/name.py b/new/name.py
similarity index 95%
rename from old/name.py
rename to new/name.py
--- a/old/name.py
+++ b/new/name.py
@@ -1 +1,2 @@
 keep
+added
"""
        )
        [file] = diff.files
        assert file.status is FileStatus.RENAMED
        assert file.path == "new/name.py"
        assert file.old_path == "old/name.py"

    def test_binary_file_is_flagged_and_has_no_hunks(self) -> None:
        diff = parse_unified_diff(
            """diff --git a/logo.png b/logo.png
index 1234..5678 100644
Binary files a/logo.png and b/logo.png differ
"""
        )
        [file] = diff.files
        assert file.is_binary
        assert file.hunks == ()

    def test_quoted_path_is_unquoted(self) -> None:
        diff = parse_unified_diff(
            '''diff --git "a/dir/caf\\303\\251.py" "b/dir/caf\\303\\251.py"
--- "a/dir/caf\\303\\251.py"
+++ "b/dir/caf\\303\\251.py"
@@ -1 +1,2 @@
 keep
+added
'''
        )
        assert diff.files[0].path.startswith("dir/caf")


class TestCorruptDiffsFailLoudly:
    def test_body_disagreeing_with_header_is_rejected(self) -> None:
        """Silently mis-numbering findings is far worse than failing: the line
        numbers here decide where a comment is rendered."""
        with pytest.raises(ValueError, match="refusing to guess line numbers"):
            parse_unified_diff(
                """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,5 +1,5 @@
 only one line
"""
            )

    def test_empty_diff_is_empty_not_an_error(self) -> None:
        assert parse_unified_diff("").files == ()
        assert parse_unified_diff("no diff headers here\n").files == ()


class TestAgainstRealGitOutput:
    """The corpus a hand-written suite cannot contain: whatever git emits."""

    def test_modification_line_numbers_match_the_file(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        original = "\n".join(f"line {n}" for n in range(1, 31)) + "\n"
        repo.commit({"app/mod.py": original})

        edited = original.replace("line 15", "line 15 CHANGED").replace(
            "line 25", "line 25 CHANGED"
        )
        (repo.path / "app/mod.py").write_text(edited)

        diff = parse_unified_diff(_diff(repo))
        [file] = diff.files
        assert file.path == "app/mod.py"
        assert file.changed_lines == {15, 25}

        # The parsed line numbers must actually index the edited file.
        lines = edited.splitlines()
        for number in sorted(file.changed_lines):
            assert lines[number - 1].endswith("CHANGED")

    def test_insertion_shifts_later_line_numbers(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """The case where naive parsing goes wrong: an early insertion moves
        every later line, and a finding anchored with old numbering lands on
        the wrong code."""
        repo = make_git_repo()
        original = "\n".join(f"line {n}" for n in range(1, 21)) + "\n"
        repo.commit({"app/mod.py": original})

        lines = original.splitlines()
        lines.insert(4, "inserted a")
        lines.insert(5, "inserted b")
        lines[15] = "line 15 CHANGED"
        edited = "\n".join(lines) + "\n"
        (repo.path / "app/mod.py").write_text(edited)

        diff = parse_unified_diff(_diff(repo))
        [file] = diff.files
        new_lines = edited.splitlines()
        for number in sorted(file.changed_lines):
            content = new_lines[number - 1]
            assert "inserted" in content or "CHANGED" in content

    def test_new_file_and_deletion_in_one_diff(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        repo.commit({"keep.py": "x = 1\n", "drop.py": "y = 2\n"})
        (repo.path / "added.py").write_text("z = 3\n")
        # Untracked files are invisible to `git diff`; stage so the new file
        # appears as an addition rather than silently not being reviewed.
        subprocess.run(["git", "add", "-A"], cwd=repo.path, check=True)
        repo.remove("drop.py")

        diff = parse_unified_diff(_diff(repo, "HEAD"))
        by_path = {file.path: file for file in diff.files}
        assert by_path["added.py"].status is FileStatus.ADDED
        assert by_path["added.py"].changed_lines == {1}
        assert by_path["drop.py"].status is FileStatus.DELETED
        assert "drop.py" not in diff.paths

    def test_multi_file_multi_hunk_totals(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        repo = make_git_repo()
        files = {
            f"pkg/mod{n}.py": "\n".join(f"line {i}" for i in range(1, 41)) + "\n"
            for n in range(4)
        }
        repo.commit(files)
        for name, text in files.items():
            lines = text.splitlines()
            lines[5] = "changed early"
            lines[30] = "changed late"
            (repo.path / name).write_text("\n".join(lines) + "\n")

        diff = parse_unified_diff(_diff(repo))
        assert len(diff.files) == 4
        assert diff.total_added_lines == 8
        for file in diff.files:
            assert file.changed_lines == {6, 31}
