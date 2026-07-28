"""The blob-sha diff that drives incremental re-indexing (sec. 4.1)."""

from __future__ import annotations

from app.domain.indexing.incremental import plan_index


class TestPlanIndex:
    def test_first_index_is_all_added(self) -> None:
        plan = plan_index({"a.py": "1", "b.py": "2"}, {})
        assert plan.added == ("a.py", "b.py")
        assert plan.modified == ()
        assert plan.removed == ()
        assert plan.unchanged == ()
        assert plan.is_full_reindex

    def test_single_file_change_is_isolated(self) -> None:
        previous = {"a.py": "1", "b.py": "2", "c.py": "3"}
        current = {"a.py": "1", "b.py": "CHANGED", "c.py": "3"}
        plan = plan_index(current, previous)
        assert plan.modified == ("b.py",)
        assert plan.unchanged == ("a.py", "c.py")
        assert plan.to_parse == ("b.py",)
        assert not plan.is_full_reindex
        assert plan.reused_count == 2

    def test_added_and_removed(self) -> None:
        plan = plan_index({"a.py": "1", "new.py": "9"}, {"a.py": "1", "old.py": "5"})
        assert plan.added == ("new.py",)
        assert plan.removed == ("old.py",)
        assert plan.unchanged == ("a.py",)

    def test_to_parse_is_added_plus_modified_sorted(self) -> None:
        plan = plan_index(
            {"z.py": "new", "a.py": "1", "m.py": "changed"},
            {"a.py": "1", "m.py": "old"},
        )
        assert plan.to_parse == ("m.py", "z.py")

    def test_buckets_are_disjoint_and_total(self) -> None:
        previous = {"keep": "1", "change": "1", "drop": "1"}
        current = {"keep": "1", "change": "2", "add": "1"}
        plan = plan_index(current, previous)
        all_paths = {*plan.added, *plan.modified, *plan.removed, *plan.unchanged}
        assert all_paths == {"keep", "change", "drop", "add"}
        assert len(all_paths) == 4  # no overlaps
