"""Symbol-boundary chunking and content-addressed reuse (sec. 4.1)."""

from __future__ import annotations

from uuid import uuid4

from app.domain.indexing.chunking import chunk_parsed_file, content_hash
from tests.conftest import parse_python

_SOURCE = '''"""Module."""


class Store:
    """A store."""

    name: str = "x"

    def save(self, item):
        return item

    def load(self, key):
        return self.save(key)


def top():
    return Store()
'''


def _chunks(repo_id=None, source: str = _SOURCE):
    repo_id = repo_id or uuid4()
    parsed = parse_python("app/store.py", source)
    return {c.symbol_fqn: c for c in chunk_parsed_file(repo_id, parsed, source)}


class TestSymbolBoundaries:
    def test_one_chunk_per_definition(self) -> None:
        chunks = _chunks()
        assert set(chunks) == {
            "app.store.Store",
            "app.store.Store.save",
            "app.store.Store.load",
            "app.store.top",
        }

    def test_method_body_only_in_method_chunk(self) -> None:
        chunks = _chunks()
        assert "self.save(key)" in chunks["app.store.Store.load"].content
        # the class chunk keeps its header/attributes but not method bodies
        class_chunk = chunks["app.store.Store"].content
        assert "class Store" in class_chunk
        assert "name: str" in class_chunk
        assert "self.save(key)" not in class_chunk

    def test_module_symbol_is_not_chunked(self) -> None:
        assert "app.store" not in _chunks()


class TestContentAddressing:
    def test_hash_is_stable_across_runs(self) -> None:
        repo = uuid4()
        first = _chunks(repo)
        second = _chunks(repo)
        assert (
            first["app.store.top"].content_hash
            == second["app.store.top"].content_hash
        )

    def test_hash_is_repo_scoped(self) -> None:
        a = _chunks(uuid4())["app.store.top"].content_hash
        b = _chunks(uuid4())["app.store.top"].content_hash
        assert a != b  # same code, different tenant -> different id

    def test_trailing_whitespace_does_not_change_hash(self) -> None:
        repo = uuid4()
        clean = _chunks(repo)["app.store.Store.save"].content_hash
        noisy_source = _SOURCE.replace(
            "        return item", "        return item    "
        )
        noisy = _chunks(repo, noisy_source)["app.store.Store.save"].content_hash
        assert clean == noisy

    def test_body_change_changes_hash(self) -> None:
        repo = uuid4()
        base = _chunks(repo)["app.store.Store.save"].content_hash
        changed_source = _SOURCE.replace("return item", "return item + 1")
        changed = _chunks(repo, changed_source)["app.store.Store.save"].content_hash
        assert base != changed

    def test_content_hash_helper_is_sha256_hex(self) -> None:
        digest = content_hash(uuid4(), "a.py", "a.f", "body")
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
