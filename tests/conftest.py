"""Shared fixtures and builders for the test suite.

Two things every test needs: a cheap way to turn source text into a
``SourceFile``/``ParsedFile`` without a git round-trip, and a real on-disk git
repository to exercise the source provider and the indexer end to end.

The Postgres fixtures live in ``tests/pg.py`` and are re-exported here so that
``pg_pool``/``pg_repository`` resolve by name in any test module. They skip
themselves when ``ARGUS_TEST_DATABASE_URL`` is unset.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from app.domain.indexing.models import Language, ParsedFile, SourceFile
from app.infra.parsing.python import PythonParser
from app.infra.parsing.typescript import TypeScriptParser
from tests.pg import pg_dsn, pg_pool, pg_repository

__all__ = ["pg_dsn", "pg_pool", "pg_repository"]

_PY = PythonParser()
_TS = TypeScriptParser()


def blob_sha(content: bytes) -> str:
    """A deterministic 40-hex id for content. Not git's blob hash (the parser
    tests do not care), just a stable, schema-valid ``blob_sha``."""
    return hashlib.sha1(content).hexdigest()


def make_source_file(
    path: str, content: bytes, *, is_test: bool = False, is_generated: bool = False
) -> SourceFile:
    language = Language.for_path(path)
    assert language is not None, f"no language for {path}"
    return SourceFile(
        path=path,
        language=language,
        blob_sha=blob_sha(content),
        size_bytes=len(content),
        line_count=content.count(b"\n") + 1,
        is_test=is_test,
        is_generated=is_generated,
    )


def parse_python(path: str, text: str, *, is_test: bool = False) -> ParsedFile:
    content = text.encode("utf-8")
    return _PY.parse(make_source_file(path, content, is_test=is_test), content)


def parse_typescript(path: str, text: str, *, is_test: bool = False) -> ParsedFile:
    content = text.encode("utf-8")
    return _TS.parse(make_source_file(path, content, is_test=is_test), content)


class GitRepo:
    """A throwaway git repository under a temp dir, committed to programmatically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._run("init", "-q", "-b", "main")
        self._run("config", "user.email", "test@argus.dev")
        self._run("config", "user.name", "Argus Test")

    @property
    def url(self) -> str:
        return str(self.path)

    def commit(self, files: Mapping[str, str | bytes], message: str = "commit") -> str:
        for rel, content in files.items():
            target = self.path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content)
        self._run("add", "-A")
        self._run("commit", "-q", "-m", message, "--allow-empty")
        return self._run("rev-parse", "HEAD").strip()

    def remove(self, *paths: str) -> None:
        self._run("rm", "-q", *paths)

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


@pytest.fixture
def make_git_repo(tmp_path: Path) -> Callable[[], GitRepo]:
    counter = {"n": 0}

    def _factory() -> GitRepo:
        counter["n"] += 1
        repo_path = tmp_path / f"repo-{counter['n']}"
        repo_path.mkdir()
        return GitRepo(repo_path)

    return _factory
