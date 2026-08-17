"""The command-line surface: git plumbing, output, and the whole chain.

The CLI is the part of Argus a person touches, so the things tested here are the
things that make it usable rather than merely correct: that it works out what to
review without being told, that it says something helpful when the environment
is not ready, and that it never moves the user's HEAD.
"""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.contracts import (
    Category,
    CodeSpan,
    Evidence,
    EvidenceRole,
    Finding,
    FindingSource,
    Severity,
    VerificationGate,
)
from argus_cli import repo
from argus_cli.main import build_parser, main
from argus_cli.render import Palette, render_findings, render_json, render_summary
from tests.conftest import GitRepo

MODULE = "pkg/service.py"
ORIGINAL = "def handle(value):\n    return value + 1\n"
EDITED = "def handle(value):\n    return value + 2\n"

# A port nothing is listening on, so "Ollama is not available" is a fact about
# the test rather than about whoever is running it.
DEAD_OLLAMA = "http://127.0.0.1:9"


def _finding(**overrides: object) -> Finding:
    span = CodeSpan(path=MODULE, line_start=2, line_end=2)
    defaults: dict[str, object] = {
        "run_id": uuid4(),
        "severity": Severity.HIGH,
        "category": Category.CORRECTNESS,
        "source": FindingSource.LLM,
        "title": "Return value is off by one",
        "explanation": "The increment was changed from 1 to 2 with no caller updated.",
        "location": span,
        "evidence": (
            Evidence(span=span, role=EvidenceRole.DEFECT_SITE, excerpt="return value + 2"),
        ),
        "confidence": 0.9,
        "prompt_version": "reviewer/v1",
    }
    return Finding(**(defaults | overrides))  # type: ignore[arg-type]


class TestArgumentParsing:
    def test_review_needs_no_arguments(self) -> None:
        """The daily path is a bare `argus review`. Every option has a default
        that does something sensible, or the tool is not one you reach for."""
        args = build_parser().parse_args(["review"])
        assert args.command == "review"
        assert args.path == Path()
        assert args.base is None
        assert args.fail_on == "never"

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_fail_on_accepts_severities_and_never(self) -> None:
        """`--fail-on high` is the CI shape; `never` is the default, because a
        review tool that breaks your build by default gets uninstalled."""
        assert build_parser().parse_args(["review", "--fail-on", "high"]).fail_on
        with pytest.raises(SystemExit):
            build_parser().parse_args(["review", "--fail-on", "nonsense"])


class TestRepositoryDiscovery:
    def test_finds_the_root_from_a_subdirectory(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL})
        nested = git_repo.path / "pkg"
        assert repo.find_repository_root(nested) == git_repo.path.resolve()

    def test_a_non_repository_says_so_and_says_what_to_do(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(repo.RepoError) as caught:
            repo.find_repository_root(tmp_path)
        assert "not inside a git repository" in str(caught.value)
        assert "git init" in str(caught.value)

    def test_default_base_prefers_the_repositorys_own_default_branch(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL})
        assert repo.default_base_ref(git_repo.path) == "main"

    def test_diff_includes_uncommitted_work(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """The point of the default: review what you are about to commit."""
        git_repo = make_git_repo()
        base = git_repo.commit({MODULE: ORIGINAL})
        (git_repo.path / MODULE).write_text(EDITED)

        diff = repo.diff_against(git_repo.path, base)
        assert "value + 2" in diff
        assert MODULE in diff

    def test_summarize_counts_files_and_added_lines(self) -> None:
        diff = (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
            "@@ -1 +1,2 @@\n x = 1\n+y = 2\n"
        )
        assert repo.summarize_diff(diff) == (1, 1)


class TestRendering:
    def test_a_finding_renders_with_its_evidence(self) -> None:
        out = io.StringIO()
        render_findings([_finding()], palette=Palette(), out=out)
        text = out.getvalue()
        assert "HIGH" in text
        assert f"{MODULE}:2" in text
        assert "Return value is off by one" in text
        assert "return value + 2" in text

    def test_findings_are_ordered_most_severe_first(self) -> None:
        out = io.StringIO()
        render_findings(
            [_finding(severity=Severity.LOW, title="Naming could be clearer here"),
             _finding(severity=Severity.CRITICAL, title="Credentials are logged")],
            palette=Palette(),
            out=out,
        )
        text = out.getvalue()
        assert text.index("CRITICAL") < text.index("LOW")

    def test_silence_is_reported_as_a_result_not_an_absence(self) -> None:
        """"No comments" has to look like a decision. A tool that prints nothing
        is indistinguishable from one that crashed."""
        out = io.StringIO()
        render_summary(
            posted=[],
            suppressed=[],
            drops={},
            decode_rejects=0,
            failures=[],
            completeness=1.0,
            units=3,
            seconds=12.0,
            model="qwen2.5-coder:7b",
            palette=Palette(),
            out=out,
        )
        assert "No comments" in out.getvalue()

    def test_suppressions_are_explained_in_english(self) -> None:
        out = io.StringIO()
        render_summary(
            posted=[_finding()],
            suppressed=[_finding(confidence=0.1)],
            drops={VerificationGate.LINE_IN_RANGE: 2},
            decode_rejects=1,
            failures=[],
            completeness=1.0,
            units=2,
            seconds=30.0,
            model="m",
            palette=Palette(),
            out=out,
        )
        text = out.getvalue()
        assert "cited a line outside the change" in text
        assert "malformed response" in text

    def test_an_incomplete_review_says_so(self) -> None:
        out = io.StringIO()
        render_summary(
            posted=[],
            suppressed=[],
            drops={},
            decode_rejects=0,
            failures=["pkg/service.py: TimeoutError: timed out"],
            completeness=0.5,
            units=2,
            seconds=9.0,
            model="m",
            palette=Palette(),
            out=out,
        )
        assert "incomplete" in out.getvalue()

    def test_json_output_is_the_contract_shape(self) -> None:
        """Serialized through the contract's own dump, so the CLI, the API and
        the frontend's generated types cannot drift apart."""
        out = io.StringIO()
        render_json(
            posted=[_finding()],
            suppressed=[],
            drops={VerificationGate.CONFIDENCE_FLOOR: 1},
            completeness=1.0,
            model="m",
            out=out,
        )
        payload = json.loads(out.getvalue())
        assert payload["cost_usd"] == 0.0
        assert payload["drops_by_gate"] == {"confidence_floor": 1}
        [finding] = payload["findings"]
        assert finding["location"]["path"] == MODULE
        assert finding["evidence"][0]["role"] == "defect_site"

    def test_colour_is_off_when_not_a_terminal(self) -> None:
        assert Palette.for_stream(io.StringIO()) == Palette()


class TestEndToEnd:
    def test_a_review_runs_the_pipeline_and_stops_at_a_missing_model(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """Everything except the model call, against a real repository: discover
        the root, work out the base, diff, index, group and retrieve. It then
        exits 3 because there is no Ollama -- which is the honest outcome, and a
        distinct code so a script can tell "not set up" from "found bugs"."""
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL, "pkg/__init__.py": ""})
        (git_repo.path / MODULE).write_text(EDITED)

        code = main([
            "review", str(git_repo.path), "--ollama-url", DEAD_OLLAMA, "--no-color",
        ])
        assert code == 3

    def test_an_empty_diff_exits_cleanly(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL})
        assert main(["review", str(git_repo.path), "--ollama-url", DEAD_OLLAMA]) == 0

    def test_reviewing_a_non_repository_is_a_clear_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["review", str(tmp_path)]) == 2
        assert "not inside a git repository" in capsys.readouterr().err

    def test_the_review_never_moves_your_head(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """The one behaviour that would make it unusable on a working day."""
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL})
        (git_repo.path / MODULE).write_text(EDITED)
        before = repo.head_sha(git_repo.path)

        main(["review", str(git_repo.path), "--ollama-url", DEAD_OLLAMA])

        assert repo.head_sha(git_repo.path) == before
        assert (git_repo.path / MODULE).read_text() == EDITED


class StubOllama:
    """An HTTP server that answers like Ollama, for one finding.

    The last seam the tests above leave open is the real one: the CLI building a
    provider, speaking HTTP to it, decoding what comes back and putting it
    through the gate. Mocking the adapter would skip exactly that. This is a
    socket, and everything from the request body to the exit code is real.
    """

    def __init__(self, content: str) -> None:
        self._content = content
        self.chat_calls = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> StubOllama:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _send(self, payload: dict[str, object]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._send({"models": [{"name": "stub"}]})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                stub.chat_calls += 1
                self._send(
                    {
                        "model": "stub",
                        "message": {"content": stub._content},
                        "prompt_eval_count": 1200,
                        "eval_count": 90,
                    }
                )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()


def _model_says(path: str, line: int, *, severity: str = "high") -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "severity": severity,
                    "category": "correctness",
                    "title": "Increment changed from 1 to 2 without updating callers",
                    "explanation": (
                        "`handle` now returns value + 2. Every caller that relied "
                        "on the previous increment is off by one."
                    ),
                    "path": path,
                    "line_start": line,
                    "line_end": line,
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "path": path,
                            "line_start": line,
                            "line_end": line,
                            "role": "defect_site",
                            "excerpt": "return value + 2",
                        }
                    ],
                }
            ]
        }
    )


class TestAgainstAModel:
    def test_a_real_finding_reaches_the_terminal(
        self,
        make_git_repo: Callable[[], GitRepo],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL, "pkg/__init__.py": ""})
        (git_repo.path / MODULE).write_text(EDITED)

        with StubOllama(_model_says(MODULE, 2)) as ollama:
            code = main([
                "review", str(git_repo.path),
                "--ollama-url", ollama.url, "--model", "stub", "--no-color",
            ])
            assert ollama.chat_calls >= 1

        out = capsys.readouterr().out
        assert code == 0
        assert "HIGH" in out
        assert "Increment changed from 1 to 2" in out
        assert "$0.00" in out

    def test_a_fabricated_line_never_reaches_the_terminal(
        self,
        make_git_repo: Callable[[], GitRepo],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The product promise, end to end through the command a user runs."""
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL, "pkg/__init__.py": ""})
        (git_repo.path / MODULE).write_text(EDITED)

        with StubOllama(_model_says(MODULE, 4096)) as ollama:
            code = main([
                "review", str(git_repo.path),
                "--ollama-url", ollama.url, "--model", "stub", "--no-color",
            ])

        out = capsys.readouterr().out
        assert code == 0
        assert "No comments" in out
        assert "cited a line outside the change" in out

    def test_fail_on_turns_a_finding_into_an_exit_code(
        self, make_git_repo: Callable[[], GitRepo]
    ) -> None:
        """The CI shape: `argus review --fail-on high` breaks the build."""
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL, "pkg/__init__.py": ""})
        (git_repo.path / MODULE).write_text(EDITED)

        with StubOllama(_model_says(MODULE, 2)) as ollama:
            code = main([
                "review", str(git_repo.path), "--fail-on", "high",
                "--ollama-url", ollama.url, "--model", "stub", "--no-color",
            ])
        assert code == 1

    def test_json_output_is_parseable_and_carries_the_finding(
        self,
        make_git_repo: Callable[[], GitRepo],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        git_repo = make_git_repo()
        git_repo.commit({MODULE: ORIGINAL, "pkg/__init__.py": ""})
        (git_repo.path / MODULE).write_text(EDITED)

        with StubOllama(_model_says(MODULE, 2)) as ollama:
            main([
                "review", str(git_repo.path), "--json",
                "--ollama-url", ollama.url, "--model", "stub",
            ])

        payload = json.loads(capsys.readouterr().out)
        assert payload["findings"][0]["location"]["line_start"] == 2
        assert payload["cost_usd"] == 0.0
