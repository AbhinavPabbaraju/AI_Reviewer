"""``argus`` -- review a change with a local model, for free.

The whole pipeline from a terminal: index the tree, diff it, retrieve context
for each changed symbol, review with a model running on your own hardware, put
every claim through the verification gate, and print what survived.

Three things this command is built around.

**It reviews what you have not committed yet.** That is the moment a review is
worth most, and it is the reason the tree is indexed from disk rather than from
a commit -- see ``WorkingTreeSource``. ``--base`` and ``--pr`` cover the other
two moments.

**Nothing is configured.** No database, no API key, no config file. Storage is
in-memory, the embedder is offline, the model is whatever Ollama has pulled.
The cost of a run is zero and that is checked rather than promised.

**It never touches your checkout.** ``--pr`` fetches into a throwaway worktree.
A review tool that moved your HEAD would be one you stop running.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Final
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx

from app.domain.contracts import Severity
from app.domain.indexing.embedding import ChunkEmbedder
from app.domain.retrieval.retriever import RetrievalConfig
from app.domain.review.reviewer import ReviewerConfig
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.llm.ollama import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaLLM
from app.infra.parsing.registry import default_parsers
from app.infra.parsing.syntax import TreeSitterSyntaxChecker
from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore
from app.infra.review.head_files import WorkingTreeHeadFiles
from app.infra.source.working_tree import WorkingTreeSource
from app.infra.store.memory import InMemoryEmbeddingCache, InMemoryIndexStore
from argus_cli import repo
from argus_cli.render import Palette, render_findings, render_json, render_summary
from worker.pipeline.indexer import Indexer
from worker.pipeline.review import ReviewPipeline

__all__ = ["main"]

type Logger = Callable[[str], None]

EMBEDDING_DIMENSIONS: Final = 1536
"""Matches the production ``vector(1536)`` column, so a local run and a stored
one are the same vector space rather than two that merely both work."""

_FAIL_LEVELS: Final[dict[str, Severity | None]] = {
    "never": None,
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description=(
            "AI pull-request review that checks its own work. Runs a local "
            "model, costs nothing, and verifies every claim against your tree "
            "before showing it to you."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  argus review                      review your uncommitted work\n"
            "  argus review --base main          review this branch against main\n"
            "  argus review --pr 42              review pull request #42\n"
            "  argus review --repo <url> --pr 7  review a PR on someone else's repo\n"
            "  argus doctor                      check that the setup works\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser(
        "review", help="review a change", description="Review a change."
    )
    review.add_argument(
        "path",
        nargs="?",
        default=".",
        type=Path,
        help="repository to review (default: the current directory)",
    )
    review.add_argument(
        "--base",
        metavar="REF",
        help="review everything since this ref (default: the fork point from "
        "your default branch)",
    )
    review.add_argument(
        "--pr",
        type=int,
        metavar="N",
        help="review pull request N from the origin remote, in a temporary "
        "worktree; needs no API token for a public repository",
    )
    review.add_argument(
        "--repo",
        metavar="URL",
        help="clone and review a remote repository instead of a local path",
    )
    review.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to review with (default: {DEFAULT_MODEL})",
    )
    review.add_argument(
        "--ollama-url",
        default=DEFAULT_BASE_URL,
        help=f"Ollama server (default: {DEFAULT_BASE_URL})",
    )
    review.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="most comments to show (default: 10); the rest are suppressed, "
        "never deleted",
    )
    review.add_argument(
        "--context-tokens",
        type=int,
        default=6000,
        metavar="N",
        help="token budget for retrieved context per review unit (default: 6000)",
    )
    review.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help="parallel model calls (default: 1; raise it only if your Ollama "
        "server has the memory)",
    )
    review.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help="per-call model timeout (default: 300)",
    )
    review.add_argument(
        "--fail-on",
        choices=list(_FAIL_LEVELS),
        default="never",
        help="exit non-zero when a comment at or above this severity is posted "
        "(default: never) -- for CI",
    )
    review.add_argument(
        "--json", action="store_true", help="emit JSON instead of prose"
    )
    review.add_argument(
        "--no-color", action="store_true", help="disable colour"
    )

    doctor = sub.add_parser(
        "doctor",
        help="check that the setup works",
        description="Check git, Ollama and the model before you need them.",
    )
    doctor.add_argument("--model", default=DEFAULT_MODEL)
    doctor.add_argument("--ollama-url", default=DEFAULT_BASE_URL)

    return parser


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #


async def _review(args: argparse.Namespace) -> int:
    palette = Palette.for_stream(sys.stdout, force=False if args.no_color else None)
    started = time.perf_counter()
    # Progress goes to stderr so `--json` on stdout stays a clean document and
    # `argus review --json > out.json` does the obvious thing.
    log = _logger(palette, quiet=args.json)

    with ExitStack() as stack:
        if args.repo:
            log(f"cloning {args.repo}")
            source_root = stack.enter_context(repo.cloned(args.repo))
        else:
            source_root = repo.find_repository_root(Path(args.path).resolve())

        root = repo.find_repository_root(source_root)

        if args.pr is not None:
            log(f"fetching pull request #{args.pr}")
            pull = stack.enter_context(repo.checkout_pull_request(root, args.pr))
            tree, base, label = pull.root, pull.base, f"pull request #{args.pr}"
        else:
            base_ref = args.base or repo.default_base_ref(root)
            base = repo.merge_base(root, base_ref)
            tree = root
            # The diff runs to the working tree, so it covers committed branch
            # work *and* anything not committed yet. Saying so matters: a user
            # who thinks only their commits were read will misread a comment
            # about a line they have not pushed.
            branch = repo.current_branch(root)
            if base == repo.head_sha(root):
                # Sitting on the base branch itself, so the only thing between
                # it and the working tree is what has not been committed.
                label = f"uncommitted changes on {branch}"
            else:
                label = f"{branch} since {base_ref}"
                if repo.is_dirty(root):
                    label += ", uncommitted changes included"

        # `--pr` checks out a commit, so the worktree is clean and diffing to
        # the working tree is the same as diffing to its head. For a local
        # review it is the point: uncommitted edits are included.
        diff_text = repo.diff_against(tree, base)
        if not diff_text.strip():
            log("no changes to review")
            print("Nothing to review: the diff is empty.", file=sys.stderr)
            return 0

        files, added = repo.summarize_diff(diff_text)
        log(f"reviewing {label}: {files} file(s), {added} added line(s)")

        snapshot_index, snapshot_vectors, symbols = await _index(tree, log)

        pipeline = ReviewPipeline(
            index=snapshot_index,
            vectors=snapshot_vectors,
            embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
            files=WorkingTreeHeadFiles(tree),
            syntax=TreeSitterSyntaxChecker(),
            known_symbols=symbols,
            retrieval=RetrievalConfig(token_budget=args.context_tokens),
            reviewer=ReviewerConfig(concurrency=max(1, args.concurrency)),
            findings_limit=args.limit,
        )

        repository_id = str(uuid5(NAMESPACE_URL, str(root)))
        plan = await pipeline.prepare(
            repository_id=repository_id, diff_text=diff_text
        )
        if plan.is_empty:
            log("no reviewable unit in this diff")
            print(
                "Nothing to review: the change touches no file Argus can parse "
                "(Python and TypeScript only).",
                file=sys.stderr,
            )
            return 0

        log(
            f"{len(plan.requests)} review unit(s), "
            f"{plan.retrieved_items} context chunk(s) retrieved"
        )
        llm = OllamaLLM(
            model=args.model,
            base_url=args.ollama_url,
            timeout_seconds=args.timeout,
        )
        if not await _ollama_ready(args.ollama_url, args.model):
            return 3

        log(f"reviewing with {args.model} (this is the slow part)")
        outcome = await pipeline.execute(plan, llm=llm, run_id=uuid4())

        elapsed = time.perf_counter() - started
        if args.json:
            render_json(
                posted=outcome.posted,
                suppressed=outcome.suppressed,
                drops=outcome.drops_by_gate,
                completeness=outcome.completeness,
                model=args.model,
            )
        else:
            render_findings(outcome.posted, palette=palette)
            render_summary(
                posted=outcome.posted,
                suppressed=outcome.suppressed,
                drops=outcome.drops_by_gate,
                decode_rejects=len(outcome.decode_rejects),
                failures=outcome.failures,
                completeness=outcome.completeness,
                units=len(plan.requests),
                seconds=elapsed,
                model=args.model,
                palette=palette,
            )

        threshold = _FAIL_LEVELS[args.fail_on]
        if threshold is not None and any(
            f.severity.rank >= threshold.rank for f in outcome.posted
        ):
            return 1
    return 0


async def _index(
    tree: Path, log: Logger
) -> tuple[InMemorySymbolIndex, InMemoryVectorStore, tuple[str, ...]]:
    """Index the checkout into memory. No database, nothing left behind."""
    store = InMemoryIndexStore()
    indexer = Indexer(
        source=WorkingTreeSource(tree),
        parsers=default_parsers(),
        cache=store,
        store=store,
        embedder=ChunkEmbedder(
            embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
            cache=InMemoryEmbeddingCache(),
        ),
    )
    repository_id = uuid5(NAMESPACE_URL, str(tree))
    log("indexing the tree")
    result = await indexer.index(
        repository_id=repository_id,
        repo_url=str(tree),
        commit_sha=repo.head_sha(tree),
    )
    log(
        f"indexed {result.files_indexed} file(s), {result.symbols} symbol(s), "
        f"{result.edges} edge(s) in {result.duration_ms / 1000:.1f}s"
    )
    snapshot = store.snapshot(result.snapshot_id)
    return (
        InMemorySymbolIndex(snapshot),
        InMemoryVectorStore(snapshot),
        tuple(symbol.fqn for symbol in snapshot.symbols),
    )


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


async def _doctor(args: argparse.Namespace) -> int:
    """Check the three things that go wrong, before they cost anyone a wait."""
    palette = Palette.for_stream(sys.stdout)
    ok = True

    def report(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        mark = f"{palette.low}ok{palette.reset}" if passed else (
            f"{palette.high}no{palette.reset}"
        )
        print(f"  [{mark}] {name}: {detail}")

    print(f"{palette.bold}argus doctor{palette.reset}")

    try:
        version = repo.git(None, "--version").strip()
        report("git", True, version)
    except Exception as error:
        report("git", False, f"not usable ({error})")

    try:
        root = repo.find_repository_root(Path.cwd())
        langs = repo.tracked_languages(root)
        report(
            "repository",
            bool(langs),
            f"{root} — {', '.join(langs) if langs else 'no Python or TypeScript files'}",
        )
    except repo.RepoError as error:
        report("repository", False, str(error).splitlines()[0])

    tags = await _ollama_models(args.ollama_url)
    if tags is None:
        report(
            "ollama",
            False,
            f"no server at {args.ollama_url} — install from https://ollama.com "
            "and run `ollama serve`",
        )
    else:
        report("ollama", True, f"{args.ollama_url}, {len(tags)} model(s) pulled")
        pulled = any(tag == args.model or tag.startswith(f"{args.model}:")
                     for tag in tags)
        report(
            f"model {args.model}",
            pulled,
            "pulled" if pulled else f"not pulled — run `ollama pull {args.model}`",
        )

    print()
    print(
        f"{palette.bold}ready{palette.reset}" if ok
        else f"{palette.medium}not ready — fix the items above{palette.reset}"
    )
    return 0 if ok else 1


async def _ollama_models(base_url: str) -> list[str] | None:
    """Model tags Ollama has, or ``None`` when there is no server there."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return None
    return [str(model.get("name", "")) for model in payload.get("models", [])]


async def _ollama_ready(base_url: str, model: str) -> bool:
    """Fail before indexing rather than after, with the command that fixes it."""
    tags = await _ollama_models(base_url)
    if tags is None:
        print(
            f"Cannot reach Ollama at {base_url}.\n"
            f"  Install it from https://ollama.com, then: ollama serve",
            file=sys.stderr,
        )
        return False
    if not any(tag == model or tag.startswith(f"{model}:") for tag in tags):
        print(
            f"Ollama has no model named {model!r}.\n"
            f"  ollama pull {model}\n"
            f"  (pulled: {', '.join(tags) if tags else 'none'})",
            file=sys.stderr,
        )
        return False
    return True


# --------------------------------------------------------------------------- #


def _logger(palette: Palette, *, quiet: bool) -> Logger:
    def log(message: str) -> None:
        if not quiet:
            print(f"{palette.dim}·{palette.reset} {message}", file=sys.stderr)

    return log


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "review":
            return asyncio.run(_review(args))
        return asyncio.run(_doctor(args))
    except (repo.RepoError, repo.GitError) as error:
        print(f"argus: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nargus: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
