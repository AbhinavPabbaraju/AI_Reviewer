"""Git operations the CLI needs, and nothing more.

Deliberately subprocess-driven rather than a library binding: every command here
is one a user could have typed, which means an unexpected result can be
reproduced by hand in the repository it happened in. That property is worth more
in a tool people run on their own code than the speed of an in-process
implementation.

Nothing here mutates the user's checkout. Reviewing a pull request happens in a
throwaway worktree, because the one behaviour a code-review tool must never have
is changing the branch you were working on.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "GitError",
    "RepoError",
    "checkout_pull_request",
    "cloned",
    "current_branch",
    "default_base_ref",
    "diff_against",
    "find_repository_root",
    "head_sha",
    "is_dirty",
    "merge_base",
    "resolve_ref",
]

_CANDIDATE_BASES = ("origin/main", "origin/master", "main", "master", "develop")


class GitError(RuntimeError):
    """A git command failed. Carries git's own stderr, which is usually the
    clearest available explanation and is worth showing verbatim."""


class RepoError(RuntimeError):
    """The repository is not in a state this command can work with."""


def git(root: Path | None, *args: str, check: bool = True) -> str:
    location = ["-C", str(root)] if root is not None else []
    process = subprocess.run(
        ["git", *location, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and process.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed: {process.stderr.strip() or 'no output'}"
        )
    return process.stdout


def find_repository_root(start: Path) -> Path:
    """The work tree containing ``start``.

    A clear error here is worth the check: "not a git repository" from a review
    tool is a far better message than a traceback out of a diff parser three
    calls later.
    """
    if not start.exists():
        raise RepoError(f"{start} does not exist")
    try:
        top = git(start, "rev-parse", "--show-toplevel").strip()
    except GitError as error:
        raise RepoError(
            f"{start} is not inside a git repository. Argus reviews diffs, so it "
            f"needs one.\n  git init && git add -A && git commit -m 'initial'"
        ) from error
    return Path(top).resolve()


def resolve_ref(root: Path, ref: str) -> str:
    try:
        return git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    except GitError as error:
        raise RepoError(f"unknown ref {ref!r} in {root}") from error


def head_sha(root: Path) -> str:
    return resolve_ref(root, "HEAD")


def current_branch(root: Path) -> str:
    name = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    return "(detached HEAD)" if name == "HEAD" else name


def is_dirty(root: Path) -> bool:
    return bool(git(root, "status", "--porcelain").strip())


def default_base_ref(root: Path) -> str:
    """The branch this work is presumably headed for.

    ``origin/HEAD`` is the honest answer when the remote publishes one, since it
    is what the remote itself calls its default. Everything after it is a guess
    at a repository that has no remote, ordered by how common the name is.
    """
    symbolic = git(root, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD",
                   check=False).strip()
    if symbolic:
        return symbolic.removeprefix("refs/remotes/")

    for candidate in _CANDIDATE_BASES:
        if git(root, "rev-parse", "--verify", "--quiet", candidate,
               check=False).strip():
            return candidate

    raise RepoError(
        "could not work out a base branch: no origin/HEAD and no branch named "
        f"{' or '.join(_CANDIDATE_BASES[:3])}. Pass --base explicitly."
    )


def merge_base(root: Path, base: str, head: str = "HEAD") -> str:
    """Where ``head`` left ``base``.

    The fork point rather than the tip of ``base``, so that commits landing on
    main while you worked do not turn up in your review as though you wrote
    them. This is the ``...`` in ``git diff a...b``, spelled out because the
    working-tree diff below cannot use that syntax.
    """
    merged = git(root, "merge-base", base, head, check=False).strip()
    if not merged:
        raise RepoError(
            f"{base!r} and {head!r} have no common ancestor; is {base!r} the "
            "right base branch?"
        )
    return merged


def diff_against(root: Path, base: str, head: str | None = None) -> str:
    """Unified diff from ``base`` to ``head``, or to the working tree.

    ``head=None`` compares against the files on disk, which is what makes
    "review what I am about to commit" work. Renames are disabled and context is
    fixed at git's default: the diff is parsed for line numbers, and a rename
    detected as such carries no hunks for the lines that moved.
    """
    args = ["diff", "--no-color", "--no-renames", "--no-ext-diff", base]
    if head is not None:
        args.append(head)
    return git(root, *args)


@dataclass(frozen=True, slots=True)
class PullRequest:
    number: int
    ref: str
    root: Path
    base: str


@contextmanager
def checkout_pull_request(
    root: Path, number: int, *, remote: str = "origin"
) -> Iterator[PullRequest]:
    """Fetch a pull request and check it out somewhere harmless.

    GitHub publishes every pull request as ``refs/pull/N/head`` on the origin
    remote, which is readable with the same credentials as a clone and needs no
    API token at all. It is checked out into a temporary worktree so the branch
    you had open stays open -- a review tool that moved your HEAD would be
    unusable on the days you most want it.
    """
    local_ref = f"refs/argus/pr-{number}"
    try:
        git(root, "fetch", "--quiet", remote, f"refs/pull/{number}/head:{local_ref}")
    except GitError as error:
        raise RepoError(
            f"could not fetch pull request #{number} from {remote}. Check the "
            f"number, and that {remote} is a GitHub remote you can read.\n"
            f"  {error}"
        ) from error

    workdir = Path(tempfile.mkdtemp(prefix=f"argus-pr-{number}-"))
    worktree = workdir / "tree"
    try:
        git(root, "worktree", "add", "--quiet", "--detach", str(worktree), local_ref)
        yield PullRequest(
            number=number,
            ref=local_ref,
            root=worktree,
            base=_pull_request_base(root, local_ref),
        )
    finally:
        git(root, "worktree", "remove", "--force", str(worktree), check=False)
        git(root, "update-ref", "-d", local_ref, check=False)
        shutil.rmtree(workdir, ignore_errors=True)


def _pull_request_base(root: Path, ref: str) -> str:
    """The fork point of a fetched pull request against the default branch."""
    return merge_base(root, default_base_ref(root), ref)


@contextmanager
def cloned(url: str, *, depth: int = 0) -> Iterator[Path]:
    """A throwaway full checkout of a remote repository.

    Not blobless, unlike the indexer's own provider: this clone *is* the working
    tree that gets indexed and that the verification gate reads files back out
    of, so the content has to be present.
    """
    workdir = Path(tempfile.mkdtemp(prefix="argus-clone-"))
    target = workdir / "repo"
    args = ["clone", "--quiet"]
    if depth:
        args += ["--depth", str(depth)]
    args += ["--", url, str(target)]
    try:
        git(None, *args)
        yield target
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def summarize_diff(diff_text: str) -> tuple[int, int]:
    """``(files, changed lines)``, for the line the user reads before waiting."""
    files = sum(1 for line in diff_text.splitlines() if line.startswith("diff --git"))
    added = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return files, added


def tracked_languages(root: Path) -> Sequence[str]:
    """Extensions Argus can parse that this repository actually contains."""
    listing = git(root, "ls-files", check=False)
    suffixes = {Path(line).suffix for line in listing.splitlines() if line}
    return sorted(suffixes & {".py", ".ts", ".tsx"})
