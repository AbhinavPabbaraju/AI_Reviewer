# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Argus** — an AI pull-request reviewer optimized for **precision, not coverage**. The
governing objective (from `files/ARCHITECTURE.md`) is:

> Maximize true findings posted, subject to a hard ceiling on the false-positive rate,
> within a fixed latency and cost budget.

The primary SLO is `comment_precision ≥ 0.80`; recall is explicitly secondary. Almost
every design decision falls out of that. When making a change, ask "does this protect
precision?" before "does this find more bugs?" — a change that raises recall at the cost
of precision is usually the wrong trade here.

## Current state — read this first

**M0 (contracts) and M1 (ingestion + symbol graph) are implemented.** M2–M8 are not yet
built. The repo layout follows `ARCHITECTURE.md` §10:

```
db/migrations/0001_init.sql          forward-only DDL (Postgres 16 + pgvector)
services/api/app/domain/             pure: contracts, ports, and the indexing algorithms
  contracts.py  ports.py             M0 output contract + pipeline ports
  base.py                            shared Frozen value-object base
  indexing/  models.py ports.py filtering.py chunking.py incremental.py resolution.py
services/api/app/infra/              adapters implementing the ports
  source/git_source.py               blobless bare clone → SourceProviderPort
  parsing/{python,typescript}.py     tree-sitter extractors → ParserPort
  parsing/registry.py  base.py       language→parser map + tree-sitter plumbing
  store/memory.py                    in-memory ParseCache + IndexStore (Postgres adapter TBD)
services/worker/worker/pipeline/indexer.py   the Stage I/II orchestrator
tests/{domain,infra,worker,eval}/    unit tests + the M1 resolution-rate gate
```

`app` (under `services/api`) and `worker` (under `services/worker`) are two top-level
packages installed as one editable distribution (ADR-005); the worker imports `app.domain`.

## Commands

Everything runs in a virtualenv at `.venv` (Python 3.14). One-time: `.venv/bin/pip install
-e ".[dev]"`.

- **Tests:** `.venv/bin/python -m pytest`  ·  single test: `… -m pytest tests/eval/test_resolution_rate.py::TestPythonResolutionRate -q`
- **Measured gates:** `… -m pytest tests/eval -s` prints the numbers (`-s` matters — the
  exit criteria are *reported*, not just asserted). The latency gate is marked `slow` and
  deselected by default: run it with `… -m pytest -m slow -s`.
- **Types (strict):** `.venv/bin/python -m mypy` — strict, over the `app` and `worker`
  packages. Must stay clean.
- **Lint:** `.venv/bin/ruff check services tests` (Ruff is the sole Python linter — it
  subsumes Flake8/Bandit; do not add Pylint/Flake8, see ARCHITECTURE §2).
- Async tests run under `pytest-asyncio` in `asyncio_mode = "auto"` — just write `async def
  test_…`; no decorator needed.
- The git-source and indexer tests build **real throwaway git repos** via the `make_git_repo`
  fixture in `tests/conftest.py` (no network); `parse_python` / `parse_typescript` there turn
  source text straight into a `ParsedFile`.

## Stage I/II (M1) architecture — how indexing fits together

The pipeline is layered strictly: **infra parses, domain reasons, the worker sequences.**

- **Parsers (infra) produce, resolver (domain) consumes.** A `ParserPort` turns bytes into a
  language-agnostic `ParsedFile` (symbols + imports + *unresolved* `Reference`s) and assigns
  fqns; it does **no** resolution. `Resolver` (pure, in `domain/indexing/resolution.py`) turns
  references into a confidence-tagged `SymbolEdge` graph. This split is why the resolver is
  unit-tested against hand-built `ParsedFile`s with no tree-sitter in the loop — mirror it
  when adding a language: extraction goes in a new infra parser, not in the resolver.
- **The resolver is language-agnostic by construction.** It keys symbols by `(module, name)`
  and `(parent, name)` and never rebuilds an fqn by string surgery, so one code path resolves
  both languages. The *only* language-specific step is import-module resolution
  (`_resolve_module` / `_from_module_path`: dotted packages vs relative paths). Re-export
  chains (a package `__init__`, a TS barrel `index.ts`) are followed through the same code
  path, which is why `export … from` is parsed *as an import*: a re-export is a binding
  brought in and re-exposed, so the resolver needs no TS-specific concept for it.
- **Type hints from the parser are validated, never trusted.** `Reference.receiver_type`
  carries "this local was assigned `Thing(...)`" and nothing more; the resolver uses it only
  if the name resolves to a class in this repo that declares the member, and tags the result
  `confidence.INFERRED_LOCAL` (0.9, not EXACT — the binding is last-write-wins). A parser
  that decided what was a class would be inventing type information.
- **Confidence is never faked.** An unresolved reference is kept with its textual target and a
  floor confidence (`confidence.UNRESOLVED`), never dropped — ARCHITECTURE §4.2. External
  (stdlib/vendor) references are `confidence.EXTERNAL` and excluded from the resolution-rate
  denominator; they are correct, not failures.
- **Incrementality is a content-addressed parse cache.** `ParseCachePort` is keyed by
  `blob_sha`, so a changed file is a cache miss and everything else is a hit; the cached
  `ParsedUnit` carries both parsed structure *and* chunks so neither is recomputed. The
  resolver still re-runs **globally** every index (cheap, pure) so a moved symbol never leaves
  a stale edge. Do not "optimize" by carrying resolved edges forward.
- **Chunking is by symbol, content-addressed.** `chunk_id = sha256(repo_id | path |
  symbol_fqn | normalized_body)`; a container's chunk excludes its methods' lines (they are
  chunked separately) to avoid double-embedding. Embeddings themselves are M2 — M1 produces
  chunks with a null embedding.

**The M1 exit gates** live in `tests/eval/`, and both are *measured and printed*, not merely
asserted:

- `test_resolution_rate.py` scores the resolver against the hand-labeled corpora in
  `tests/eval/corpus/` (304 Python + 225 TypeScript references; ≥ 0.85 required, currently
  0.990 / 0.996). This is the regression gate for any resolver or parser change.
- `test_index_performance.py` (`slow`) builds a real 5,050-file git repo: cold index
  ≈ 21 s of a 240 s budget, single-file push ≈ 4 s of a 10 s budget with exactly one file
  re-parsed and 29,000 chunks reused.

**Labels are ground truth, not a recording of resolver output** — that is what makes the
gate a gate. Write an expectation by reading the corpus source; if the resolver disagrees,
either it has a bug or the case is genuinely ambiguous, and the honest move is to leave the
correct label in place as a known miss (the four current ones are commented as such). Three
label kinds: an fqn, `EXTERNAL(name)` (leaves the repo), and `UNRESOLVED(name)` (**must not**
bind to any repo symbol — this is the one that penalizes fabricated edges).

## Architecture that spans files

**The dependency rule.** `domain/` imports nothing from `infra/`. Adapters (GitHub client,
Postgres repos, LLM providers, analyzers) implement the `Protocol`s in `ports.py`. This is
the *only* Clean-Architecture ceremony kept, and it exists for one concrete reason: the M6
eval harness runs the whole pipeline against a **fake `GitHubPort`** and a **recorded
`LLMPort`** so eval runs are deterministic and free. Do not let the domain reach for infra
— it breaks the harness.

**One contract, three consumers.** `contracts.py` is serialized by the API, produced by the
worker, scored by the eval harness, and is the source for the frontend's generated
TypeScript types. Keep it as the single source of truth; never fork a second copy of these
shapes. `0001_init.sql` deliberately mirrors it (native enums, CHECK constraints re-stating
the domain validators as belt-and-braces).

**The pipeline** (six stages, run in the worker): incremental indexing → symbol graph →
hybrid retrieval (structure-first, embeddings supplement) → deterministic analyzers
(parallel) → LLM review → **verification gate**. Analyzer findings and LLM findings both
flow into the same `Finding` contract and the same gate.

**The verification gate is the load-bearing component** (ARCHITECTURE §4.6). "Never
hallucinate" is a *mechanism*, not a prompt instruction. Every finding must pass file/line/
symbol/patch/dedup/confidence gates before it is postable. `Finding.demote()` and
`Finding.reject()` implement the soft/hard outcomes; both return **new** instances
(findings are immutable) so the audit trail stays honest.

## Invariants you must not break

These are enforced by validators in `contracts.py` and mirrored by tests and SQL
constraints. Changing them is a deliberate, reviewed act — not a casual edit:

- **Evidence is required and non-empty**, and must include a `DEFECT_SITE` span that
  *overlaps* `location`. This single constraint kills most fabrications. A finding that
  can't point at code is not a finding.
- **`confidence` is calibrated, not self-reported.** Until M6 fits the calibration map,
  `confidence_calibrated` is `False` and the value is an *ordinal ranking signal only* —
  never treat it as a probability.
- **Findings are immutable and append-only per run.** Re-review creates a new `ReviewRun`;
  nothing is updated in place (keeps A/B prompt comparison trivial).
- **`ReviewRun.idempotency_key`** = `(repository_id, pr_number, head_sha, ruleset_version,
  prompt_version)`, enforced by a UNIQUE constraint in Postgres, **not** in the queue.
  GitHub redelivers webhooks; a redelivery must be a no-op, not a second paid review.
- **Paths are repo-relative POSIX, no `..`, no absolute.** Cloned repos are untrusted
  input; `CodeSpan` rejects traversal at the type boundary.
- **`severity × confidence`** ranks the findings budget; severity and confidence are kept
  orthogonal so a low-confidence CRITICAL doesn't outrank a high-confidence HIGH.
- **Attribution:** LLM findings require `prompt_version`; analyzer (deterministic) findings
  require `rule_id`.

## Working norms from the roadmap

- A milestone is done only when its **exit criterion is measured and green in CI** — not
  when the code exists. Exit criteria are numeric on purpose ("resolution rate ≥ 85%",
  "zero findings escape the gate with a nonexistent file"). Measure, don't assert.
- Resist scope creep back toward the original brief. ARCHITECTURE §2 lists what was
  deliberately cut (knowledge graph, CQRS, K8s day-one, 7 analyzers → 3, 8 languages →
  Python+TypeScript) and why. Re-read it before adding a component.
- Scope is **Python + TypeScript only.** Adding a third language is a budgeted decision (it
  costs a whole language-specific symbol resolver), not a config flag.
