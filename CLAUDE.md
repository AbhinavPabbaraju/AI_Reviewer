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

**M0–M3 are complete, and M6 is done in part** — the eval harness and 30 labeled PRs,
following the roadmap's own sequencing (M3 → **M6 partial** → M4). Every exit criterion is
measured and green: retrieval p95 against real pgvector (16 ms of an 800 ms budget), zero
fabrications escaping the verification gate over 20 seeded PRs, and a full 30-case eval run
in **0.9 s of a 600 s budget at $0.00**. M4, M5, M7 and M8 are untouched, and so are M6's
remaining two pieces — the isotonic calibration map and the CI regression gate.

**The harness's first run found two real weaknesses in Argus, which is what it is for.**
One is fixed (`SYMBOL_RESOLVES` was demoting correct findings for naming a parameter — see
`domain/review/vocabulary.py`); one is pinned as a `TestKnownLimitations` test with the
rejected fix written up, because the only honest fix is semantic. Read the M6 section before
touching the verification gate.

## Argus runs for free

**No AI credits are required to operate any part of this system, and that is a
constraint, not a coincidence.** Every stage has a free implementation:

| Stage | Free path |
|---|---|
| Embedding | `DeterministicEmbedder` — offline hashing, no network |
| Retrieval | structure-first over local Postgres; 98% of items arrive via the graph |
| LLM review | `OllamaLLM` — a local model on your own hardware, or `RecordedLLM` replay |
| Verification | pure computation, no model involved |

`tests/eval/test_review_pipeline.py` runs M1→M3 end to end and **asserts
`cost_usd == 0.0`** across the whole run. Keep it that way: an adapter that reports a
nonzero price corrupts the one number the M6 harness compares configurations on.

`OllamaLLM.cost_usd` is always `0.0` — that is a fact about local inference, not a
placeholder. Running free trades **recall**, not precision: a small local model finds
fewer defects, but its fabrications hit the same verification gate as any other model's.
Precision is structural; recall is what you give up.

**A paid adapter is optional and not built.** If one is added, note that `temperature` was
*removed* on Anthropic's current models (Opus 5 / 4.8 / 4.7) and sending it returns a 400 —
so `LLMPort.complete`'s `temperature` argument must be dropped by that adapter, not passed
through. It is kept on the port because local providers do honour it.

```
db/migrations/0001_init.sql          forward-only DDL (Postgres + pgvector)
db/migrations/0002_snapshot_scoping.sql  snapshot-scoped reads; chunks outlive snapshots
services/api/app/domain/             pure: contracts, ports, and the algorithms
  contracts.py  ports.py             M0 output contract + pipeline ports
  base.py                            shared Frozen value-object base
  indexing/  models.py ports.py filtering.py chunking.py incremental.py
             resolution.py embedding.py
  retrieval/ models.py ports.py expansion.py fusion.py retriever.py
  review/    diff.py grouping.py verification.py ports.py vocabulary.py
             prompts.py decoding.py reviewer.py budget.py
services/api/app/infra/              adapters implementing the ports
  source/git_source.py               blobless bare clone → SourceProviderPort
  parsing/{python,typescript}.py     tree-sitter extractors → ParserPort
  parsing/registry.py  base.py       language→parser map + tree-sitter plumbing
  embedding/deterministic.py         offline hashing embedder → EmbeddingPort
  store/{memory,postgres}.py         ParseCache + IndexStore + EmbeddingCache
  retrieval/{memory,postgres}.py     SymbolIndex + VectorStore
  db/{pool,migrate}.py               asyncpg pool + vector codec, migration runner
  parsing/syntax.py                  tree-sitter SyntaxChecker → PATCH_PARSES
  github/fake.py                     in-memory GitHubPort (M6 harness + gate tests)
  review/head_files.py               HeadFilePort bound to one (repo, sha)
  llm/ollama.py  llm/recorded.py     free LLMPorts: local model, and replay
services/api/app/api/                FastAPI: main.py deps.py schemas.py routers/
services/api/app/config.py           env-driven Settings (ARGUS_ prefix)
services/worker/worker/pipeline/indexer.py   the Stage I/II orchestrator
tests/{domain,infra,worker,api,eval}/  unit tests + the measured milestone gates
tests/eval/corpus/labeled_prs.py     M6 ground truth: 30 labeled PRs (20 seeded, 10 clean)
tests/eval/harness/                  transcript.py (input) runner.py (pipeline)
                                     metrics.py (scorer; imports no transcript)
```

**What M2 still lacks:** a real network `EmbeddingPort` — only `DeterministicEmbedder`
exists, so `POST /search` refuses any snapshot embedded by a different model rather than
comparing vectors across two embedding spaces. That guard is the thing to keep when a
network adapter lands.

**Known thin margin.** The M1 incremental budget (single-file push < 10 s) is the one to
watch now that persistence is real. Observed **5.8–9.5 s** across runs on the same machine
(in-memory: 3.8–6.2 s) — it is sensitive to machine load and page cache, and the 9.5 s end
of that range is uncomfortably close to the budget. Persistence costs ~2.5 s per push, and
the profile is symbols 0.91 s + edges 0.78 s + chunk membership 0.63 s. Symbols
and edges are rewritten in full on every snapshot because they are snapshot-scoped rows.
The fix, if the margin gets uncomfortable, is to derive membership from the snapshot's
file list rather than storing it: a symbol and a chunk both belong to a *blob*, so
`(path, blob_sha)` identifies them across snapshots and only genuinely global rows (edges)
need rewriting. That is a migration 0003, not a tweak — measure before starting it.

`app` (under `services/api`) and `worker` (under `services/worker`) are two top-level
packages installed as one editable distribution (ADR-005); the worker imports `app.domain`.

## Commands

Everything runs in a virtualenv at `.venv` (Python 3.14). One-time: `.venv/bin/pip install
-e ".[dev]"`.

- **Tests:** `.venv/bin/python -m pytest`  ·  single test: `… -m pytest tests/eval/test_resolution_rate.py::TestPythonResolutionRate -q`
- **Postgres tests:** set `ARGUS_TEST_DATABASE_URL`; they skip cleanly without it, so the
  default run needs no database. `tests/pg.py` applies migrations once and truncates
  between tests.
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
  chunked separately) to avoid double-embedding. That hash is also the embedding cache key,
  which is what makes a one-line push re-embed one chunk instead of the repository.

## Stage III (M2) — retrieval

- **Structure first, embeddings as supplement (ADR-002), and the code says so.** Fusion
  weights graph proximity above cosine, and `test_retrieval_quality.py` measures the
  provenance split (currently 98% of retrieved items arrive through the graph). If that
  ratio ever flips, the retriever has quietly become a vector search with extra steps.
- **M1's confidence tags do real work here.** Expansion proximity is
  `edge_confidence / (distance + 1)`, so a heuristic edge pulls its neighbour in weakly and
  a confident two-hop neighbour can outrank a doubtful one-hop one. This is the payoff for
  never faking confidence in the resolver.
- **Retrieval ports are set-at-a-time**, one query per BFS level. A neighbours-of-one-symbol
  port would read better and be an N+1 query against Postgres.
- **The budget drops whole symbols, never truncates one**, and anchors are never dropped —
  a pack that dropped the changed code to fit a caller in would be reviewing the wrong
  thing. An over-budget pack reports the overrun rather than hiding it.
- **Every item carries a `reason`** ("calls app.store.save"). M7's retrieval inspector
  renders these; keep them populated when adding a retrieval path.
- **Reads are scoped by repository *and* snapshot.** Repository scoping is tenancy;
  snapshot scoping is correctness, because every index run re-creates the repository's
  symbol rows and the union of all commits is not a codebase. `PostgresSymbolIndex`
  binds to one snapshot at construction (`for_latest`), mirroring the in-memory adapter,
  which is built from exactly one `SnapshotWrite`.
- **The two adapter sets must stay interchangeable.** The eval harness runs on the fakes,
  so any behaviour Postgres has that `InMemorySymbolIndex` lacks is a behaviour the gates
  cannot see. `tests/infra/test_postgres_*.py` compare them method by method, and
  `test_retrieval_pgvector.py` asserts both build byte-identical context packs. One known
  and permanent difference: pgvector stores **float4**, so chunks that tie exactly in the
  fake's float64 cosine get distinct distances and break ties differently. Assert on
  members and scores-within-epsilon, never on tie order.

## Postgres notes

- **`chunks` are repository-scoped and outlive snapshots**; membership lives in
  `snapshot_chunks`. This is why a chunk upsert never overwrites a stored vector with
  NULL (`COALESCE(EXCLUDED.embedding, chunks.embedding)`) — an unembedded re-index must
  not silently un-do the expensive half of indexing.
- **A chunk that is already stored is not rewritten.** Chunks are immutable and
  content-addressed, so `save()` writes only genuinely new chunks (plus any stored
  without a vector that this run can supply one for) and gives everything else a
  membership row. Rewriting all 29,000 on a one-file push is what put the incremental
  gate over budget at 10.5 s before this landed.
- **Bulk writes go through `COPY` into `ON COMMIT DROP` temp tables**, then one
  `INSERT ... SELECT` per entity, so the path→file-id and fqn→symbol-id joins happen in
  the server. Row counts are checked against the input (`_require_all_written`): a join
  that silently drops rows is a smaller graph, which surfaces much later as a missing
  review comment.
- **Local Postgres for the gates:** `ARGUS_TEST_DATABASE_URL=postgresql://user@host/db`.
  Unset, every Postgres test skips and the suite stays offline. Verified against
  Postgres 18.4 + pgvector 0.8.6.

## Stage V/VI (M3) — review and the verification gate

- **The gate is a mechanism, not a prompt instruction.** Everything the model emits is a
  *claim about a repository*, re-checked against that repository. `verification.py`
  implements sec. 4.6's seven gates one method each.
- **`SYMBOL_RESOLVES` asks "could this model have known this name", not "does this name
  exist".** Build its vocabulary with `vocabulary.build_vocabulary(symbols=…, packs=…)`:
  the symbol table *and* the identifiers in the context pack the model was given. Passing
  the symbol table alone is the narrow version M6 measured demoting correct findings, since
  parameters, locals, fields and module constants are real code that is not in that table.
- **Hard gates reject; soft gates demote.** A finding citing a missing file is
  unsalvageable. A finding whose *patch* fails still has prose worth posting, so the patch
  is stripped and the prose survives. Conflating the two either posts fabrications or
  discards real bugs over formatting.
- **Demotion penalties are per gate, and that is load-bearing.** A flat 0.3 penalty made
  "strip the patch, keep the prose" a fiction: one demotion pushed a typical finding under
  its severity floor, so the prose was suppressed a step later. Penalties now match what
  each failure says about *truth* — `SYMBOL_RESOLVES` 0.30 (the model invented part of its
  reasoning), `PATCH_PARSES` 0.10 (the patch is wrong, the prose is not), `PATCH_APPLIES`
  0.05 (says nothing about correctness, only about where GitHub can render it).
- **Gate order matters.** Hard rejections short-circuit, so one root cause produces one
  failure rather than four and the per-gate metrics stay meaningful. Dedup runs after
  per-finding checks; `CONFIDENCE_FLOOR` runs last, because demotions lower confidence and
  a twice-demoted finding must be measured after both.
- **Dedup compares word sets, not characters.** Character similarity scores "Missing null
  check on user lookup" / "Missing bounds check on index lookup" at 0.78 — higher than
  genuine duplicates — because the differing words share letters. Jaccard over title tokens
  separates them (0.50 vs 0.83). There is a regression test for exactly this.
- **Changed lines are new-file numbers and deletions contribute none.** A deleted line does
  not exist at head, so nothing can anchor to it. A finding in a file the diff never touched
  is allowed *only* if it cites evidence in a changed file — "your new caller breaks this"
  is a review comment; "this unrelated file has a bug" is noise the author cannot action.
- **The review unit is a symbol, not a hunk.** Three hunks in one function are one change;
  reviewing them separately asks the model the same question three times with less context
  each time. Grouping uses the parsed symbol table, not git's `@@` heading guess.

## Stage V (M3) — the review engine

- **The model emits a draft, never a `Finding`.** `DraftFinding` carries the claim;
  identity (`id`, `run_id`), attribution (`source`, `prompt_version`, `model`) and
  verification status are assigned by `decoding.py`. A model that could write
  `"verification": "verified"` would be grading its own work, and one that could set
  `"source": "semgrep"` could launder an unverifiable claim as a deterministic one.
- **Repository content is fenced, and the fence is neutralized.** A PR can contain
  `</untrusted-diff>` followed by instructions. `prompts.py` strips any closing tag from
  the content, so a diff cannot close its own fence and escape into the instruction
  channel. Tested with case and whitespace variants — this is a security boundary, not
  formatting.
- **Decoding is total.** Model output is the least trustworthy input in the system, so
  `decode_findings` never raises: malformed JSON, markdown fences, surrounding prose,
  contract violations and out-of-scope paths all become *counted rejects*.
  `reject_rate` is the companion metric to the gate's drop rate.
- **A provider failure loses one group, not the review.** `Reviewer` catches broadly on
  purpose (adapters raise provider-specific exceptions the domain must not import) and
  reports `completeness` — a review that silently covered half the diff is worse than one
  that says so.
- **The budget suppresses, never deletes.** Findings over the per-PR limit are marked
  rejected with `CONFIDENCE_FLOOR` and retained. The ordering is *total* — priority, then
  severity, then confidence, then fingerprint, then id — because a tie broken by dict
  ordering would make the same run post different comments on different days.

## M6 (partial) — the eval harness

`tests/eval/harness/` runs whole pull requests through M1→M3 against a fake `GitHubPort`
and a recorded `LLMPort`, and scores what was posted against hand-labeled ground truth in
`tests/eval/corpus/labeled_prs.py` (30 PRs: 20 seeded defects, 10 clean).

- **Three modules, split where it matters.** `transcript.py` is what the reviewer *says*
  (the input), `runner.py` is the pipeline, `metrics.py` is the scorer. The scorer does not
  import the transcript and must not: a scorer that could see what the reviewer intended
  would be grading intent instead of output.
- **Ten of the thirty PRs contain no defect.** Precision is the SLO, and a corpus of pure
  defects cannot measure it — every comment would be arguably on target. The clean cases are
  the changes a reviewer is *tempted* to comment on (a named local, a loop rewritten), so a
  trigger-happy reviewer pays for it there and only there.
- **The transcript is fixed, so any change in the numbers is a change in the pipeline.**
  That is the same reason `RecordedLLM` exists, and it is what makes these metrics a
  regression gate. The absolute values are a property of the transcript, so the enforced
  floors sit deliberately below the published SLOs; the *delta* between stages is the number
  that describes the code.
- **Every metric is computed twice** — over what the model emitted and over what was posted.
  Verification lifts precision **0.567 → 0.842** at **zero** recall cost. That gain is not a
  property of the transcript, since both stages read the same one.
- **Two precisions are reported, and the gap is the point.** `precision` is the SLO's own
  metric — posted findings a human agrees with — so a duplicate of a true finding counts as
  agreed with, because the author reads it and agrees. `precision_strict` charges it as
  noise. Reporting only the strict number would hold the product to a bar stricter than its
  own definition; reporting only the lenient one would hide what imperfect dedup costs.
- **Matching is location overlap plus a label's `signals`.** Location alone would score a
  confident wrong comment on the right line as a hit. The signal terms live in the label, in
  the open, so a disputed case is arguable rather than buried in the scorer.
- **`FILE_EXISTS` is unreachable from Stage V** and the harness asserts so: `decode_findings`
  is handed the group's path as `allowed_paths`, so a cross-file claim is dropped at decode.
  Only the M3 gate, which builds findings directly, can exercise it.

**What the first run found.** Two real weaknesses, handled differently on purpose:

1. **Fixed — `SYMBOL_RESOLVES` was punishing correct reviews.** It checked backticked tokens
   against the symbol table alone, which holds modules, classes, functions and methods — not
   parameters, locals, fields or module constants. Writing the way reviewers write ("slicing
   to `limit`…") cost 0.30 confidence for quoting real code; four demotions in the corpus were
   of that kind, and one left a true finding sitting exactly on its 0.60 floor.
   `domain/review/vocabulary.py` now also harvests identifiers from the **context pack** —
   the half of `known_symbols`'s own docstring that was never built. The pack is by
   construction the code the model saw, so a name in it was read rather than invented, while
   a name in neither the table nor the pack is still unaccounted for. Demotions went 6 → 2,
   and the two survivors are exactly the fabricated helpers. Both ends are asserted: widening
   a vocabulary until the gate cannot fire would be the obvious way to get this wrong.
2. **Not fixed — a paraphrased duplicate escapes `NOT_DUPLICATE`.** Title word sets separate
   two different bugs with similar titles (why dedup is not character similarity) but do not
   catch one bug described twice in different words; `ts-04` scores 0.44 against a 0.70
   threshold. **A lexical fix was tried and rejected**: stopwords plus stemming separate the
   decisive pair 0.667 vs 0.429, but only with a stemmer special-cased to collapse *validated*
   and *validation* — without that one hand-tuned rule both pairs score 0.429 and no threshold
   exists. That is fitting a threshold to a single word pair, and the next paraphrase lands
   somewhere else. Telling "same bug, different words" from "different bug, similar words" is
   a semantic problem and wants a semantic tool. The cost is bounded and measured: one
   redundant comment, which is the whole 0.842 → 0.789 gap between the two precisions. **Do
   not "fix" this by lowering `duplicate_similarity`** — it breaks the documented
   counterexample, and there is a regression test for exactly that.

## The measured gates

Every milestone gate lives in `tests/eval/` and is *measured and printed*, not merely
asserted (run with `-s`):

- `test_resolution_rate.py` scores the resolver against the hand-labeled corpora in
  `tests/eval/corpus/` (304 Python + 225 TypeScript references; ≥ 0.85 required, currently
  0.990 / 0.996). This is the regression gate for any resolver or parser change.
- `test_index_performance.py` (`slow`) builds a real 5,050-file git repo: cold index
  ≈ 21 s of a 240 s budget, single-file push ≈ 4 s of a 10 s budget with exactly one file
  re-parsed and 29,000 chunks reused.
- `test_retrieval_quality.py` (M2) runs 40 hand-built queries over the same corpora:
  the file a reviewer would need appears **95%** of the time against a ≥ 90% criterion.
- `test_retrieval_pgvector.py` (M2) re-runs those queries against real Postgres: p95
  **16 ms** of an 800 ms budget, identical recall, and byte-identical context packs.
- `test_verification_gate.py` (M3) runs 20 seeded-defect PRs past an *adversarial* reviewer
  that fabricates on purpose — one failure mode per gate. **Zero escapes** (the criterion is
  a zero, not a low number), 70% drop rate, all seven gates exercised. Survivors are
  re-derived from the head tree independently, so agreeing with the verifier is not enough;
  and the legitimate finding seeded into each PR must survive in 20/20, because a gate that
  suppresses everything would score a perfect zero and ship a product that never comments.
- `test_eval_harness.py` (M6 partial) runs all 30 labeled PRs end to end: **precision 0.842**
  (SLO 0.80; strict 0.789), **recall 0.750** (SLO 0.55), 0.10 wrong comments per PR, 9/10
  clean PRs silent, **$0.00** and **0.9 s** of a 600 s budget, byte-identical across runs.
  Zero defects are lost to verification, and `recall_cost` is asserted to be exactly 0.

**Sensitivity is the point of M6, and it is asserted.** A deliberately more speculative
reviewer — one extra confident, unfalsifiable comment per PR — drops precision 0.842 → 0.327
while the per-gate drop counts stay **byte-identical**. Every quality signal Argus had before
M6 would have reported that regression as a clean run. That test is the argument for the
labeled corpus; do not weaken it.

**A gate that cannot fail measures nothing.** The retrieval gate runs at a deliberately
tight 300-token budget, because at a production-sized budget these small corpora fit
entirely in one pack and recall is trivially 100% (reported alongside, for scale). If you
grow the corpora, re-check that the budget still forces the ranking to choose.

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
