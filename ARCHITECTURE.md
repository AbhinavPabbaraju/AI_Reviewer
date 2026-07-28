# Argus — Architecture

> An AI pull-request reviewer optimized for **precision**, not coverage.

**Status:** Milestone 0 (design frozen, contracts implemented)
**Owner:** Abhinav Pabbaraju
**Last updated:** 2026-07-28

---

## 1. The problem this system actually solves

There are hundreds of "AI PR reviewer" projects. Almost all of them are the same
program: fetch the diff, paste it into a model, post whatever comes back. They
fail for one reason, and it is not model quality.

**A reviewer that posts 40 comments of which 8 are real gets muted within a
week.** Developers do not grade a reviewer on recall. They grade it on whether
the last three comments it left were worth reading. One confidently wrong
comment costs more trust than ten correct ones earn.

So the design objective is not "review everything in the nine-stage pipeline."
It is:

> Maximize the number of true findings posted, subject to a hard ceiling on the
> false-positive rate, within a fixed latency and cost budget.

Every architectural decision below falls out of that sentence.

### Service Level Objectives

| Objective | Target | Why |
|---|---|---|
| Comment precision (posted findings that a human agrees with) | ≥ 0.80 | Below ~0.7 the tool gets muted; this is the product-defining metric |
| Seeded-defect recall (on the eval corpus) | ≥ 0.55 | Secondary. Recall is worth nothing if precision collapses |
| p50 latency, 300-line diff | ≤ 90 s | Must land before the human reviewer opens the PR |
| p95 latency, 2000-line diff | ≤ 6 min | Long tail is acceptable; a stale review is not |
| Cost per review | ≤ $0.15 median | Forces context-pack discipline instead of "stuff the window" |
| Index freshness after push | ≤ 60 s | Incremental indexing, not full re-embed |

**Load model (design point, not aspiration):** 100 repositories, 500 PR events
per day, peak ~2 events/sec, largest repo ~50k files / 4M LOC. This is small.
It is deliberately small — the interesting engineering here is in *quality
control*, not in horizontal scale, and pretending otherwise would be
architecture theatre.

---

## 2. What I cut from the brief, and why

The brief describes roughly twenty person-years of work. Shipping 15% of it at
production quality is worth far more than 100% of it as scaffolding. Here is
the cut list, with reasoning, because "why we didn't build it" is the part of a
design doc that actually gets read.

| Cut | Reasoning |
|---|---|
| **Knowledge graph / "long-term memory"** | Undefined requirement. The symbol graph (§4.2) already provides cross-file reasoning with precise semantics. A second, fuzzier graph adds surface area and no measurable win. |
| **CQRS** | There is one write path and one read path over the same small dataset. CQRS here is cargo cult; an interviewer will ask what problem it solved and there is no answer. |
| **DDD tactical patterns (aggregates, value objects, domain events)** | Kept: bounded contexts and a clean dependency direction. Dropped: the ceremony. The domain is thin — the complexity lives in retrieval and verification, not in business rules. |
| **7 static analyzers → 3** | Semgrep (multi-language, security), Ruff (Python, subsumes Flake8/Bandit/parts of Pylint), ESLint + `tsc --noEmit` (TS). Adding Pylint and Flake8 on top of Ruff produces duplicate findings whose dedup cost exceeds their value. |
| **Multi-language day one → Python + TypeScript** | Every additional language multiplies the tree-sitter grammar work, the symbol-resolution rules, the analyzer integration, *and* the eval corpus. Two languages done properly beats eight done by regex. |
| **S3 storage** | Deferred to M5. Nothing needs durable blob storage until we persist review artifacts and diff snapshots. Postgres holds everything at current scale. |
| **Kubernetes** | Docker Compose is the development and single-node production target. K8s manifests land in M8 as a deployment option, not as the architecture. Designing for K8s from day one on a 3-service system inverts the cost/benefit. |
| **Rich frontend dashboard** | Demoted to M7 and kept thin. **The real UI is the GitHub Check Run and the inline comments.** A dashboard nobody opens is not a feature. |
| **Streaming responses** | Only where a human is waiting (the dashboard's live run view). The webhook path is async by nature — streaming into a queue worker buys nothing. |

### And the one thing I added

**An offline evaluation harness with a CI regression gate (M6).**

This is not in the brief and it is the most important component in the system.
Without it, every prompt change is a vibe. With it, you can say in an interview:
*"Changing the retrieval fusion weight from 0.6 to 0.75 moved precision from
0.74 to 0.81 and recall from 0.51 to 0.58 across 120 labeled PRs, and the CI
gate blocks any prompt change that regresses precision by more than 2 points."*

That sentence is the whole project. Everything else is plumbing that makes it
possible to say.

---

## 3. Component view

```
                    GitHub
                       │
        ┌──────────────┴───────────────┐
        │ webhook (pull_request, push) │  ── HMAC-SHA256 verified
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  api  (FastAPI)              │   Thin. Verifies, persists an
        │  - webhook ingest            │   idempotent ReviewRun row,
        │  - run status / findings     │   enqueues, returns 202 in <200ms.
        │  - semantic search           │   Never calls an LLM inline.
        └──────────────┬───────────────┘
                       │ Redis (broker)
                       ▼
        ┌──────────────────────────────┐
        │  worker  (Celery)            │
        │                              │
        │   indexer ──► retriever ──►  │
        │   analyzers ──► reviewer ──► │
        │   verifier ──► publisher     │
        └───┬──────────────────────┬───┘
            │                      │
            ▼                      ▼
     ┌─────────────┐        ┌─────────────┐
     │  Postgres   │        │  Redis      │
     │  + pgvector │        │  cache +    │
     │             │        │  rate limit │
     └─────────────┘        └─────────────┘
```

Three deployable units: `api`, `worker`, `web`. Shared domain contracts live in
a single Python package imported by both `api` and `worker`; the TypeScript
types for `web` are **generated** from the same Pydantic models via the OpenAPI
schema, so the contract cannot drift.

**Dependency rule:** `domain` imports nothing from `infra`. Adapters (GitHub
client, Postgres repositories, LLM providers, analyzers) implement Protocols
declared in `domain/ports.py`. This is the only piece of Clean Architecture
worth the ceremony here, and it exists for one concrete reason: the eval harness
(M6) runs the entire review pipeline with a fake GitHub adapter and a recorded
LLM adapter. Without the port boundary, the eval harness is impossible.

---

## 4. The pipeline

Six stages, not nine. Stages 1–2 of the brief collapse into indexing; 7–9
(security/performance/testing) are **finding categories**, not pipeline stages —
running three separate LLM passes over the same context triples cost for
marginal recall.

### 4.1 Stage I — Incremental indexing

Triggered on `push` to the default branch, and on first install.

```
clone (bare, --filter=blob:none)  →  walk tree  →  filter
  →  parse (tree-sitter)  →  extract symbols + edges
  →  chunk by symbol boundary  →  embed changed chunks only
```

**Filtering** happens before parsing, and aggressively: `.gitignore` semantics,
vendored-path heuristics (`node_modules`, `vendor`, `dist`, `.venv`, generated
protobuf/OpenAPI markers), binary sniff, files > 1 MB, minified detection
(mean line length > 200).

**Chunking is by symbol, never by token window.** A chunk is a function, method,
class body, or top-level block, with its docstring and decorators attached. A
chunk that splits a function in half is worse than useless — it produces
embeddings for meaningless fragments and context packs that show the reviewer
half an implementation.

**Incrementality** is content-addressed: `chunk_id = sha256(repo_id | path |
symbol_fqn | normalized_body)`. On re-index, only chunks whose hash changed are
re-embedded. A one-line change to a 50k-file repo re-embeds ~1 chunk, not 50k.

### 4.2 Stage II — Symbol graph

This is where Argus diverges from typical RAG-over-code.

Tree-sitter gives an AST per file. On top of it we resolve, per language, a
graph of:

| Edge | Meaning |
|---|---|
| `DEFINES` | file → symbol |
| `CALLS` | symbol → symbol |
| `IMPORTS` | file → module/symbol |
| `INHERITS` | class → base class |
| `TESTS` | test symbol → symbol under test (name heuristic + import evidence) |
| `REFERENCES` | symbol → symbol (non-call: type annotation, decorator, attribute) |

Resolution is best-effort and **explicitly confidence-tagged**. Python dynamic
dispatch and TypeScript structural typing both defeat exact resolution;
pretending otherwise would be the first hallucination in the pipeline. Unresolved
references are stored with `confidence < 1.0` and are down-weighted during
expansion rather than dropped.

### 4.3 Stage III — Retrieval (hybrid, structure-first)

For each diff hunk, build a **context pack**:

1. **Anchor**: the changed symbol(s), full body, at head SHA.
2. **Graph expansion** (BFS, depth ≤ 2, budgeted): direct callers, direct
   callees, type/base-class definitions, and any `TESTS` neighbours.
3. **Semantic supplement**: pgvector ANN search over chunk embeddings for
   material the graph cannot reach — similar patterns elsewhere in the repo,
   relevant docs/README sections, config and migration files.
4. **Fusion rank**: `score = w_g · graph_proximity + w_s · cosine + w_r ·
   recency_of_co_change`. Weights are tuned against the eval corpus, not chosen
   by taste.
5. **Budget enforcement**: pack is truncated to a token ceiling by dropping the
   lowest-scoring items whole — never by truncating a symbol mid-body.

> Pure vector search over code is the industry default and it is mediocre,
> because "semantically similar to this function" and "necessary to review this
> function" are different questions. The caller that passes `None` is what makes
> the null-deref a bug, and it is often not textually similar to the callee at
> all. Structure first, embeddings as a supplement.

### 4.4 Stage IV — Deterministic analyzers

Semgrep, Ruff, and ESLint/`tsc` run **in parallel with** retrieval, sandboxed
(no network, read-only FS, memory + wall-clock capped, non-root). Their output
is high-precision and free of hallucination by construction.

Analyzer findings are not posted directly. They feed the reviewer as evidence,
which lets the LLM do the thing it is genuinely good at and analyzers are bad at:
**suppression with context.** `S105 hardcoded password` in a test fixture is
noise; the same rule in `auth/session.py` is a P0. Deciding which is which
requires reading the surrounding code, and that is an LLM's job.

### 4.5 Stage V — LLM review

One structured call per hunk-group, fanned out with bounded concurrency.
Constrained JSON decoding against the `Finding` schema (§5). Temperature 0.
Prompt version is recorded on every run so results are attributable.

Division of labour, stated explicitly because getting it wrong is the most
common failure mode in this product category:

- **Analyzers find**: known-pattern security bugs, style, type errors, unused
  code, dependency CVEs.
- **The LLM finds**: logic errors, violated API contracts, unhandled edge cases,
  concurrency hazards, missing test coverage for a new branch, misleading names.
- **The LLM never**: acts as a linter. If Ruff can find it, the LLM is not asked
  to.

### 4.6 Stage VI — Verification gate  ⟵ *the load-bearing component*

"Never hallucinate" is not a prompt instruction. It is a mechanism. Every
finding passes these checks before it is eligible to be posted:

| Gate | Check | On failure |
|---|---|---|
| `FILE_EXISTS` | Cited path exists at head SHA | Drop |
| `LINE_IN_RANGE` | Cited span is within the file and, unless flagged cross-file, within changed lines | Drop |
| `SYMBOL_RESOLVES` | Every symbol named in the explanation exists in the symbol table or context pack | Demote confidence |
| `PATCH_PARSES` | `improved_code` parses under the file's grammar | Strip the patch, keep the prose |
| `PATCH_APPLIES` | Suggestion applies cleanly as a GitHub suggested-change block | Downgrade to a plain comment |
| `NOT_DUPLICATE` | Not a near-duplicate of an analyzer finding or another LLM finding (span overlap + normalized-title similarity) | Merge |
| `CONFIDENCE_FLOOR` | Calibrated confidence ≥ per-severity threshold | Suppress, retain in DB |

Every drop is counted and exported as a metric. `verification_drop_rate` is a
direct, unfakeable measure of model reliability, and it is how you detect a bad
prompt change *before* users do.

Finally, a **findings budget**: at most N comments per PR (default 10), selected
by `severity × confidence`. Everything else is summarized in the Check Run
output. Suppressed findings stay in the database and appear in the dashboard —
they are hidden from the PR, not thrown away.

---

## 5. The output contract

Implemented in `services/api/app/domain/contracts.py` (Milestone 0, done).

```jsonc
{
  "id": "f_01J...",
  "severity": "high",              // critical|high|medium|low|info
  "category": "security",          // correctness|security|performance|concurrency|
                                   // maintainability|testing|api_contract|style
  "title": "User-controlled path passed to open() without normalization",
  "explanation": "…why this is a bug, in terms of this codebase…",
  "evidence": [                    // REQUIRED. ≥1. Every claim is anchored.
    {"path": "app/files.py", "line_start": 42, "line_end": 47,
     "excerpt": "…", "role": "defect_site"},
    {"path": "app/routes.py", "line_start": 88, "line_end": 90,
     "excerpt": "…", "role": "caller"}
  ],
  "location": {"path": "app/files.py", "line_start": 42, "line_end": 47},
  "suggested_fix": "…prose…",
  "improved_code": "…patch text…",   // nullable
  "confidence": 0.86,                // calibrated, not self-reported
  "source": "llm",                   // llm|semgrep|ruff|eslint|tsc
  "references": ["https://cwe.mitre.org/data/definitions/22.html"],
  "verification": {"status": "verified", "gates_failed": []},
  "prompt_version": "reviewer/v3",
  "model": "…"
}
```

Two fields do the heavy lifting:

- **`evidence` is required and must be non-empty.** A finding that cannot point
  at code is not a finding. This single constraint eliminates the majority of
  plausible-sounding fabrications, because the model must produce a citable span
  that the verifier will independently check.
- **`confidence` is calibrated, not self-reported.** Raw model confidence is
  near-useless (everything clusters at 0.9). The calibration map is fit on the
  eval corpus — isotonic regression from raw score → observed precision, per
  category. Until M6 produces that map, the field carries the raw score and is
  flagged `calibrated: false`.

---

## 6. Data model (sketch — full DDL in `db/migrations/0001_init.sql`)

```
installations ──< repositories ──< index_snapshots
                       │
                       ├──< files ──< chunks (embedding vector(1536))
                       │        └──< symbols ──< symbol_edges
                       │
                       └──< pull_requests ──< review_runs ──< findings
                                                    │
                                                    └──< run_events
eval_cases ──< eval_results
```

Notable choices:

- **`review_runs` carries an idempotency key** `(repo_id, pr_number, head_sha,
  ruleset_version, prompt_version)`, unique. GitHub redelivers webhooks; a
  redelivery must be a no-op, not a second $0.15 review.
- **`findings` are immutable and append-only per run.** Re-review creates a new
  run. This makes the eval harness and A/B comparison of prompt versions trivial.
- **`chunks.embedding` uses HNSW** (`vector_cosine_ops`, `m=16`,
  `ef_construction=64`). IVFFlat needs retraining as the corpus grows and
  degrades badly on incremental inserts, which is exactly our write pattern.

---

## 7. Architecture Decision Records

### ADR-001: Precision-first review policy

**Status:** Accepted · **Date:** 2026-07-28

**Context.** LLM review output is high-recall and low-precision by default. The
product fails at low precision regardless of recall.

**Decision.** Enforce a mandatory verification gate (§4.6) plus a per-PR
findings budget, and treat `comment_precision` as the primary SLO with recall
secondary.

**Options considered.**

| Option | Complexity | Precision | Recall | Notes |
|---|---|---|---|---|
| A. Post everything the model emits | Low | ~0.4 | High | Industry default. Gets muted. |
| B. Raise the confidence threshold only | Low | ~0.6 | Medium | Model confidence is uncalibrated; threshold is arbitrary |
| C. Verification gate + budget (chosen) | Medium | ~0.8 target | Medium | Adds a subsystem, and it is the subsystem worth building |

**Consequences.** Easier: trusting the output; measuring quality; debugging bad
prompt changes. Harder: recall on genuinely subtle cross-file bugs, since strict
evidence requirements suppress some true positives. Revisit if
`verification_drop_rate` exceeds 0.4, which would indicate the gate is
mis-calibrated rather than the model being wrong.

---

### ADR-002: Hybrid structural + semantic retrieval

**Status:** Accepted · **Date:** 2026-07-28

**Context.** "Never review files in isolation" requires deciding *which* other
files. Cosine similarity is the cheap answer.

**Decision.** Symbol-graph expansion is primary; vector search supplements.

**Options considered.**

| Option | Complexity | Recall of *relevant* context | Failure mode |
|---|---|---|---|
| A. Vector search only | Low | Medium | Retrieves things that *look* like the diff rather than things that *constrain* it |
| B. Graph only | Medium | Medium | Misses docs, config, and cross-module conventions with no edges |
| C. Fusion (chosen) | High | High | Two subsystems to maintain; weights need tuning against the eval set |

**Trade-off.** C costs a language-specific resolver per supported language —
which is precisely why §2 caps the language list at two. The resolver is the
main cost driver for adding language #3, and that should be a deliberate,
budgeted decision rather than a config flag.

**Consequences.** Easier: cross-file reasoning; explaining retrieval decisions.
Harder: adding languages. Revisit when a third language is genuinely needed.

---

### ADR-003: pgvector over a dedicated vector database

**Status:** Accepted · **Date:** 2026-07-28

**Context.** Corpus ceiling ~5M chunks. Every vector query is filtered by
`repo_id` and joined to relational rows (`files`, `symbols`).

**Decision.** pgvector with HNSW, in the primary Postgres.

| Option | Ops burden | Filtered-query story | Transactional consistency |
|---|---|---|---|
| A. pgvector (chosen) | None (already running Postgres) | Native — it's a `WHERE` clause | Yes: index updates commit with metadata |
| B. Qdrant / Weaviate | New service, backups, upgrades | Good, via payload filters | No — dual-write, needs reconciliation |
| C. pinecone | Managed, but external + $ | Good | No |

**Trade-off.** Dedicated stores win above roughly 10–50M vectors. We are two
orders of magnitude below that. Choosing B now would be optimizing for a scale
we will not reach while paying dual-write consistency bugs immediately — the
classic premature-distribution mistake.

**Consequences.** Revisit if p95 ANN latency exceeds 150 ms or the corpus passes
10M chunks.

---

### ADR-004: Celery + Redis for the work queue

**Status:** Accepted · **Date:** 2026-07-28

**Context.** Reviews are minutes-long, fan out per hunk, must survive worker
restarts, and must not double-charge on webhook redelivery.

**Decision.** Celery with a Redis broker; **idempotency enforced in Postgres**,
not in the broker.

| Option | Maturity | Observability | Fit |
|---|---|---|---|
| A. Celery + Redis (chosen) | Very high | Best-in-class (Flower, OTel instrumentation) | Canvas primitives (`chord`, `group`) map directly onto fan-out/fan-in |
| B. arq | Medium | Thin | Lighter, async-native, but we'd rebuild retry/routing |
| C. Postgres-backed queue (SKIP LOCKED) | High | DIY | One fewer service; but we already need Redis for caching and rate limiting |

**Consequences.** Redis is not a durable broker — a Redis loss can drop enqueued
tasks. Mitigated by making the `ReviewRun` row the source of truth: a sweeper
re-enqueues runs stuck in `queued` beyond a threshold. Easier: fan-out,
retries, scheduled re-index. Harder: exactly-once semantics, which we do not
attempt — we get idempotency instead, which is the correct target.

---

### ADR-005: Monorepo, two languages, generated types

**Status:** Accepted · **Date:** 2026-07-28

**Decision.** Single repository. `services/api` and `services/worker` share
`app.domain` as an installed local package. `web/` consumes TypeScript types
generated from the FastAPI OpenAPI schema in CI.

**Rationale.** The contract in §5 is the spine of the system; three copies of it
(Python domain, API serializer, TS frontend) drifting apart is the most likely
source of production bugs in a system like this. One source, generated
downstream, checked in CI.

---

## 8. Security posture

| Surface | Control |
|---|---|
| Webhook | HMAC-SHA256 with constant-time compare; reject if `X-Hub-Signature-256` absent; 5-minute delivery-timestamp window |
| GitHub credentials | App installation tokens (1 h TTL), minted per-job, never persisted; private key from the secret store, never the repo |
| Cloned code | Treated as **untrusted input**, always. Bare clone, blobless, no hooks, no submodule recursion, no `git-lfs` fetch |
| Analyzer execution | Sandboxed container: non-root, read-only rootfs, no network, 512 MB / 60 s caps, seccomp default |
| Prompt injection | Repository content is fenced and role-tagged as untrusted data. Instructions in code comments are ignored by construction: the reviewer's output is constrained-decoded to the `Finding` schema, and every finding must pass the verification gate. **A prompt injection cannot make the system emit an unverifiable finding** — the worst it can do is suppress a true one, which fails safe. |
| API | JWT (GitHub OAuth exchange), RBAC scoped to installation, per-installation rate limits in Redis, request-body caps |
| Secrets in output | Findings are scrubbed for high-entropy strings before posting — an AI reviewer quoting a leaked key back into a public PR comment is a genuine incident class |

The prompt-injection row is worth restating: defence-in-depth here is
*structural*, not a system-prompt instruction telling the model to be careful.
Instruction-based defences fail; schema constraints and independent verification
do not.

---

## 9. Observability

- **OpenTelemetry** traces spanning webhook → task → each pipeline stage →
  provider call. One trace per `review_run`, carried across the queue boundary.
- **Metrics that matter** (beyond RED): `verification_drop_rate` by gate,
  `findings_per_run` by severity, `context_pack_tokens` p50/p95,
  `cost_usd_per_run`, `analyzer_duration` by tool, `index_lag_seconds`,
  `comment_resolution_rate` (did the human accept the suggestion?).
- **`comment_resolution_rate` is the only true online quality signal.** Everything
  else is a proxy. Tracking whether a suggested change was committed, dismissed,
  or ignored gives real labels for free — and those labels feed straight back
  into the M6 calibration corpus. That feedback loop is the system's long-term
  moat.
- **Structured logs**, JSON, with `run_id` / `trace_id` on every line. LLM
  request/response pairs persisted (content-addressed) behind a retention flag —
  mandatory for reproducing a bad review.

---

## 10. Repository layout

```
argus/
├── ARCHITECTURE.md            ← this document
├── ROADMAP.md                 ← milestones + exit criteria
├── db/migrations/             ← plain SQL, forward-only, applied by Alembic
├── services/
│   ├── api/
│   │   └── app/
│   │       ├── domain/        ← contracts, ports, pure logic. Imports no infra.
│   │       │   ├── contracts.py
│   │       │   ├── ports.py
│   │       │   └── verification.py
│   │       ├── infra/         ← adapters: github, postgres, redis, llm
│   │       ├── api/           ← routers, deps, auth
│   │       └── config.py
│   └── worker/
│       ├── pipeline/          ← indexer, retriever, analyzers, reviewer, publisher
│       └── tasks/
├── packages/
│   └── analyzers/             ← sandboxed analyzer runners + rulepacks
├── eval/
│   ├── corpus/                ← labeled PRs (seeded + historical)
│   ├── harness/
│   └── report/
├── web/                       ← Next.js. Types generated from OpenAPI.
├── deploy/
│   ├── compose/
│   └── k8s/
└── .github/workflows/
```

---

## 11. Known risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Eval corpus too small to detect real regressions | High | Target ≥ 120 labeled PRs before the CI gate is enforcing; until then it reports without blocking |
| Symbol resolution accuracy is the retrieval ceiling | High | Measure resolution rate per language as its own metric; do not let it hide inside end-to-end numbers |
| Cost per review drifts up as context packs grow | Medium | Hard token budget enforced in code, alarmed at p95, not left to prompt discipline |
| Verification gate suppresses true positives | Medium | Track suppressed findings and hand-label a sample each iteration |
| Scope creep back toward the original brief | **Certain** | This document. Re-read §2 before adding a milestone. |
