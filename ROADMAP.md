# Argus — Implementation Roadmap

Nine milestones. Each has a **demoable outcome** and a **hard exit criterion**.
A milestone is not done because the code exists; it is done when the exit
criterion is measured and passing in CI.

Rule I'm enforcing throughout: **no milestone starts until the previous one's
tests are green and the code has been refactored once.** The refactor pass is
not optional — first-draft code that "works" is how a portfolio project starts
looking like a portfolio project.

---

## M0 — Contracts and skeleton  ✅ *done*

The output schema, the DB schema, and the port boundaries. Everything else
depends on these, and changing them later is expensive, so they are frozen
first.

- `Finding` / `Evidence` / `ReviewRun` contracts with validators
- Full DDL: installations → repositories → files → chunks/symbols → runs → findings
- `domain/ports.py`: `GitHubPort`, `EmbeddingPort`, `LLMPort`, `AnalyzerPort`
- Compose stack: Postgres 16 + pgvector, Redis 7
- CI: ruff, mypy --strict on `domain/`, pytest

**Exit:** contract tests pass; migration applies and rolls back cleanly; mypy
strict clean on `domain/`.

---

## M1 — Ingestion and the symbol graph

The hardest correctness work in the project. Do it first, while you have
patience for it.

- Blobless bare clone, `.gitignore`-aware walk, vendored/generated/minified filtering
- tree-sitter parsers for Python + TypeScript
- Symbol extraction: functions, methods, classes, exports, with spans and signatures
- Edge resolution: `CALLS`, `IMPORTS`, `INHERITS`, `REFERENCES`, `TESTS` — each confidence-tagged
- Symbol-boundary chunking with content-addressed IDs
- Incremental re-index driven by blob SHA diffing

**Exit:** index a 5k-file repo in < 4 min cold; a single-file push re-indexes in
< 10 s; **symbol resolution rate ≥ 85% on a hand-labeled sample of 200
references per language** — measured, not asserted.

---

## M2 — Embeddings and hybrid retrieval

- Batched embedding with a content-hash cache (never pay twice for one chunk)
- pgvector HNSW index; ANN queries always filtered by `repo_id`
- Graph expansion: BFS depth ≤ 2, budgeted, from diff hunks
- Fusion ranking + token-budget enforcement that drops whole symbols
- `POST /search` for semantic code search (also the manual debugging tool for retrieval)

**Exit:** on 30 hand-built queries, the file a human would need appears in the
context pack ≥ 90% of the time; p95 retrieval < 800 ms.

---

## M3 — Review engine and the verification gate

The centerpiece. Nothing gets posted to GitHub until this is trustworthy.

- Diff parsing → hunk grouping by enclosing symbol
- Prompt templates with fenced untrusted-content blocks and forced evidence
- Constrained JSON decoding against `Finding`; temperature 0; prompt versioning
- All seven verification gates, each independently tested
- Dedup/merge; findings budget; severity × confidence selection

**Exit:** on 20 seeded-defect PRs, **zero findings escape the gate with a
nonexistent file, an out-of-range line, or an unparseable patch.** That is a
zero, not a low number.

---

## M4 — Deterministic analyzers

- Sandboxed runners: non-root, read-only FS, no network, memory + wall-clock caps
- Semgrep (security rulepacks), Ruff, ESLint + `tsc --noEmit`
- Normalization into the same `Finding` contract with `source` set
- LLM-driven suppression of contextually-irrelevant analyzer hits (test fixtures, generated code)

**Exit:** analyzers cannot exceed their sandbox (proven by a hostile-repo test
case); merged output has < 5% duplicate findings.

---

## M5 — GitHub App

- App manifest, installation flow, per-job installation tokens
- HMAC-verified webhooks with replay protection
- Check Runs with summary, scores, and the suppressed-findings rollup
- Inline review comments + native suggested-change blocks
- Idempotent redelivery; secondary-rate-limit backoff

**Exit:** review a real PR on a real repo end to end; redelivering the same
webhook three times produces exactly one review and one set of comments.

---

## M6 — Evaluation harness  ⟵ *the milestone that makes this project worth building*

- Corpus: mutation-seeded defects + reversed historical bug-fix commits, hand-labeled
- Fake `GitHubPort` + recorded `LLMPort` so runs are deterministic and free
- Metrics: precision, recall, FP rate, verification drop rate by gate, cost, latency
- Confidence calibration (isotonic, per category) → replaces raw model scores
- CI gate: block any prompt/model/retrieval change regressing precision > 2 pts

**Exit:** ≥ 120 labeled cases; a full eval run in < 10 min; the gate demonstrably
catches a deliberately-worsened prompt.

---

## M7 — Dashboard (thin)

Next.js, TypeScript types generated from OpenAPI. Repo list, run timeline with
live status, finding detail with the evidence spans rendered, retrieval
inspector (what context did this review actually see?), suppressed-findings view.

The retrieval inspector is the only genuinely interesting page — it is the
debugging tool you'll use constantly and the thing worth demoing.

**Exit:** Lighthouse ≥ 90; zero `any` in generated client code.

---

## M8 — Production hardening

OTel end to end, cost/latency dashboards, Alembic in CI, K8s manifests + HPA on
queue depth, deployment guide, load test at 10× the design point.

**Exit:** a documented cold-start deploy that a stranger can follow.

---

## Suggested sequencing

M0 → M1 → M2 → M3 → **M6 (partial: harness + 30 cases)** → M4 → M5 → M6 (full)
→ M7 → M8

Pulling a slice of M6 forward before M4/M5 is deliberate. Once the review engine
exists, every subsequent change needs a way to answer "did that help?" Building
the analyzer fusion and GitHub integration without measurement means tuning
blind for two milestones.

---

## What to put on the résumé when this is done

Not "built an AI code review platform" — everyone has that line. The line is:

> Built a PR review system with a verification layer that rejects unverifiable
> model output (file/line/AST/patch-applicability gates), and an offline eval
> harness over 120 labeled PRs that gates prompt changes in CI. Raised comment
> precision from 0.41 (naive diff-to-LLM baseline) to 0.83 while holding p50
> latency under 90 s and cost under $0.15/review.

Which means: **measure the naive baseline in M3 before you build the gate.**
Without that 0.41, the 0.83 means nothing. Write the baseline number down the
day you get it.
