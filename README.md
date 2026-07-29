# Argus

An AI pull-request reviewer built for **precision, not coverage**: a system designed around a
verification layer that rejects unverifiable model output, and milestone gates that are
measured rather than asserted.

**Work in progress.** What's built today is the foundation the reviewer stands on — the
incremental indexer, the confidence-tagged symbol graph, and structure-first retrieval, each
behind a measured exit gate. The LLM review pass and the verification gate are the next
milestones and are not implemented yet; see [Status](#status) for exactly what does and
doesn't exist.

---

## The problem this actually solves

There are hundreds of "AI PR reviewer" projects, and almost all of them are the same
program: fetch the diff, paste it into a model, post whatever comes back. They fail for one
reason, and it isn't model quality.

**A reviewer that posts 40 comments of which 8 are real gets muted within a week.**
Developers don't grade a reviewer on recall. They grade it on whether the last three
comments it left were worth reading. One confidently wrong comment costs more trust than
ten correct ones earn.

So the design objective isn't "review everything":

> Maximize true findings posted, subject to a hard ceiling on the false-positive rate,
> within a fixed latency and cost budget.

Every decision in [`ARCHITECTURE.md`](ARCHITECTURE.md) falls out of that sentence. The
primary SLO is comment precision ≥ 0.80; recall is explicitly secondary.

Two consequences shape the whole codebase:

- **"Never hallucinate" is a mechanism, not a prompt instruction.** Every finding must pass
  file/line/symbol/patch/dedup/confidence gates before it is eligible to be posted, and
  every drop is counted as a metric.
- **Confidence is never faked.** When the symbol resolver can't determine what
  `self._conn.close()` refers to, it says so and keeps the reference textual, rather than
  binding it to a same-named method on a different class. Fabricated structure is the
  failure mode this project exists to prevent.

## Status

This is a milestone-driven build. A milestone is done when its exit criterion is
**measured and green**, not when the code exists.

| Milestone | Status | What exists |
|---|---|---|
| **M0** Contracts & schema | Contracts done | `Finding`/`Evidence`/`ReviewRun` contracts with validators, forward-only DDL (Postgres 16 + pgvector), port boundaries. The Compose stack and CI workflow from its exit criteria are still outstanding |
| **M1** Ingestion & symbol graph | **Complete — all exit criteria measured** | Blobless clone, filtering, tree-sitter parsers, confidence-tagged edge resolution, symbol-boundary chunking, incremental re-index |
| **M2** Embeddings & retrieval | **Partial** | Embedding pipeline + content-hash cache, graph expansion, fusion ranking, budget enforcement, retrieval-quality gate |
| M3 Review engine & verification gate | Not started | — |
| M4 Deterministic analyzers | Not started | — |
| M5 GitHub App | Not started | — |
| M6 Evaluation harness | Not started | — |
| M7 Dashboard · M8 Hardening | Not started | — |

**Not built yet, and deliberately not claimed:** there is no LLM call anywhere in the
pipeline, no GitHub integration, no HTTP API, and nothing persists to Postgres — the DDL is
written but the only storage adapters are in-memory. There is also no Docker Compose file and
no CI workflow yet, so the gates below are run locally. M2 still needs the pgvector adapter,
a network embedding provider, and `POST /search`. See [`ROADMAP.md`](ROADMAP.md).

## Measured results

These come from the gates in `tests/eval/`, which print their numbers rather than silently
asserting them (`pytest tests/eval -s`):

| Gate | Criterion | Measured |
|---|---|---|
| Symbol resolution, Python | ≥ 85% on hand-labeled references | **99.0%** over 304 references |
| Symbol resolution, TypeScript | ≥ 85% | **99.6%** over 225 references |
| Cold index, 5,050-file repo | < 4 min | **20.4 s** |
| Single-file push re-index | < 10 s | **4.0 s** — 1 file re-parsed, 29,000 of 29,001 chunks reused |
| Retrieval: needed file in the pack | ≥ 90% over 30 hand-built queries | **95.0%** over 40 queries |
| Retrieval p95 | < 800 ms | 2.6 ms *(in-memory adapter — a floor, not a production forecast)* |

Two things worth knowing about how these are measured, because they're the difference
between a gate and a decoration:

- **The corpus labels are ground truth, not a recording of the resolver's output.** Every
  expectation was written by reading the corpus source. Where the resolver disagrees, the
  correct label stays and the case is committed as a known miss — the four current ones are
  inherent ambiguity (dispatch through an interface-typed parameter, attributes with no
  declared type). A corpus labeled from program output can only ever score 100%.
- **A gate that can't fail measures nothing.** The retrieval gate runs at a deliberately
  tight token budget, because at a production-sized budget these corpora fit entirely in one
  pack and recall is trivially 100%. At the gate's budget, ~28 of ~40 candidates are dropped
  per pack, so the ranking has to actually be right.

## How it works

Six stages. Stages I–III are built — the worker pipeline runs I–II end to end today, and
retrieval is driven directly by its tests and its gate, since there is no review stage to
call it yet:

```
I.   Incremental indexing   clone (bare, blobless) → filter → parse (tree-sitter)
                            → chunk by symbol → embed changed chunks only
II.  Symbol graph           CALLS · IMPORTS · INHERITS · REFERENCES · TESTS,
                            each confidence-tagged, unresolved refs kept not dropped
III. Hybrid retrieval       anchor on changed symbols → BFS depth ≤ 2 → ANN supplement
                            → fusion rank → token budget, dropping whole symbols
IV.  Analyzers              Semgrep · Ruff · ESLint/tsc, sandboxed        (not built)
V.   LLM review             constrained JSON, temperature 0, versioned    (not built)
VI.  Verification gate      seven gates; every drop counted                (not built)
```

**Retrieval is structure-first, and the code says so.** Cosine similarity answers
"what looks like this diff?" — but the caller that passes `None` is what makes the
null-deref a bug, and it's often not textually similar to the callee at all. Graph proximity
is weighted above semantic similarity, and the gate measures the split: currently **98% of
retrieved context arrives through the symbol graph**, with embeddings supplementing. If that
ratio ever flips, the retriever has quietly become a vector search with extra steps.

**Incrementality is content-addressed throughout.** `chunk_id = sha256(repo_id | path |
symbol_fqn | normalized_body)` is both the parse-cache key and the embedding-cache key, so a
one-line change to a 50k-file repo re-embeds ~1 chunk, not 50k.

## Getting started

Requires Python 3.12+ and `git`. No network, database, or API key is needed to run anything
in the repo today.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest              # 255 tests, ~1.5 s
.venv/bin/python -m pytest tests/eval -s   # the milestone gates, with their numbers
.venv/bin/python -m pytest -m slow -s      # the 5,050-file latency gate (~25 s)
.venv/bin/python -m mypy                # strict, must stay clean
.venv/bin/ruff check services tests
```

### Indexing a repository and building a context pack

Argus can index itself. From the repository root:

```python
import asyncio
from uuid import uuid4

from app.domain.contracts import CodeSpan
from app.domain.indexing.embedding import ChunkEmbedder
from app.domain.retrieval.retriever import ContextRetriever
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.parsing.registry import default_parsers
from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore
from app.infra.source.git_source import GitSourceProvider
from app.infra.store.memory import InMemoryEmbeddingCache, InMemoryIndexStore
from worker.pipeline.indexer import Indexer


async def main() -> None:
    repo_id, store = uuid4(), InMemoryIndexStore()

    result = await Indexer(
        source=GitSourceProvider(),
        parsers=default_parsers(),
        cache=store,
        store=store,
        embedder=ChunkEmbedder(
            embeddings=DeterministicEmbedder(), cache=InMemoryEmbeddingCache()
        ),
    ).index(repository_id=repo_id, repo_url=".", commit_sha="HEAD")

    print(f"{result.files_indexed} files, {result.symbols} symbols, {result.edges} edges")

    snapshot = store.snapshot(result.snapshot_id)
    pack = await ContextRetriever(
        index=InMemorySymbolIndex(snapshot),
        vectors=InMemoryVectorStore(snapshot),
        embeddings=DeterministicEmbedder(),
    ).retrieve(
        repository_id=str(repo_id),
        hunks=[CodeSpan(path="services/api/app/domain/indexing/resolution.py",
                        line_start=253, line_end=258)],
    )

    for item in pack.items[:6]:
        print(f"  {item.provenance.value:8} {item.symbol_fqn}  ({item.reason})")


asyncio.run(main())
```

Every item in a pack explains why it's there:

```
60 files, 688 symbols, 2116 edges
  anchor   ...indexing.resolution.Resolver                    (changed by this diff)
  anchor   ...indexing.resolution.Resolver._follow_wildcard   (changed by this diff)
  graph    ...indexing.resolution._Resolved                   (called by ..._follow_wildcard)
  graph    ...indexing.resolution.Resolver._follow_reexport   (calls ..._follow_wildcard)
  graph    ...indexing.resolution.Resolver._resolve_module    (called by ..._follow_wildcard)
  graph    ...indexing.resolution._Scope                      (referenced by ..._follow_wildcard)
```

*(fully-qualified names abbreviated for width)*

That provenance isn't decoration — it's what makes a bad review diagnosable, and it's what
the M7 retrieval inspector renders.

> Note: the resolution rate reported on an arbitrary real-world repository runs lower than
> on the labeled corpus (~74% on Argus's own source, which leans on third-party libraries
> and dynamic attribute access). That's the design working as intended: references that
> can't be resolved are kept with their textual target and a low confidence, then
> down-weighted during expansion — never dropped, and never guessed.

## Repository layout

```
db/migrations/0001_init.sql   forward-only DDL (Postgres 16 + pgvector)
services/api/app/
  domain/                     pure: contracts, ports, algorithms — imports nothing from infra
    contracts.py ports.py     the output contract and pipeline ports
    indexing/                 filtering, chunking, incremental planning, resolution, embedding
    retrieval/                graph expansion, fusion ranking, budget, the retriever
  infra/                      adapters implementing the ports
    source/                   blobless bare clone
    parsing/                  tree-sitter extractors (Python, TypeScript)
    embedding/                offline deterministic embedder
    store/  retrieval/        in-memory caches, index store, symbol index, vector store
services/worker/worker/pipeline/indexer.py    the Stage I/II orchestrator
tests/{domain,infra,worker,eval}/             unit tests + the measured milestone gates
```

`app` and `worker` are two top-level packages installed as one editable distribution; the
worker imports `app.domain`.

**The dependency rule:** `domain/` imports nothing from `infra/`. Adapters implement the
`Protocol`s in `ports.py`. This is the only piece of Clean Architecture ceremony kept, and
it exists for one concrete reason: the M6 eval harness runs the whole pipeline against a
fake GitHub adapter and a recorded LLM adapter, so eval runs are deterministic and free.
Without the port boundary, that harness is impossible.

## Design decisions worth knowing

Full reasoning, including the options rejected, is in [`ARCHITECTURE.md`](ARCHITECTURE.md)
§7. In brief:

- **Precision-first review policy** (ADR-001) — a hard false-positive ceiling, with recall
  as the adjustable variable.
- **Hybrid structural + semantic retrieval** (ADR-002) — the symbol graph is primary,
  vectors supplement. Costs a language-specific resolver per language, which is precisely
  why the language list is capped at two.
- **pgvector over a dedicated vector DB** (ADR-003) — the corpus ceiling is ~5M chunks, two
  orders of magnitude below where a dedicated store wins, and every query is filtered by
  `repo_id` and joined to relational rows anyway.
- **Scope is Python + TypeScript only.** Two languages done properly beats eight done by
  regex. Adding a third is a budgeted decision, not a config flag.

[`ARCHITECTURE.md`](ARCHITECTURE.md) §2 also documents what was deliberately cut from the
original brief and why — the knowledge graph, CQRS, day-one Kubernetes, seven analyzers down
to three. "Why we didn't build it" is the part of a design doc that actually gets read.

## Contributing / working on this

[`CLAUDE.md`](CLAUDE.md) holds the working norms: the invariants that must not be broken,
how the measured gates work, and where the layering boundaries are. The short version — when
making a change, ask "does this protect precision?" before "does this find more bugs?"

## License

MIT — see [`LICENSE`](LICENSE).
