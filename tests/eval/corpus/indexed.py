"""Turn a corpus of source text into an indexed snapshot, once, for both gates.

Extracted from ``test_retrieval_quality.py`` when the pgvector gate arrived.
The extraction is the point: the in-memory gate and the Postgres gate have to
measure the *same* corpus, the same queries and the same pack-scoring code, or
"recall is 95% on both" would be two numbers about two systems rather than
evidence that the adapters are interchangeable.

Nothing here is a fixture or a fake -- it is real parsing, real resolution, real
chunking and real (deterministic) embedding. Only git is skipped.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol
from uuid import UUID, uuid4

from app.domain.contracts import CodeSpan
from app.domain.indexing.chunking import chunk_parsed_file
from app.domain.indexing.embedding import ChunkEmbedder
from app.domain.indexing.models import Chunk, ParsedFile, Symbol
from app.domain.indexing.ports import SnapshotWrite
from app.domain.indexing.resolution import Resolver
from app.domain.retrieval.models import ContextPack
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.store.memory import InMemoryEmbeddingCache
from tests.conftest import parse_python, parse_typescript
from tests.eval.corpus.retrieval_queries import RetrievalQuery

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "IndexedCorpus",
    "QueryOutcome",
    "RetrieverLike",
    "build_corpus",
    "run_queries",
]

EMBEDDING_DIMENSIONS: Final = 1536
"""Production width, matching ``chunks.embedding vector(1536)`` in the DDL.

Was 256 while retrieval was in-memory only, where narrower vectors were simply
cheaper. pgvector will not accept them: the column is dimensioned, and it has to
be, because an HNSW index cannot be built over a column of unknown width. Rather
than embed one width for the fake and another for the database -- which would
make "the adapters agree" a claim about two different vector spaces -- both run
at 1536. The gate's recall and provenance numbers are unchanged by the switch;
only the fake's own cosine loop got slower, which is a cost paid by the thing
that is not production."""

_PARSER_VERSION: Final = "treesitter/1"


@dataclass(frozen=True, slots=True)
class IndexedCorpus:
    """An indexed corpus: the snapshot plus the symbol table queries name."""

    repository_id: UUID
    snapshot: SnapshotWrite
    symbols: Mapping[str, Symbol]

    @property
    def repository_key(self) -> str:
        """The string form the retrieval ports are keyed by."""
        return str(self.repository_id)

    def span_of(self, fqn: str) -> CodeSpan:
        symbol = self.symbols.get(fqn)
        assert symbol is not None, f"query names a symbol that does not exist: {fqn}"
        return symbol.span


async def build_corpus(
    sources: Mapping[str, str], test_files: frozenset[str], *, typescript: bool
) -> IndexedCorpus:
    """Index a corpus without git: parse, resolve, chunk, embed."""
    parse = parse_typescript if typescript else parse_python
    repository_id = uuid4()
    parsed: list[ParsedFile] = [
        parse(path, text, is_test=path in test_files) for path, text in sources.items()
    ]
    resolution = Resolver(parsed).resolve()
    chunks: list[Chunk] = [
        chunk
        for unit in parsed
        for chunk in chunk_parsed_file(repository_id, unit, sources[unit.file.path])
    ]
    embedder = ChunkEmbedder(
        embeddings=DeterministicEmbedder(dimensions=EMBEDDING_DIMENSIONS),
        cache=InMemoryEmbeddingCache(),
    )
    embedded = await embedder.embed(chunks)

    snapshot = SnapshotWrite(
        id=uuid4(),
        repository_id=repository_id,
        commit_sha="0" * 40,
        parent_snapshot_id=None,
        parser_version=_PARSER_VERSION,
        embedding_model=embedder.model,
        files=tuple(unit.file for unit in parsed),
        symbols=tuple(s for unit in parsed for s in unit.symbols),
        edges=resolution.edges,
        chunks=tuple(chunks),
        embeddings=embedded.vectors,
    )
    return IndexedCorpus(
        repository_id=repository_id,
        snapshot=snapshot,
        symbols={s.fqn: s for unit in parsed for s in unit.symbols},
    )


class RetrieverLike(Protocol):
    """What the gates need from a retriever -- satisfied by ``ContextRetriever``
    wired to either adapter set."""

    async def retrieve(
        self, *, repository_id: str, hunks: Sequence[CodeSpan]
    ) -> ContextPack: ...


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    query: RetrievalQuery
    hit: bool
    duration_ms: float
    pack: ContextPack


async def run_queries(
    retriever: RetrieverLike,
    corpus: IndexedCorpus,
    queries: Sequence[RetrievalQuery],
) -> list[QueryOutcome]:
    """Retrieve for each query and time it. The timing is wall-clock around the
    whole ``retrieve`` call, so the Postgres runs include their round trips --
    which is the entire reason to re-measure p95 against a real database."""
    outcomes: list[QueryOutcome] = []
    for query in queries:
        hunk = corpus.span_of(query.changed_symbol)
        started = time.perf_counter()
        pack = await retriever.retrieve(
            repository_id=corpus.repository_key, hunks=[hunk]
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        outcomes.append(
            QueryOutcome(
                query=query,
                hit=query.needs in pack.paths,
                duration_ms=elapsed_ms,
                pack=pack,
            )
        )
    return outcomes
