"""End-to-end retrieval: a real indexed repository in, a context pack out.

Nothing is hand-built here -- the repo is cloned and indexed by the real Stage
I/II pipeline, embedded by the deterministic embedder, and retrieved through the
in-memory adapters. A pack assembled this way exercises every seam at once:
chunker to symbol index, resolver to graph expansion, embedder to ANN.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import pytest

from app.domain.contracts import CodeSpan
from app.domain.indexing.embedding import ChunkEmbedder
from app.domain.indexing.ports import SnapshotWrite
from app.domain.retrieval.models import Provenance
from app.domain.retrieval.retriever import ContextRetriever, RetrievalConfig
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.parsing.registry import default_parsers
from app.infra.retrieval.memory import InMemorySymbolIndex, InMemoryVectorStore
from app.infra.source.git_source import GitSourceProvider
from app.infra.store.memory import InMemoryEmbeddingCache, InMemoryIndexStore
from tests.conftest import GitRepo
from worker.pipeline.indexer import Indexer

# A caller that passes None, the callee that dereferences it, a base class, a
# test, and an unrelated module that merely *looks* similar. That last file is
# the trap for a similarity-only retriever.
_REPO = {
    "app/store.py": (
        "class BaseStore:\n"
        "    def connect(self):\n"
        "        return None\n"
        "\n"
        "\n"
        "class UserStore(BaseStore):\n"
        "    def save(self, user):\n"
        "        return user.name.lower()\n"
    ),
    "app/handler.py": (
        "from app.store import UserStore\n"
        "\n"
        "\n"
        "def handle(request):\n"
        "    store = UserStore()\n"
        "    return store.save(request.get('user'))\n"
    ),
    "tests/test_store.py": (
        "from app.store import UserStore\n"
        "\n"
        "\n"
        "def test_save():\n"
        "    return UserStore().save(None)\n"
    ),
    "app/reporting.py": (
        "def save_report(report):\n"
        "    return report.name.lower()\n"
    ),
}


@pytest.fixture
async def indexed(
    make_git_repo: Callable[[], GitRepo],
) -> tuple[SnapshotWrite, str]:
    repo = make_git_repo()
    sha = repo.commit(_REPO, "init")
    repository_id = uuid4()
    store = InMemoryIndexStore()
    result = await Indexer(
        source=GitSourceProvider(),
        parsers=default_parsers(),
        cache=store,
        store=store,
        embedder=ChunkEmbedder(
            embeddings=DeterministicEmbedder(dimensions=256),
            cache=InMemoryEmbeddingCache(),
        ),
    ).index(repository_id=repository_id, repo_url=repo.url, commit_sha=sha)
    return store.snapshot(result.snapshot_id), str(repository_id)


def _retriever(
    snapshot: SnapshotWrite, *, config: RetrievalConfig | None = None
) -> ContextRetriever:
    return ContextRetriever(
        index=InMemorySymbolIndex(snapshot),
        vectors=InMemoryVectorStore(snapshot),
        embeddings=DeterministicEmbedder(dimensions=256),
        config=config,
    )


_SAVE_HUNK = CodeSpan(path="app/store.py", line_start=7, line_end=8)


class TestContextPack:
    async def test_anchors_on_the_changed_symbol(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        snapshot, repository_id = indexed
        pack = await _retriever(snapshot).retrieve(
            repository_id=repository_id, hunks=[_SAVE_HUNK]
        )
        anchors = {item.symbol_fqn for item in pack.by_provenance(Provenance.ANCHOR)}
        assert "app.store.UserStore.save" in anchors

    async def test_reaches_the_caller_that_makes_the_change_a_bug(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        # `handle` passes a possibly-missing value into `save`, which
        # dereferences it. Nothing about the two functions is textually similar;
        # only the CALLS edge connects them. This is ADR-002's whole argument.
        snapshot, repository_id = indexed
        pack = await _retriever(snapshot).retrieve(
            repository_id=repository_id, hunks=[_SAVE_HUNK]
        )
        graph = {item.symbol_fqn: item for item in pack.by_provenance(Provenance.GRAPH)}
        assert "app.handler.handle" in graph
        # The hunk covers both the class and the method, so `handle` is reached
        # by whichever of its two calls scores higher -- either explains it.
        assert graph["app.handler.handle"].reason.startswith(
            "calls app.store.UserStore"
        )
        assert graph["app.handler.handle"].graph_distance == 1

    async def test_reaches_the_base_class_and_the_test(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        snapshot, repository_id = indexed
        pack = await _retriever(snapshot).retrieve(
            repository_id=repository_id, hunks=[_SAVE_HUNK]
        )
        reached = {item.symbol_fqn for item in pack.items}
        assert "app.store.BaseStore" in reached, "base class of the changed class"
        assert "tests.test_store.test_save" in reached, "the test that covers it"

    async def test_every_item_explains_itself(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        # The M7 retrieval inspector renders these; a pack that cannot say why
        # it contains something makes a bad review impossible to diagnose.
        snapshot, repository_id = indexed
        pack = await _retriever(snapshot).retrieve(
            repository_id=repository_id, hunks=[_SAVE_HUNK]
        )
        assert pack.items
        assert all(item.reason for item in pack.items)
        assert all(
            item.graph_distance is not None
            for item in pack.by_provenance(Provenance.GRAPH)
        )

    async def test_semantic_supplement_does_not_duplicate_the_graph(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        snapshot, repository_id = indexed
        pack = await _retriever(snapshot).retrieve(
            repository_id=repository_id, hunks=[_SAVE_HUNK]
        )
        hashes = [item.chunk.content_hash for item in pack.items]
        assert len(hashes) == len(set(hashes))
        graph_paths = {
            item.path
            for item in (
                *pack.by_provenance(Provenance.ANCHOR),
                *pack.by_provenance(Provenance.GRAPH),
            )
        }
        semantic_paths = {
            item.path for item in pack.by_provenance(Provenance.SEMANTIC)
        }
        assert not (graph_paths & semantic_paths)

    async def test_similar_but_unconnected_code_ranks_below_the_caller(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        # `save_report` is nearly a textual twin of `save` and has no edge to it.
        # A similarity-first retriever puts it above the caller; this one must not.
        snapshot, repository_id = indexed
        pack = await _retriever(snapshot).retrieve(
            repository_id=repository_id, hunks=[_SAVE_HUNK]
        )
        ranked = [item.symbol_fqn for item in pack.items]
        assert "app.handler.handle" in ranked
        if "app.reporting.save_report" in ranked:
            assert ranked.index("app.handler.handle") < ranked.index(
                "app.reporting.save_report"
            )

    async def test_budget_drops_whole_symbols_and_keeps_the_anchor(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        snapshot, repository_id = indexed
        pack = await _retriever(
            snapshot, config=RetrievalConfig(token_budget=12)
        ).retrieve(repository_id=repository_id, hunks=[_SAVE_HUNK])

        assert pack.by_provenance(Provenance.ANCHOR)
        assert pack.stats.dropped_by_budget > 0
        stored = {chunk.content_hash: chunk for chunk in snapshot.chunks}
        assert all(
            item.chunk.content == stored[item.chunk.content_hash].content
            for item in pack.items
        ), "every kept item is the whole indexed chunk, byte for byte"

    async def test_retrieval_without_a_vector_store_still_works(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        # Graph-only retrieval is a valid configuration: the structural half is
        # the foundation, embeddings are the supplement (ADR-002).
        snapshot, repository_id = indexed
        pack = await ContextRetriever(index=InMemorySymbolIndex(snapshot)).retrieve(
            repository_id=repository_id, hunks=[_SAVE_HUNK]
        )
        assert pack.by_provenance(Provenance.SEMANTIC) == ()
        assert {item.symbol_fqn for item in pack.items} >= {
            "app.store.UserStore.save",
            "app.handler.handle",
        }

    async def test_another_repositorys_id_retrieves_nothing(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        # An unscoped retrieval is a cross-tenant data leak, not a bug report.
        snapshot, _ = indexed
        pack = await _retriever(snapshot).retrieve(
            repository_id=str(uuid4()), hunks=[_SAVE_HUNK]
        )
        assert pack.items == ()

    async def test_stats_are_measured_not_asserted(
        self, indexed: tuple[SnapshotWrite, str]
    ) -> None:
        snapshot, repository_id = indexed
        pack = await _retriever(snapshot).retrieve(
            repository_id=repository_id, hunks=[_SAVE_HUNK]
        )
        assert pack.stats.anchors >= 1
        assert pack.stats.tokens_used == sum(item.token_count for item in pack.items)
        assert 0.0 < pack.stats.budget_utilization <= 1.0
