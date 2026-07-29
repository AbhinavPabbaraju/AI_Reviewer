"""Chunk embedding: the batching, dedup and cache policy (sec. 4.1).

These pin the *cost* behaviour, because that is what the policy exists for. A
change that quietly embeds every chunk on every run still passes any test that
only checks the vectors came back.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from uuid import UUID

import pytest

from app.domain.contracts import CodeSpan
from app.domain.indexing.chunking import content_hash
from app.domain.indexing.embedding import (
    ChunkEmbedder,
    EmbeddingResult,
    EmbeddingSizeError,
)
from app.domain.indexing.models import Chunk, Embedding, Language
from app.infra.embedding.deterministic import DeterministicEmbedder
from app.infra.store.memory import InMemoryEmbeddingCache


class RecordingEmbedder:
    """Counts calls and batch sizes; vectors are positional, not meaningful."""

    def __init__(self, dimensions: int = 4, model: str = "recording-v1") -> None:
        self._dimensions = dimensions
        self._model = model
        self.batches: list[list[str]] = []
        self.in_flight = 0
        self.peak_in_flight = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.batches.append(list(texts))
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(0)  # let sibling batches interleave
        self.in_flight -= 1
        return [[float(index)] * self._dimensions for index in range(len(texts))]

    @property
    def calls(self) -> int:
        return len(self.batches)


_REPO = UUID("00000000-0000-0000-0000-0000000000aa")


def _chunk(content: str, *, fqn: str = "app.m.f", line: int = 1) -> Chunk:
    return Chunk(
        content_hash=content_hash(_REPO, "app/m.py", fqn, content),
        symbol_fqn=fqn,
        span=CodeSpan(path="app/m.py", line_start=line, line_end=line + 1),
        language=Language.PYTHON,
        token_count=max(1, len(content) // 4),
        content=content,
    )


def _embedder(
    provider: RecordingEmbedder, cache: InMemoryEmbeddingCache, **kwargs: int
) -> ChunkEmbedder:
    return ChunkEmbedder(embeddings=provider, cache=cache, **kwargs)


class TestCostPolicy:
    async def test_second_run_over_unchanged_chunks_pays_nothing(self) -> None:
        chunks = [_chunk(f"def f{i}(): return {i}", fqn=f"app.m.f{i}") for i in range(5)]
        provider, cache = RecordingEmbedder(), InMemoryEmbeddingCache()

        first = await _embedder(provider, cache).embed(chunks)
        second = await _embedder(provider, cache).embed(chunks)

        assert first.embedded == 5
        assert second.embedded == 0
        assert second.reused == 5
        assert second.vectors == first.vectors
        assert provider.calls == 1, "the second run must not call the provider"

    async def test_single_changed_chunk_embeds_only_itself(self) -> None:
        chunks = [_chunk(f"def f{i}(): return {i}", fqn=f"app.m.f{i}") for i in range(50)]
        provider, cache = RecordingEmbedder(), InMemoryEmbeddingCache()
        await _embedder(provider, cache).embed(chunks)

        changed = [*chunks[:-1], _chunk("def f49(): return 'changed'", fqn="app.m.f49")]
        result = await _embedder(provider, cache).embed(changed)

        assert result.embedded == 1
        assert result.reused == 49
        assert sum(len(batch) for batch in provider.batches[1:]) == 1

    async def test_duplicate_chunks_are_embedded_once(self) -> None:
        # The same helper copied into two places is one content hash.
        duplicate = _chunk("def helper(): return 1", fqn="app.m.helper")
        provider, cache = RecordingEmbedder(), InMemoryEmbeddingCache()

        result = await _embedder(provider, cache).embed([duplicate, duplicate, duplicate])

        assert result.embedded == 1
        assert result.duplicates == 2
        assert result.total == 1

    async def test_model_change_invalidates_the_cache(self) -> None:
        # Vectors from two models are not comparable; a hit here would silently
        # mix embedding spaces in one ANN index.
        chunks = [_chunk("def f(): return 1")]
        cache = InMemoryEmbeddingCache()
        await _embedder(RecordingEmbedder(model="v1"), cache).embed(chunks)

        result = await _embedder(RecordingEmbedder(model="v2"), cache).embed(chunks)

        assert result.embedded == 1
        assert result.reused == 0


class TestBatching:
    async def test_requests_are_batched_to_the_configured_size(self) -> None:
        chunks = [_chunk(f"def f{i}(): pass", fqn=f"app.m.f{i}") for i in range(10)]
        provider, cache = RecordingEmbedder(), InMemoryEmbeddingCache()

        result = await _embedder(provider, cache, batch_size=4).embed(chunks)

        assert result.batches == 3
        assert [len(batch) for batch in provider.batches] == [4, 4, 2]

    async def test_concurrency_is_bounded(self) -> None:
        chunks = [_chunk(f"def f{i}(): pass", fqn=f"app.m.f{i}") for i in range(20)]
        provider, cache = RecordingEmbedder(), InMemoryEmbeddingCache()

        await _embedder(provider, cache, batch_size=1, max_concurrency=3).embed(chunks)

        assert provider.peak_in_flight <= 3

    async def test_empty_input_calls_nothing(self) -> None:
        provider, cache = RecordingEmbedder(), InMemoryEmbeddingCache()
        result = await _embedder(provider, cache).embed([])
        assert result == EmbeddingResult(
            vectors={}, embedded=0, reused=0, duplicates=0, batches=0
        )
        assert provider.calls == 0

    @pytest.mark.parametrize(("batch_size", "concurrency"), [(0, 1), (1, 0)])
    def test_degenerate_configuration_is_rejected(
        self, batch_size: int, concurrency: int
    ) -> None:
        with pytest.raises(ValueError):
            ChunkEmbedder(
                embeddings=RecordingEmbedder(),
                cache=InMemoryEmbeddingCache(),
                batch_size=batch_size,
                max_concurrency=concurrency,
            )


class TestProviderResponseIsVerified:
    """A provider that returns the wrong shape corrupts every later search, and
    nothing downstream can detect it -- so it fails here, loudly."""

    async def test_short_response_is_fatal(self) -> None:
        class Short(RecordingEmbedder):
            async def embed(
                self, texts: Sequence[str]
            ) -> Sequence[Sequence[float]]:
                return [[0.0] * self.dimensions for _ in texts][:-1]

        with pytest.raises(EmbeddingSizeError, match="vectors for"):
            await _embedder(Short(), InMemoryEmbeddingCache()).embed(
                [_chunk("a"), _chunk("b", fqn="app.m.g")]
            )

    async def test_wrong_dimensionality_is_fatal(self) -> None:
        class Narrow(RecordingEmbedder):
            async def embed(
                self, texts: Sequence[str]
            ) -> Sequence[Sequence[float]]:
                return [[0.0, 1.0] for _ in texts]

        with pytest.raises(EmbeddingSizeError, match="dimensions"):
            await _embedder(Narrow(), InMemoryEmbeddingCache()).embed([_chunk("a")])

    async def test_nothing_is_cached_when_a_batch_is_rejected(self) -> None:
        class Narrow(RecordingEmbedder):
            async def embed(
                self, texts: Sequence[str]
            ) -> Sequence[Sequence[float]]:
                return [[0.0, 1.0] for _ in texts]

        cache = InMemoryEmbeddingCache()
        with pytest.raises(EmbeddingSizeError):
            await _embedder(Narrow(), cache).embed([_chunk("a")])
        assert len(cache) == 0


class TestDeterministicEmbedder:
    """The offline provider the eval harness depends on."""

    async def test_identical_text_gives_identical_vectors(self) -> None:
        embedder = DeterministicEmbedder(dimensions=64)
        first, second = await embedder.embed(["def read(self): ...", "def read(self): ..."])
        assert tuple(first) == tuple(second)

    async def test_vectors_are_unit_length(self) -> None:
        embedder = DeterministicEmbedder(dimensions=64)
        [vector] = await embedder.embed(["def read_user(user_id): return user_id"])
        assert sum(value * value for value in vector) == pytest.approx(1.0)

    async def test_shared_vocabulary_scores_higher_than_unrelated_text(self) -> None:
        embedder = DeterministicEmbedder(dimensions=512)
        query, related, unrelated = await embedder.embed(
            [
                "def load_user(user_id): return store.get(user_id)",
                "def load_user_profile(user_id): return store.get(user_id).profile",
                "class HttpRetryPolicy: backoff = 2.0",
            ]
        )
        assert _cosine(query, related) > _cosine(query, unrelated)

    async def test_snake_case_and_camel_case_spellings_are_close(self) -> None:
        # The two in-scope languages spell the same concept both ways.
        embedder = DeterministicEmbedder(dimensions=512)
        snake, camel, other = await embedder.embed(
            ["get_user_profile", "getUserProfile", "compile_regex_pattern"]
        )
        assert _cosine(snake, camel) > _cosine(snake, other)

    async def test_empty_text_does_not_divide_by_zero(self) -> None:
        [vector] = await DeterministicEmbedder(dimensions=8).embed(["...  ..."])
        assert all(value == 0.0 for value in vector)


class TestInMemoryEmbeddingCache:
    async def test_absent_hashes_are_omitted_not_none(self) -> None:
        cache = InMemoryEmbeddingCache()
        vectors: Mapping[str, Embedding] = {"a" * 64: (1.0, 0.0)}
        await cache.put_many("m", vectors)

        found = await cache.get_many("m", ["a" * 64, "b" * 64])

        assert found == vectors
        assert cache.hits == 1
        assert cache.misses == 1


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
