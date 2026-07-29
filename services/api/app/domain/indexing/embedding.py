"""Chunk embedding: the batching and cache policy (ARCHITECTURE sec. 4.1).

Embedding is the one part of indexing that costs money per token, so the policy
around it is the interesting part, not the call itself:

* **Never pay twice for one chunk.** Chunk ids are content hashes, so a chunk
  whose hash is already in the cache is already embedded, whatever file or commit
  it arrived in. "A one-line change to a 50k-file repo re-embeds ~1 chunk, not
  50k" is this rule plus the content-addressed chunker.
* **Deduplicate within the run too.** The same helper copied into two files is
  one hash and must be one embedding, not two.
* **Batch, then bound the concurrency.** Providers price and rate-limit per
  request; one call per chunk is both slow and expensive, and unbounded fan-out
  gets the account throttled at exactly the moment a large repo is indexing.
* **Verify what comes back.** A provider returning the wrong number of vectors,
  or vectors of the wrong width, is a silent corruption of the ANN index -- every
  later search returns plausible nonsense. It is checked here, once, loudly.

The result is deterministic: vectors come back keyed by content hash, so nothing
downstream depends on batch ordering or on which chunk happened to be first.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.indexing.models import Chunk, Embedding
from app.domain.indexing.ports import EmbeddingCachePort
from app.domain.ports import EmbeddingPort

__all__ = ["ChunkEmbedder", "EmbeddingResult", "EmbeddingSizeError"]

DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CONCURRENCY = 4


class EmbeddingSizeError(RuntimeError):
    """The provider returned the wrong number of vectors, or the wrong width.

    Deliberately fatal to the indexing run: a snapshot embedded from a
    mismatched response is worse than an unembedded one, because nothing
    downstream can detect it.
    """


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """What one embedding pass cost. Every field is a measured fact, reported on
    the run so cache effectiveness is observable rather than assumed."""

    vectors: Mapping[str, Embedding]
    embedded: int
    """Chunks sent to the provider (i.e. actually paid for)."""
    reused: int
    """Distinct chunks served from the cache."""
    duplicates: int
    """Chunks in this run that collapsed onto another chunk's hash."""
    batches: int

    @property
    def total(self) -> int:
        return self.embedded + self.reused


class ChunkEmbedder:
    """Embeds a snapshot's chunks through an :class:`EmbeddingPort`, paying only
    for hashes no cache has seen."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingPort,
        cache: EmbeddingCachePort,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._embeddings = embeddings
        self._cache = cache
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency

    @property
    def model(self) -> str:
        return self._embeddings.model

    async def embed(self, chunks: Sequence[Chunk]) -> EmbeddingResult:
        unique: dict[str, Chunk] = {}
        for chunk in chunks:
            unique.setdefault(chunk.content_hash, chunk)
        duplicates = len(chunks) - len(unique)
        if not unique:
            return EmbeddingResult(
                vectors={}, embedded=0, reused=0, duplicates=duplicates, batches=0
            )

        model = self._embeddings.model
        cached = dict(await self._cache.get_many(model, tuple(unique)))
        missing = [h for h in unique if h not in cached]
        if not missing:
            return EmbeddingResult(
                vectors=cached,
                embedded=0,
                reused=len(cached),
                duplicates=duplicates,
                batches=0,
            )

        batches = [
            missing[start : start + self._batch_size]
            for start in range(0, len(missing), self._batch_size)
        ]
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def run(batch: Sequence[str]) -> Mapping[str, Embedding]:
            async with semaphore:
                return await self._embed_batch(batch, unique)

        fresh: dict[str, Embedding] = {}
        for produced in await asyncio.gather(*(run(batch) for batch in batches)):
            fresh.update(produced)

        await self._cache.put_many(model, fresh)
        return EmbeddingResult(
            vectors={**cached, **fresh},
            embedded=len(fresh),
            reused=len(cached),
            duplicates=duplicates,
            batches=len(batches),
        )

    async def _embed_batch(
        self, content_hashes: Sequence[str], chunks: Mapping[str, Chunk]
    ) -> dict[str, Embedding]:
        texts = [chunks[content_hash].content for content_hash in content_hashes]
        vectors = await self._embeddings.embed(texts)
        if len(vectors) != len(texts):
            raise EmbeddingSizeError(
                f"provider returned {len(vectors)} vectors for {len(texts)} chunks"
            )
        expected = self._embeddings.dimensions
        produced: dict[str, Embedding] = {}
        for content_hash, vector in zip(content_hashes, vectors, strict=True):
            if len(vector) != expected:
                raise EmbeddingSizeError(
                    f"chunk {content_hash[:12]} embedded to {len(vector)} "
                    f"dimensions, expected {expected}"
                )
            produced[content_hash] = tuple(vector)
        return produced
