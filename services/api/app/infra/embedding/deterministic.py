"""A deterministic, offline :class:`EmbeddingPort` for tests and eval runs.

Not a stub and not random: this is a real hashing embedder (the classic "hashing
trick"), so semantically identical text always produces the identical vector, and
text sharing tokens produces vectors with a real, stable cosine similarity. That
is exactly what the eval harness needs -- ARCHITECTURE sec. 3 requires eval runs
to be *deterministic and free*, and an embedding provider that costs money per
run and drifts between model versions is neither.

It is not a substitute for a trained model: it captures lexical overlap, not
meaning. Retrieval quality numbers measured against it are a floor, not a
forecast, and the graph-expansion half of retrieval is what carries a context
pack anyway (ADR-002). A production adapter implementing the same port is a
network client with the same three members.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

__all__ = ["DEFAULT_DIMENSIONS", "DeterministicEmbedder"]

DEFAULT_DIMENSIONS = 1536
"""Matches ``chunks.embedding vector(1536)`` in the DDL, so fake-embedded
snapshots are storable without a schema change."""

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_MODEL = "deterministic-hash-v1"


class DeterministicEmbedder:
    """Implements :class:`app.domain.ports.EmbeddingPort` with no network."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def model(self) -> str:
        return f"{_MODEL}-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        """Sub-word hashing with sign, then L2 normalization.

        Identifiers are also split on ``_`` and camelCase boundaries so that
        ``get_user`` and ``getUser`` land near each other -- the two languages in
        scope spell the same concept both ways, and a retriever that treats them
        as unrelated tokens would be systematically worse on TypeScript.
        """
        buckets = [0.0] * self._dimensions
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dimensions
            sign = 1.0 if (value >> 63) & 1 else -1.0
            buckets[bucket] += sign
        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0.0:  # no tokens at all (e.g. a chunk of pure punctuation)
            return buckets
        return [value / norm for value in buckets]


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        word = match.group(0)
        tokens.append(word)
        parts = [part for part in word.split("_") if part]
        if len(parts) > 1:
            tokens.extend(parts)
    for match in _TOKEN.finditer(text):
        word = match.group(0)
        camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word)
        if len(camel) > 1:
            tokens.extend(part.lower() for part in camel)
    return tokens
