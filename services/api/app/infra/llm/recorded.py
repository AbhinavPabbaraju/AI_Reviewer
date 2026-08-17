"""A replaying :class:`LLMPort`. Deterministic, offline, and free.

ARCHITECTURE sec. 3 names this as half of why the port boundary exists: the M6
evaluation harness runs the whole pipeline against a fake GitHub and a *recorded*
LLM so eval runs are deterministic and free. It is also what lets the review
pipeline be tested at all -- a test that called a real model would be slow, cost
money, and fail intermittently for reasons that have nothing to do with the code
under test.

Cassettes are keyed by a hash of the exact ``(system, user)`` pair. That is a
deliberately brittle key: change the prompt and every recording misses. Which is
correct -- a recording made under ``reviewer/v1`` is not evidence about what
``reviewer/v2`` does, and silently replaying it would make a prompt change look
free when it is exactly the thing being evaluated.

Two modes:

* **Replay** (default) -- a miss raises. Used by tests and CI, where a miss means
  the prompt changed and the cassette needs regenerating.
* **Record** -- wraps a live provider, forwards misses, and stores the result.
  Used once, deliberately, to produce a cassette others replay for free.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.domain.ports import LLMPort, LLMResponse

__all__ = ["CassetteMiss", "RecordedLLM"]


class CassetteMiss(LookupError):
    """No recording for this prompt.

    Carries the key and a prompt excerpt because the usual cause is a prompt
    edit, and "which prompt" is the first thing anyone asks.
    """


def cassette_key(system: str, user: str) -> str:
    """Stable identity for one prompt pair."""
    digest = hashlib.sha256()
    digest.update(system.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(user.encode("utf-8"))
    return digest.hexdigest()[:32]


class RecordedLLM:
    """Implements :class:`app.domain.ports.LLMPort` from stored responses."""

    def __init__(
        self,
        entries: dict[str, dict[str, Any]] | None = None,
        *,
        delegate: LLMPort | None = None,
        model: str = "recorded",
    ) -> None:
        self._entries: dict[str, dict[str, Any]] = dict(entries or {})
        self._delegate = delegate
        self._model = model
        self.hits = 0
        self.misses = 0

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_file(cls, path: Path, *, delegate: LLMPort | None = None) -> RecordedLLM:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw.get("entries", {}), delegate=delegate)

    def to_file(self, path: Path) -> None:
        """Write the cassette. Sorted and indented so a re-record produces a
        reviewable diff rather than a reordered blob."""
        Path(path).write_text(
            json.dumps({"entries": self._entries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def replaying(self) -> RecordedLLM:
        """A replay-only copy: same recordings, no provider behind it.

        The record-then-replay handoff, made explicit. A cassette produced with
        a delegate attached would still reach for that delegate on a miss, which
        is exactly what an offline run must not do -- so the offline object is a
        different object, and a miss on it raises.
        """
        return RecordedLLM(self._entries, delegate=None, model=self._model)

    def record(self, *, system: str, user: str, content: str, **usage: Any) -> None:
        """Add a response by hand. For tests that want a specific reply without
        ever touching a provider."""
        self._entries[cassette_key(system, user)] = {
            "content": content,
            "tokens_in": usage.get("tokens_in", 0),
            "tokens_out": usage.get("tokens_out", 0),
            "cost_usd": usage.get("cost_usd", 0.0),
            "model": usage.get("model", self._model),
            "prompt_excerpt": user[:200],
        }

    # -- LLMPort ----------------------------------------------------------- #

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        key = cassette_key(system, user)
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
            return LLMResponse(
                content=entry["content"],
                tokens_in=int(entry.get("tokens_in", 0)),
                tokens_out=int(entry.get("tokens_out", 0)),
                # Replayed responses are free by construction. Reporting the
                # original call's price would inflate every eval run's cost with
                # money nobody is spending.
                cost_usd=0.0,
                model=str(entry.get("model", self._model)),
            )

        self.misses += 1
        if self._delegate is None:
            raise CassetteMiss(
                f"no recorded response for prompt {key}; the prompt has changed "
                f"or the cassette is incomplete. First 200 characters of the "
                f"user prompt: {user[:200]!r}"
            )

        live = await self._delegate.complete(
            system=system,
            user=user,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._entries[key] = {
            "content": live.content,
            "tokens_in": live.tokens_in,
            "tokens_out": live.tokens_out,
            "cost_usd": live.cost_usd,
            "model": live.model,
            "prompt_excerpt": user[:200],
        }
        return live

    @property
    def model(self) -> str:
        return self._model

    def keys(self) -> Sequence[str]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
