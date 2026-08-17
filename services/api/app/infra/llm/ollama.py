"""A local-model :class:`LLMPort` via Ollama. Runs on your hardware, costs nothing.

This is the adapter that makes Argus operable without buying API credits. Every
other stage was already free -- the embedder is an offline hashing model, retrieval
is structure-first over a local Postgres, the verification gate is pure computation
-- and Stage V was the only step that assumed a paid provider. Ollama closes that
gap with a real model rather than a stub: it is genuinely doing the review, just on
a model you host.

``cost_usd`` is therefore always ``0.0``, and that is a fact rather than a
placeholder. The cost SLO in ARCHITECTURE sec. 1 is about money leaving the
building; electricity is real but it is not what the budget tracks, and reporting
an invented dollar figure would corrupt the one number the eval harness uses to
compare configurations.

**Model choice matters more here than anywhere else in the system.** A small
local model is a weaker reviewer than a frontier one, and the honest consequence
is lower recall -- it will miss defects. It is *not* licence for lower precision:
the verification gate applies identically whatever produced the finding, so a
local model's fabrications are caught by the same mechanism. Precision is
structural; recall is what you trade away by running free.
"""

from __future__ import annotations

import json
from typing import Any, Final

import httpx

from app.domain.ports import LLMResponse

__all__ = ["DEFAULT_MODEL", "OllamaLLM"]

DEFAULT_BASE_URL: Final = "http://localhost:11434"

DEFAULT_MODEL: Final = "qwen2.5-coder:7b"
"""A code-specialized model that fits in ~8 GB of RAM and follows a JSON schema
reliably enough for the decoder. Chosen as a default that works on a laptop, not
as a recommendation over anything larger -- pass ``model=`` to use what you have."""


class OllamaLLM:
    """Implements :class:`app.domain.ports.LLMPort` against a local Ollama server."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        # Local generation on CPU is slow -- minutes for a long review is normal,
        # and a default 5-second client timeout would turn "working" into
        # "failed" on exactly the hardware this adapter exists to support.
        self._timeout = timeout_seconds
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_schema is not None:
            # Ollama constrains decoding to a JSON schema when given one, which
            # is the local equivalent of structured outputs. Passing the schema
            # rather than the bare "json" mode is what stops the model inventing
            # its own envelope around the findings array.
            payload["format"] = json_schema

        data = await self._post(payload)

        message = data.get("message") or {}
        return LLMResponse(
            content=str(message.get("content", "")),
            # Ollama reports token counts under these names; absent on some
            # versions, in which case zero is the honest answer rather than a
            # guess derived from string length.
            tokens_in=int(data.get("prompt_eval_count", 0)),
            tokens_out=int(data.get("eval_count", 0)),
            cost_usd=0.0,
            model=str(data.get("model", self._model)),
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            response = await self._client.post(
                f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
            )
            return self._parse(response)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            return self._parse(response)

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 404:
            # Ollama's 404 for an unpulled model is easy to mistake for a bad
            # URL, and the fix ("ollama pull <model>") is worth stating.
            raise OllamaUnavailable(
                f"Ollama returned 404: the model may not be pulled yet. "
                f"Try `ollama pull <model>`. Response: {response.text[:200]}"
            )
        response.raise_for_status()
        parsed: dict[str, Any] = json.loads(response.text)
        return parsed


class OllamaUnavailable(RuntimeError):
    """Ollama is reachable but cannot serve the request as configured."""
