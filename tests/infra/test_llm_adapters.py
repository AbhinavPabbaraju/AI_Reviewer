"""The two free ``LLMPort`` adapters: recorded replay and local Ollama.

Neither test touches a paid provider, and neither needs one to be meaningful:
the recorded adapter *is* the offline path, and the Ollama adapter is exercised
against a mock transport that asserts the exact request shape it sends. What a
mock transport cannot prove is that a real Ollama server accepts that shape --
that is stated in the module docstring rather than implied by a green test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.domain.ports import LLMResponse
from app.infra.llm.ollama import DEFAULT_MODEL, OllamaLLM, OllamaUnavailable
from app.infra.llm.recorded import CassetteMiss, RecordedLLM, cassette_key

SYSTEM = "You are Argus."
USER = "Review this change."


class TestRecordedLLM:
    async def test_replays_a_recorded_response(self) -> None:
        llm = RecordedLLM()
        llm.record(system=SYSTEM, user=USER, content='{"findings": []}')
        response = await llm.complete(system=SYSTEM, user=USER)
        assert response.content == '{"findings": []}'
        assert llm.hits == 1

    async def test_a_miss_raises_with_the_prompt_in_the_message(self) -> None:
        """A miss means the prompt changed. The usual next question is 'which
        prompt', so the error answers it."""
        llm = RecordedLLM()
        with pytest.raises(CassetteMiss, match="Review this change"):
            await llm.complete(system=SYSTEM, user=USER)
        assert llm.misses == 1

    async def test_a_changed_prompt_misses(self) -> None:
        """Deliberately brittle: a recording made under one prompt is not
        evidence about what a different prompt does."""
        llm = RecordedLLM()
        llm.record(system=SYSTEM, user=USER, content="{}")
        with pytest.raises(CassetteMiss):
            await llm.complete(system=SYSTEM, user=USER + " Also check style.")

    async def test_replayed_responses_are_free(self) -> None:
        """Reporting the original call's price would inflate every eval run's
        cost with money nobody is spending."""
        llm = RecordedLLM()
        llm.record(system=SYSTEM, user=USER, content="{}", cost_usd=0.42)
        response = await llm.complete(system=SYSTEM, user=USER)
        assert response.cost_usd == 0.0

    async def test_token_counts_are_preserved(self) -> None:
        llm = RecordedLLM()
        llm.record(
            system=SYSTEM, user=USER, content="{}", tokens_in=120, tokens_out=30
        )
        response = await llm.complete(system=SYSTEM, user=USER)
        assert (response.tokens_in, response.tokens_out) == (120, 30)

    async def test_records_through_a_delegate_on_miss(self) -> None:
        class Live:
            calls = 0

            async def complete(self, **kwargs: Any) -> LLMResponse:
                Live.calls += 1
                return LLMResponse(
                    content='{"findings": []}',
                    tokens_in=10,
                    tokens_out=5,
                    cost_usd=0.01,
                    model="live",
                )

        llm = RecordedLLM(delegate=Live())
        await llm.complete(system=SYSTEM, user=USER)
        assert Live.calls == 1
        # Second call is served from the recording, not the delegate.
        again = await llm.complete(system=SYSTEM, user=USER)
        assert Live.calls == 1
        assert again.cost_usd == 0.0

    def test_round_trips_through_a_file(self, tmp_path: Path) -> None:
        llm = RecordedLLM()
        llm.record(system=SYSTEM, user=USER, content='{"findings": []}')
        path = tmp_path / "cassette.json"
        llm.to_file(path)

        reloaded = RecordedLLM.from_file(path)
        assert len(reloaded) == 1
        assert reloaded.keys() == (cassette_key(SYSTEM, USER),)

    def test_cassette_is_sorted_for_reviewable_diffs(self, tmp_path: Path) -> None:
        llm = RecordedLLM()
        for n in range(3):
            llm.record(system=SYSTEM, user=f"prompt {n}", content="{}")
        path = tmp_path / "cassette.json"
        llm.to_file(path)
        raw = path.read_text()
        assert raw.endswith("\n")
        keys = list(json.loads(raw)["entries"])
        assert keys == sorted(keys)

    def test_key_depends_on_both_halves_of_the_prompt(self) -> None:
        assert cassette_key("a", "b") != cassette_key("b", "a")
        assert cassette_key("a", "b") == cassette_key("a", "b")


def ollama_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestOllamaLLM:
    async def test_sends_system_and_user_turns(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": DEFAULT_MODEL,
                    "message": {"content": '{"findings": []}'},
                    "prompt_eval_count": 900,
                    "eval_count": 40,
                },
            )

        async with ollama_client(handler) as client:
            llm = OllamaLLM(client=client)
            response = await llm.complete(system=SYSTEM, user=USER)

        assert captured["messages"] == [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ]
        assert captured["stream"] is False
        assert response.content == '{"findings": []}'
        assert response.tokens_in == 900
        assert response.tokens_out == 40

    async def test_local_inference_is_free(self) -> None:
        """The point of this adapter. Zero is a fact here, not a placeholder."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"message": {"content": "{}"}, "model": "local"}
            )

        async with ollama_client(handler) as client:
            response = await OllamaLLM(client=client).complete(
                system=SYSTEM, user=USER
            )
        assert response.cost_usd == 0.0

    async def test_passes_the_schema_for_constrained_decoding(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "{}"}})

        schema = {"type": "object", "properties": {"findings": {"type": "array"}}}
        async with ollama_client(handler) as client:
            await OllamaLLM(client=client).complete(
                system=SYSTEM, user=USER, json_schema=schema
            )
        assert captured["format"] == schema

    async def test_omits_format_when_no_schema_is_given(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "{}"}})

        async with ollama_client(handler) as client:
            await OllamaLLM(client=client).complete(system=SYSTEM, user=USER)
        assert "format" not in captured

    async def test_temperature_and_token_cap_are_forwarded(self) -> None:
        """Local models still honour temperature -- the parameter is only
        removed on Anthropic's current models."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "{}"}})

        async with ollama_client(handler) as client:
            await OllamaLLM(client=client).complete(
                system=SYSTEM, user=USER, temperature=0.0, max_tokens=2048
            )
        assert captured["options"] == {"temperature": 0.0, "num_predict": 2048}

    async def test_missing_token_counts_report_zero_not_a_guess(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": {"content": "{}"}})

        async with ollama_client(handler) as client:
            response = await OllamaLLM(client=client).complete(
                system=SYSTEM, user=USER
            )
        assert response.tokens_in == 0
        assert response.tokens_out == 0

    async def test_unpulled_model_gives_an_actionable_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text='{"error":"model not found"}')

        async with ollama_client(handler) as client:
            with pytest.raises(OllamaUnavailable, match="ollama pull"):
                await OllamaLLM(client=client).complete(system=SYSTEM, user=USER)

    async def test_server_error_propagates(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        async with ollama_client(handler) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await OllamaLLM(client=client).complete(system=SYSTEM, user=USER)

    async def test_targets_the_chat_endpoint(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"message": {"content": "{}"}})

        async with ollama_client(handler) as client:
            await OllamaLLM(client=client, base_url="http://box:11434/").complete(
                system=SYSTEM, user=USER
            )
        assert seen == ["http://box:11434/api/chat"]
