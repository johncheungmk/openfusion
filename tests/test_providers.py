from __future__ import annotations

import json

import httpx
import pytest

from openfusion.config import AppConfig, FusionConfig, ProviderConfig
from openfusion.fusion import FusionEngine
from openfusion.providers import OpenAICompatibleProvider
from openfusion.schema import ChatMessage, ProviderRequest


@pytest.mark.asyncio
async def test_http_status_error_includes_status_snippet_without_secrets(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "secret-token")

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            401,
            text="bad key secret-token Authorization: Bearer secret-token",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            name="provider",
            base_url="https://example.test/v1",
            model="model",
            api_key_env="TEST_PROVIDER_API_KEY",
        ),
        client=client,
    )

    try:
        result = await provider.chat(
            ProviderRequest(messages=[ChatMessage(role="user", content="hello")])
        )
    finally:
        await client.aclose()

    assert not result.ok
    assert result.error is not None
    assert "HTTP 401 from provider" in result.error
    assert "secret-token" not in result.error
    assert "Bearer [redacted]" in result.error


@pytest.mark.asyncio
async def test_http_status_error_redacts_configured_header_secrets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            403,
            text="upstream rejected X-Api-Key header-secret-value",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            name="provider",
            base_url="https://example.test/v1",
            model="model",
            headers={"X-Api-Key": "header-secret-value"},
        ),
        client=client,
    )

    try:
        result = await provider.chat(
            ProviderRequest(messages=[ChatMessage(role="user", content="hello")])
        )
    finally:
        await client.aclose()

    assert not result.ok
    assert result.error is not None
    assert "HTTP 403 from provider" in result.error
    assert "header-secret-value" not in result.error
    assert "X-Api-Key [redacted]" in result.error


@pytest.mark.asyncio
async def test_timeout_has_clear_error_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            name="provider",
            base_url="https://example.test/v1",
            model="model",
            timeout_seconds=12,
        ),
        client=client,
    )
    try:
        result = await provider.chat(
            ProviderRequest(messages=[ChatMessage(role="user", content="hello")])
        )
    finally:
        await client.aclose()

    assert not result.ok
    assert result.error == "Provider timeout after 12s"


@pytest.mark.asyncio
async def test_extra_body_cannot_override_fixed_provider_request_fields() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            name="provider",
            base_url="https://example.test/v1",
            model="configured-model",
        ),
        client=client,
    )
    try:
        result = await provider.chat(
            ProviderRequest(
                messages=[ChatMessage(role="user", content="hello")],
                temperature=0.3,
                extra_body={
                    "model": "attacker-model",
                    "messages": [{"role": "user", "content": "override"}],
                    "temperature": 1.0,
                    "stream": True,
                    "top_p": 0.8,
                },
            )
        )
    finally:
        await client.aclose()

    assert result.ok
    assert captured["model"] == "configured-model"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["temperature"] == 0.3
    assert captured["stream"] is False
    assert captured["top_p"] == 0.8


@pytest.mark.asyncio
async def test_missing_upstream_usage_keeps_estimated_cost_incomplete() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider_config = ProviderConfig(
        name="provider",
        base_url="https://example.test/v1",
        model="model",
        input_cost_per_million_tokens_usd=1,
        output_cost_per_million_tokens_usd=2,
    )
    provider = OpenAICompatibleProvider(provider_config, client=client)
    engine = FusionEngine(
        AppConfig(
            providers=[provider_config],
            fusion=FusionConfig(panel=["provider"]),
        ),
        providers={"provider": provider},
    )
    try:
        result = await engine.run_provider(
            "provider",
            [ChatMessage(role="user", content="hello")],
        )
    finally:
        await client.aclose()

    assert result.final == "ok"
    assert result.candidates[0].metadata["usage_available"] is False
    assert result.estimated_cost_usd is None


@pytest.mark.asyncio
async def test_provider_reported_model_is_preserved_in_trace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200,
            json={
                "model": "resolved-model-2026-07-10",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider_config = ProviderConfig(
        name="provider",
        base_url="https://example.test/v1",
        model="stable-alias",
    )
    provider = OpenAICompatibleProvider(provider_config, client=client)
    engine = FusionEngine(
        AppConfig(
            providers=[provider_config],
            fusion=FusionConfig(panel=["provider"]),
        ),
        providers={"provider": provider},
    )
    try:
        result = await engine.run_provider(
            "provider",
            [ChatMessage(role="user", content="hello")],
        )
    finally:
        await client.aclose()

    assert result.candidates[0].model == "stable-alias"
    assert (
        result.candidates[0].metadata["provider_reported_model"]
        == "resolved-model-2026-07-10"
    )
    assert result.trace[0].provider_reported_model == "resolved-model-2026-07-10"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": None, "completion_tokens": 0, "total_tokens": 0},
        {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
    ],
)
async def test_invalid_usage_does_not_produce_complete_cost(
    usage: dict[str, int | None],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": usage,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider_config = ProviderConfig(
        name="provider",
        base_url="https://example.test/v1",
        model="model",
        input_cost_per_million_tokens_usd=1,
        output_cost_per_million_tokens_usd=2,
    )
    provider = OpenAICompatibleProvider(provider_config, client=client)
    engine = FusionEngine(
        AppConfig(
            providers=[provider_config],
            fusion=FusionConfig(panel=["provider"]),
        ),
        providers={"provider": provider},
    )
    try:
        result = await engine.run_provider(
            "provider",
            [ChatMessage(role="user", content="hello")],
        )
    finally:
        await client.aclose()

    assert result.candidates[0].metadata["usage_available"] is False
    assert result.candidates[0].usage.prompt_tokens >= 0
    assert result.candidates[0].usage.completion_tokens >= 0
    assert result.candidates[0].usage.total_tokens >= 0
    assert result.estimated_cost_usd is None
