from __future__ import annotations

import httpx
import pytest

from openfusion.config import ProviderConfig
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
