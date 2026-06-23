from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod

import httpx

from .config import ProviderConfig
from .schema import CandidateResult, ProviderRequest, Usage


class ModelProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def chat(self, request: ProviderRequest) -> CandidateResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class ProviderClientPool:
    """Small shared AsyncClient pool keyed by timeout.

    Providers can safely share clients when the timeout settings match because base URLs,
    headers, and request bodies are still supplied per request.
    """

    def __init__(self) -> None:
        self._clients: dict[float, httpx.AsyncClient] = {}

    def get(self, timeout_seconds: float) -> httpx.AsyncClient:
        client = self._clients.get(timeout_seconds)
        if client is None:
            client = httpx.AsyncClient(timeout=timeout_seconds)
            self._clients[timeout_seconds] = client
        return client

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


class OpenAICompatibleProvider(ModelProvider):
    """Provider for APIs exposing POST /v1/chat/completions."""

    def __init__(
        self,
        config: ProviderConfig,
        client: httpx.AsyncClient | None = None,
        client_pool: ProviderClientPool | None = None,
    ):
        super().__init__(config)
        self._client = client
        self._client_pool = client_pool
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _client_for_request(self) -> httpx.AsyncClient:
        if self._client_pool is not None:
            return self._client_pool.get(self.config.timeout_seconds)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self._client

    async def chat(self, request: ProviderRequest) -> CandidateResult:
        started = time.perf_counter()
        url = f"{self.config.base_url}/chat/completions"
        headers = dict(self.config.headers)
        api_key = self.config.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {
            "model": self.config.model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "stream": False,
            **request.extra_body,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens

        try:
            response = await self._client_for_request().post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content", "")
            usage_raw = data.get("usage") or {}
            usage = Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
                total_tokens=int(usage_raw.get("total_tokens", 0) or 0),
            )
            return CandidateResult(
                provider=self.config.name,
                model=self.config.model,
                content=content,
                ok=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
                usage=usage,
            )
        except httpx.HTTPStatusError as exc:
            return CandidateResult(
                provider=self.config.name,
                model=self.config.model,
                content="",
                ok=False,
                error=self._http_status_error_message(exc, api_key),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - user needs provider error surfaced
            return CandidateResult(
                provider=self.config.name,
                model=self.config.model,
                content="",
                ok=False,
                error=str(exc),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

    @staticmethod
    def _http_status_error_message(exc: httpx.HTTPStatusError, api_key: str | None) -> str:
        snippet = exc.response.text.replace("\n", " ").strip()
        snippet = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", snippet)
        if api_key:
            snippet = snippet.replace(api_key, "[redacted]")
        if len(snippet) > 300:
            snippet = f"{snippet[:300]}..."
        status_code = exc.response.status_code
        if snippet:
            return f"HTTP {status_code} from provider: {snippet}"
        return f"HTTP {status_code} from provider"


class StaticProvider(ModelProvider):
    """Simple provider used by tests and examples."""

    def __init__(self, config: ProviderConfig, response: str):
        super().__init__(config)
        self.response = response
        self.last_request: ProviderRequest | None = None

    async def chat(self, request: ProviderRequest) -> CandidateResult:
        self.last_request = request
        return CandidateResult(
            provider=self.config.name,
            model=self.config.model,
            weight=self.config.weight,
            content=self.response,
            ok=True,
            latency_ms=0,
        )


def make_provider(
    config: ProviderConfig,
    client_pool: ProviderClientPool | None = None,
) -> ModelProvider:
    if config.type == "openai_compatible":
        return OpenAICompatibleProvider(config, client_pool=client_pool)
    raise ValueError(f"Unsupported provider type: {config.type}")
