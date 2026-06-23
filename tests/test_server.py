from __future__ import annotations

from fastapi.testclient import TestClient

from openfusion.config import AppConfig, FusionConfig, ProviderConfig
from openfusion.providers import StaticProvider
from openfusion.server import create_app


def test_health_and_models() -> None:
    config = AppConfig(
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                model="qwen",
            )
        ]
    )
    client = TestClient(create_app(config))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"ok": True, "providers": ["local"]}

    models = client.get("/v1/models")
    assert models.status_code == 200
    model_ids = [item["id"] for item in models.json()["data"]]
    assert "openfusion/panel-judge" in model_ids
    assert "openfusion/fallback" in model_ids
    assert "provider/local/qwen" in model_ids


def test_streaming_request_returns_sse_and_done() -> None:
    provider = StaticProvider(
        ProviderConfig(name="local", base_url="http://localhost:11434/v1", model="qwen"),
        "Streamed answer.",
    )
    config = AppConfig(
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                model="qwen",
            )
        ]
    )
    client = TestClient(create_app(config, providers={"local": provider}))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "provider/local/qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: " in response.text
    assert '"object": "chat.completion.chunk"' in response.text
    assert "Streamed answer." in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


def test_chat_completions_with_static_providers() -> None:
    providers = {
        "a": StaticProvider(
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            "Candidate answer.",
        ),
        "judge": StaticProvider(
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
            "Final answer.",
        ),
    }
    config = AppConfig(
        providers=[
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
        ],
        fusion=FusionConfig(panel=["a"], judge_provider="judge"),
    )
    client = TestClient(create_app(config, providers=providers))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openfusion/panel-judge",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Final answer."
    assert payload["openfusion"]["strategy"] == "panel_judge"


def test_direct_provider_model_routing() -> None:
    provider = StaticProvider(
        ProviderConfig(name="local", base_url="http://local", model="qwen"),
        "Direct answer.",
    )
    config = AppConfig(
        providers=[
            ProviderConfig(name="local", base_url="http://local", model="qwen"),
        ]
    )
    client = TestClient(create_app(config, providers={"local": provider}))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "provider/local/qwen",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Direct answer."
    assert payload["openfusion"]["strategy"] == "direct_provider"
    assert payload["openfusion"]["candidates"][0]["provider"] == "local"


def test_direct_provider_missing_or_disabled_returns_clear_error() -> None:
    config = AppConfig(
        providers=[
            ProviderConfig(name="disabled", base_url="http://disabled", model="qwen", enabled=False),
        ]
    )
    client = TestClient(create_app(config, providers={}))

    missing = client.post(
        "/v1/chat/completions",
        json={
            "model": "provider/missing/qwen",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Provider not found: missing"

    disabled = client.post(
        "/v1/chat/completions",
        json={
            "model": "provider/disabled/qwen",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert disabled.status_code == 400
    assert disabled.json()["detail"] == "Provider is disabled: disabled"


def test_openai_compatible_request_extra_fields_are_forwarded() -> None:
    provider = StaticProvider(
        ProviderConfig(name="local", base_url="http://local", model="qwen"),
        "Direct answer.",
    )
    config = AppConfig(
        providers=[
            ProviderConfig(name="local", base_url="http://local", model="qwen"),
        ]
    )
    client = TestClient(create_app(config, providers={"local": provider}))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "provider/local/qwen",
            "messages": [
                {
                    "role": "user",
                    "name": "tester",
                    "content": [{"type": "text", "text": "hello"}],
                    "tool_call_id": "call_123",
                }
            ],
            "top_p": 0.8,
            "stop": ["END"],
            "seed": 42,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "user": "user-123",
            "n": 1,
        },
    )

    assert response.status_code == 200
    assert provider.last_request is not None
    assert provider.last_request.extra_body == {
        "top_p": 0.8,
        "stop": ["END"],
        "seed": 42,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "user": "user-123",
        "n": 1,
    }
    assert provider.last_request.messages[0].name == "tester"
    assert provider.last_request.messages[0].tool_call_id == "call_123"
