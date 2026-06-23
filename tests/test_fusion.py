from __future__ import annotations

import pytest

from openfusion.config import AppConfig, FusionConfig, ProviderConfig
from openfusion.fusion import FusionEngine
from openfusion.providers import ModelProvider, StaticProvider
from openfusion.schema import CandidateResult, ChatMessage, ProviderRequest


class FailingProvider(ModelProvider):
    def __init__(self, config: ProviderConfig, error: str = "failed", content: str = ""):
        super().__init__(config)
        self.error = error
        self.content = content

    async def chat(self, request: ProviderRequest) -> CandidateResult:  # noqa: ARG002
        return CandidateResult(
            provider=self.config.name,
            model=self.config.model,
            weight=self.config.weight,
            content=self.content,
            ok=False,
            error=self.error,
            latency_ms=0,
        )


@pytest.mark.asyncio
async def test_panel_judge_uses_judge_provider() -> None:
    providers = {
        "a": StaticProvider(
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            "Candidate A says use local models.",
        ),
        "b": StaticProvider(
            ProviderConfig(name="b", base_url="http://b", model="model-b"),
            "Candidate B says use cloud models.",
        ),
        "judge": StaticProvider(
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
            "Final fused answer.",
        ),
    }
    config = AppConfig(
        providers=[
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            ProviderConfig(name="b", base_url="http://b", model="model-b"),
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
        ],
        fusion=FusionConfig(panel=["a", "b"], judge_provider="judge"),
    )
    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="How should I deploy RAG?")]
    )
    assert result.final == "Final fused answer."
    assert result.judge_provider == "judge"
    assert len(result.candidates) == 2


@pytest.mark.asyncio
async def test_fallback_returns_first_success() -> None:
    providers = {
        "a": StaticProvider(
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            "First answer.",
        ),
        "b": StaticProvider(
            ProviderConfig(name="b", base_url="http://b", model="model-b"),
            "Second answer.",
        ),
    }
    config = AppConfig(
        providers=[
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            ProviderConfig(name="b", base_url="http://b", model="model-b"),
        ],
        fusion=FusionConfig(panel=["a", "b"]),
    )
    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Test")], strategy="fallback"
    )
    assert result.final == "First answer."
    assert result.strategy == "fallback"


@pytest.mark.asyncio
async def test_include_candidate_outputs_false_redacts_all_paths() -> None:
    providers = {
        "a": StaticProvider(
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            "Secret candidate answer.",
        ),
        "judge": FailingProvider(
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
            "judge failed",
        ),
    }
    config = AppConfig(
        providers=[
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
        ],
        fusion=FusionConfig(
            panel=["a"],
            judge_provider="judge",
            include_candidate_outputs=False,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Test")]
    )

    assert result.final == "Secret candidate answer."
    assert len(result.candidates) == 1
    assert result.candidates[0].provider == "a"
    assert result.candidates[0].content == ""


@pytest.mark.asyncio
async def test_judge_failure_gracefully_returns_best_candidate() -> None:
    providers = {
        "short": StaticProvider(
            ProviderConfig(name="short", base_url="http://short", model="short-model"),
            "Short.",
        ),
        "long": StaticProvider(
            ProviderConfig(name="long", base_url="http://long", model="long-model"),
            "This is the longer fallback answer.",
        ),
        "judge": FailingProvider(
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
            "judge failed",
        ),
    }
    config = AppConfig(
        providers=[
            ProviderConfig(name="short", base_url="http://short", model="short-model"),
            ProviderConfig(name="long", base_url="http://long", model="long-model"),
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
        ],
        fusion=FusionConfig(panel=["short", "long"], judge_provider="judge"),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Test")]
    )

    assert result.final == "This is the longer fallback answer."
    assert result.judge_analysis == "Judge failed: judge failed"


@pytest.mark.asyncio
async def test_fallback_all_providers_fail() -> None:
    providers = {
        "a": FailingProvider(ProviderConfig(name="a", base_url="http://a", model="model-a"), "a down"),
        "b": FailingProvider(ProviderConfig(name="b", base_url="http://b", model="model-b"), "b down"),
    }
    config = AppConfig(
        providers=[
            ProviderConfig(name="a", base_url="http://a", model="model-a"),
            ProviderConfig(name="b", base_url="http://b", model="model-b"),
        ],
        fusion=FusionConfig(panel=["a", "b"]),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Test")], strategy="fallback"
    )

    assert result.final == "No provider produced a usable answer."
    assert [candidate.ok for candidate in result.candidates] == [False, False]


@pytest.mark.asyncio
async def test_judge_prompt_includes_context_weights_and_truncates_candidates() -> None:
    provider = StaticProvider(
        ProviderConfig(name="a", base_url="http://a", model="model-a", weight=2.5),
        "A" * 250,
    )
    judge = StaticProvider(
        ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
        "Final fused answer.",
    )
    config = AppConfig(
        providers=[
            ProviderConfig(name="a", base_url="http://a", model="model-a", weight=2.5),
            ProviderConfig(name="judge", base_url="http://judge", model="judge-model"),
        ],
        fusion=FusionConfig(
            panel=["a"],
            judge_provider="judge",
            judge_candidate_max_chars=10,
        ),
    )

    await FusionEngine(config, providers={"a": provider, "judge": judge}).run(
        [
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Latest question"),
        ]
    )

    assert judge.last_request is not None
    prompt = judge.last_request.messages[-1].content
    assert isinstance(prompt, str)
    assert "system: You are concise." in prompt
    assert "assistant: First answer" in prompt
    assert "Latest user request:" in prompt
    assert "weight=2.5" in prompt
    assert "[truncated" in prompt
