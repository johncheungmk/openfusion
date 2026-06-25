from __future__ import annotations

import pytest

from openfusion.config import AppConfig, FusionConfig, ProviderConfig
from openfusion.fusion import FusionEngine
from openfusion.providers import ModelProvider, StaticProvider
from openfusion.schema import CandidateResult, ChatMessage, ProviderRequest


class QueueProvider(ModelProvider):
    def __init__(self, config: ProviderConfig, responses: list[str]):
        super().__init__(config)
        self.responses = list(responses)
        self.requests: list[ProviderRequest] = []

    async def chat(self, request: ProviderRequest) -> CandidateResult:
        self.requests.append(request)
        response = self.responses.pop(0) if self.responses else ""
        return CandidateResult(
            provider=self.config.name,
            model=self.config.model,
            weight=self.config.weight,
            content=response,
            ok=bool(response),
            error=None if response else "no scripted response",
            latency_ms=0,
        )


def provider_config(name: str, weight: float = 1.0) -> ProviderConfig:
    return ProviderConfig(name=name, base_url=f"http://{name}", model=f"model-{name}", weight=weight)


@pytest.mark.asyncio
async def test_best_of_n_selects_candidate_without_rewriting() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Candidate A"),
        "b": StaticProvider(provider_config("b"), "Candidate B is correct"),
        "judge": StaticProvider(provider_config("judge"), '{"winner": 2, "reason": "more accurate"}'),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b"), provider_config("judge")],
        fusion=FusionConfig(panel=["a", "b"], judge_provider="judge"),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Pick the best")],
        strategy="best_of_n",
    )

    assert result.final == "Candidate B is correct"
    assert result.strategy == "best_of_n"
    assert result.judge_analysis == "more accurate"
    assert [step.stage for step in result.trace] == ["candidate", "candidate", "selection"]


@pytest.mark.asyncio
async def test_majority_and_weighted_vote_can_choose_different_groups() -> None:
    providers = {
        "a": StaticProvider(provider_config("a", 1.0), "Answer: blue"),
        "b": StaticProvider(provider_config("b", 1.0), "Answer: blue"),
        "c": StaticProvider(provider_config("c", 3.0), "Answer: red"),
    }
    config = AppConfig(
        providers=[provider_config("a", 1.0), provider_config("b", 1.0), provider_config("c", 3.0)],
        fusion=FusionConfig(panel=["a", "b", "c"]),
    )
    engine = FusionEngine(config, providers=providers)

    majority = await engine.run(
        [ChatMessage(role="user", content="Choose a color")],
        strategy="majority_vote",
    )
    weighted = await engine.run(
        [ChatMessage(role="user", content="Choose a color")],
        strategy="weighted_vote",
    )

    assert majority.final == "Answer: blue"
    assert weighted.final == "Answer: red"


@pytest.mark.asyncio
async def test_critique_revision_uses_distinct_roles() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Draft A"),
        "b": StaticProvider(provider_config("b"), "Draft B"),
        "critic": StaticProvider(provider_config("critic"), "Correct the unsupported date."),
        "reviser": StaticProvider(provider_config("reviser"), "Revised final answer."),
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(
            panel=["a", "b"],
            critic_provider="critic",
            reviser_provider="reviser",
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Write a careful answer")],
        strategy="critique_revision",
    )

    assert result.final == "Revised final answer."
    assert result.critic_provider == "critic"
    assert result.reviser_provider == "reviser"
    assert result.workflow_outputs["critique"] == "Correct the unsupported date."
    assert [step.stage for step in result.trace] == ["draft", "draft", "critique", "revision"]


@pytest.mark.asyncio
async def test_layered_refinement_runs_second_layer_and_synthesis() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Improved A"),
        "b": StaticProvider(provider_config("b"), "Improved B"),
        "judge": StaticProvider(provider_config("judge"), "Layered final."),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b"), provider_config("judge")],
        fusion=FusionConfig(panel=["a", "b"], judge_provider="judge", refinement_rounds=1),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Refine this solution")],
        strategy="layered_refinement",
    )

    assert result.final == "Layered final."
    assert len(result.candidates) == 4
    assert [candidate.stage for candidate in result.candidates] == [
        "layer_0",
        "layer_0",
        "layer_1",
        "layer_1",
    ]
    assert result.trace[-1].stage == "final_synthesis"


@pytest.mark.asyncio
async def test_adaptive_heuristic_selects_workflow_without_model_call() -> None:
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b")],
        fusion=FusionConfig(panel=["a", "b"], adaptive_use_model_planner=False),
    )
    engine = FusionEngine(
        config,
        providers={
            "a": StaticProvider(provider_config("a"), "A"),
            "b": StaticProvider(provider_config("b"), "B"),
        },
    )

    plan, trace = await engine.plan(
        [
            ChatMessage(
                role="user",
                content=(
                    "Compare two production RAG deployment architectures, analyze operational "
                    "risks, and recommend a plan with evidence and caveats."
                ),
            )
        ]
    )

    assert plan.strategy == "critique_revision"
    assert plan.source == "heuristic"
    assert trace == []


@pytest.mark.asyncio
async def test_model_planner_is_constrained_and_parsed() -> None:
    planner = StaticProvider(
        provider_config("planner"),
        '{"strategy":"best_of_n","panel":["a"],"samples_per_provider":2,'
        '"refinement_rounds":0,"rationale":"Use two independent attempts."}',
    )
    config = AppConfig(
        providers=[provider_config("a"), provider_config("planner")],
        fusion=FusionConfig(
            panel=["a"],
            planner_provider="planner",
            adaptive_use_model_planner=True,
            max_total_calls=5,
        ),
    )
    engine = FusionEngine(
        config,
        providers={"a": StaticProvider(provider_config("a"), "A"), "planner": planner},
    )

    plan, trace = await engine.plan(
        [ChatMessage(role="user", content="Solve this code problem")],
        use_model_planner=True,
    )

    assert plan.strategy == "best_of_n"
    assert plan.panel == ["a"]
    assert plan.samples_per_provider == 2
    assert plan.source == "model"
    assert trace[0].stage == "planning"


@pytest.mark.asyncio
async def test_call_budget_is_enforced_and_visible() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "A"),
        "b": StaticProvider(provider_config("b"), "Longer answer B"),
        "judge": StaticProvider(provider_config("judge"), "Should not run"),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b"), provider_config("judge")],
        fusion=FusionConfig(panel=["a", "b"], judge_provider="judge", max_total_calls=2),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Test")],
        strategy="parallel_synthesis",
        max_total_calls=2,
    )

    assert result.final == "Longer answer B"
    assert result.trace[-1].status == "skipped"
    assert "budget exhausted" in (result.trace[-1].note or "").lower()


@pytest.mark.asyncio
async def test_workflow_outputs_can_be_suppressed() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Draft"),
        "critic": StaticProvider(provider_config("critic"), "Private critique"),
        "reviser": StaticProvider(provider_config("reviser"), "Final"),
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(
            panel=["a"],
            critic_provider="critic",
            reviser_provider="reviser",
            include_workflow_outputs=False,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Test")],
        strategy="critique_revision",
    )

    assert result.final == "Final"
    assert result.workflow_outputs == {}
