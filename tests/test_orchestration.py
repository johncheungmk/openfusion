from __future__ import annotations

import pytest

from openfusion.config import AppConfig, FusionConfig, PanelRoleConfig, ProviderConfig
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
async def test_panel_roles_are_applied_and_appear_in_trace() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Draft A"),
        "b": StaticProvider(provider_config("b"), "Draft B"),
        "judge": StaticProvider(provider_config("judge"), "Final answer."),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b"), provider_config("judge")],
        fusion=FusionConfig(
            panel=["a", "b"],
            judge_provider="judge",
            panel_roles=[
                PanelRoleConfig(
                    name="factual_checker",
                    instruction="Focus on factual accuracy and cite uncertainty.",
                ),
                PanelRoleConfig(
                    name="edge_case_reviewer",
                    instruction="Focus on edge cases and missing assumptions.",
                ),
            ],
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Review this plan")],
        strategy="parallel_synthesis",
    )

    assert [candidate.metadata["role_name"] for candidate in result.candidates] == [
        "factual_checker",
        "edge_case_reviewer",
    ]
    assert [step.role_name for step in result.trace[:2]] == [
        "factual_checker",
        "edge_case_reviewer",
    ]
    provider_request = providers["a"].last_request
    assert provider_request is not None
    role_message = "\n".join(str(message.content) for message in provider_request.messages)
    assert "OpenFusion panel role: factual_checker" in role_message
    assert "Focus on factual accuracy" in role_message


@pytest.mark.asyncio
async def test_panel_roles_cycle_when_fewer_roles_than_calls() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Draft A"),
        "b": StaticProvider(provider_config("b"), "Draft B"),
        "judge": StaticProvider(provider_config("judge"), "Final answer."),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b"), provider_config("judge")],
        fusion=FusionConfig(
            panel=["a", "b"],
            judge_provider="judge",
            samples_per_provider=2,
            panel_roles=[
                PanelRoleConfig(name="factual_checker", instruction="Check facts."),
                PanelRoleConfig(name="concise_summarizer", instruction="Be concise."),
            ],
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Draft an answer")],
        strategy="parallel_synthesis",
    )

    assert [candidate.metadata["role_name"] for candidate in result.candidates] == [
        "factual_checker",
        "concise_summarizer",
        "factual_checker",
        "concise_summarizer",
    ]


@pytest.mark.asyncio
async def test_structured_synthesis_parse_success() -> None:
    structured = (
        '{"consensus_points":["Both drafts agree."],'
        '"contradictions":["No contradiction."],'
        '"unique_insights":["Draft B adds rollout risk."],'
        '"missing_information":["Budget is unknown."],'
        '"final_answer":"Use a staged rollout."}'
    )
    providers = {
        "a": StaticProvider(provider_config("a"), "Draft A"),
        "b": StaticProvider(provider_config("b"), "Draft B"),
        "judge": StaticProvider(provider_config("judge"), structured),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b"), provider_config("judge")],
        fusion=FusionConfig(
            panel=["a", "b"],
            judge_provider="judge",
            structured_synthesis=True,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Synthesize")],
        strategy="parallel_synthesis",
    )

    assert result.final == "Use a staged rollout."
    assert result.workflow_outputs["consensus_points"] == "Both drafts agree."
    assert result.workflow_outputs["contradictions"] == "No contradiction."
    assert result.workflow_outputs["unique_insights"] == "Draft B adds rollout risk."
    assert result.workflow_outputs["missing_information"] == "Budget is unknown."
    assert result.workflow_outputs["final_answer"] == "Use a staged rollout."
    assert result.judge_analysis == "Structured synthesis parsed 2 independent candidate(s)."


@pytest.mark.asyncio
async def test_structured_synthesis_parse_failure_falls_back_to_plain_answer() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Draft A"),
        "b": StaticProvider(provider_config("b"), "Draft B"),
        "judge": StaticProvider(provider_config("judge"), "Plain fused answer."),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b"), provider_config("judge")],
        fusion=FusionConfig(
            panel=["a", "b"],
            judge_provider="judge",
            structured_synthesis=True,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Synthesize")],
        strategy="parallel_synthesis",
    )

    assert result.final == "Plain fused answer."
    assert result.workflow_outputs == {}
    assert result.judge_analysis == "Structured synthesis parsing failed; returned plain synthesis."


@pytest.mark.asyncio
async def test_structured_workflow_outputs_can_be_suppressed() -> None:
    structured = (
        '{"consensus_points":["Shared point."],'
        '"contradictions":[],"unique_insights":[],"missing_information":[],'
        '"final_answer":"Public final."}'
    )
    providers = {
        "a": StaticProvider(provider_config("a"), "Draft A"),
        "judge": StaticProvider(provider_config("judge"), structured),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("judge")],
        fusion=FusionConfig(
            panel=["a"],
            judge_provider="judge",
            structured_synthesis=True,
            include_workflow_outputs=False,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Synthesize")],
        strategy="parallel_synthesis",
    )

    assert result.final == "Public final."
    assert result.workflow_outputs == {}


@pytest.mark.asyncio
async def test_non_structured_synthesis_still_returns_plain_judge_answer() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Draft A"),
        "judge": StaticProvider(provider_config("judge"), "Plain final answer."),
    }
    config = AppConfig(
        providers=[provider_config("a"), provider_config("judge")],
        fusion=FusionConfig(panel=["a"], judge_provider="judge", structured_synthesis=False),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Synthesize")],
        strategy="parallel_synthesis",
    )

    assert result.final == "Plain final answer."
    assert result.workflow_outputs == {}
    assert result.judge_analysis == "Synthesized 1 independent candidate(s)."


@pytest.mark.asyncio
async def test_self_moa_select_mode_uses_one_provider_and_selects_candidate() -> None:
    proposer = QueueProvider(
        provider_config("self"),
        ["Short draft.", "Longer correct draft.", '{"winner": 2, "reason": "more complete"}'],
    )
    config = AppConfig(
        providers=[provider_config("self")],
        fusion=FusionConfig(
            panel=["self"],
            self_moa_provider="self",
            self_moa_samples=2,
            self_moa_mode="select",
        ),
    )

    result = await FusionEngine(config, providers={"self": proposer}).run(
        [ChatMessage(role="user", content="Answer carefully")],
        strategy="self_moa",
    )

    assert result.final == "Longer correct draft."
    assert result.strategy == "self_moa"
    assert result.judge_provider == "self"
    assert result.judge_analysis == "more complete"
    assert [request.temperature for request in proposer.requests[:2]] == [0.7, 0.7]
    assert [step.stage for step in result.trace] == [
        "self_moa_sample",
        "self_moa_sample",
        "self_moa_selection",
        "self_moa_summary",
    ]
    assert "samples=2" in (result.trace[-1].note or "")
    assert "mode=select" in (result.trace[-1].note or "")


@pytest.mark.asyncio
async def test_self_moa_synthesize_mode_writes_new_final() -> None:
    proposer = QueueProvider(provider_config("self"), ["Draft A.", "Draft B.", "Fused final."])
    config = AppConfig(
        providers=[provider_config("self")],
        fusion=FusionConfig(
            panel=["self"],
            self_moa_provider="self",
            self_moa_samples=2,
            self_moa_mode="synthesize",
        ),
    )

    result = await FusionEngine(config, providers={"self": proposer}).run(
        [ChatMessage(role="user", content="Fuse these ideas")],
        strategy="self_moa",
    )

    assert result.final == "Fused final."
    assert result.judge_analysis == "Synthesized 2 Self-MoA sample(s)."
    assert result.trace[-2].stage == "self_moa_synthesis"


@pytest.mark.asyncio
async def test_self_moa_respects_max_total_calls() -> None:
    proposer = QueueProvider(provider_config("self"), ["A", "Longer B", "Should not be called"])
    config = AppConfig(
        providers=[provider_config("self")],
        fusion=FusionConfig(
            panel=["self"],
            self_moa_provider="self",
            self_moa_samples=3,
            max_total_calls=2,
        ),
    )

    result = await FusionEngine(config, providers={"self": proposer}).run(
        [ChatMessage(role="user", content="Budget test")],
        strategy="self_moa",
        max_total_calls=2,
    )

    assert result.final == "Longer B"
    assert len(proposer.requests) == 2
    assert result.trace[-2].status == "skipped"
    assert "call_count=2/2" in (result.trace[-1].note or "")


@pytest.mark.asyncio
async def test_self_moa_uses_judge_then_panel_provider_default() -> None:
    judge = QueueProvider(provider_config("judge"), ["Judge sample."])
    panel = QueueProvider(provider_config("panel"), ["Panel sample."])
    config = AppConfig(
        providers=[provider_config("panel"), provider_config("judge")],
        fusion=FusionConfig(panel=["panel"], judge_provider="judge", self_moa_samples=1),
    )

    result = await FusionEngine(
        config,
        providers={"panel": panel, "judge": judge},
    ).run([ChatMessage(role="user", content="Default provider")], strategy="self_moa")

    assert result.final == "Judge sample."
    assert result.candidates[0].provider == "judge"
    assert len(judge.requests) == 1
    assert panel.requests == []


@pytest.mark.asyncio
async def test_self_moa_seq_respects_max_total_calls() -> None:
    proposer = QueueProvider(provider_config("self"), ["A", "B", "Running fused answer."])
    config = AppConfig(
        providers=[provider_config("self")],
        fusion=FusionConfig(
            panel=["self"],
            self_moa_provider="self",
            self_moa_samples=5,
            self_moa_batch_size=2,
            max_total_calls=3,
        ),
    )

    result = await FusionEngine(config, providers={"self": proposer}).run(
        [ChatMessage(role="user", content="Sequential budget test")],
        strategy="self_moa_seq",
        max_total_calls=3,
    )

    assert result.final == "Running fused answer."
    assert len(proposer.requests) == 3
    assert len(result.candidates) == 2
    assert "call_count=3/3" in (result.trace[-1].note or "")


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
