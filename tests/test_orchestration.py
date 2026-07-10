from __future__ import annotations

import json

import pytest

from openfusion import fusion as fusion_module
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


@pytest.mark.parametrize(
    "prompt",
    [
        fusion_module.PARALLEL_SYNTHESIS_SYSTEM_PROMPT,
        fusion_module.STRUCTURED_SYNTHESIS_SYSTEM_PROMPT,
        fusion_module.SELECTOR_SYSTEM_PROMPT,
        fusion_module.CRITIC_SYSTEM_PROMPT,
        fusion_module.REVISION_SYSTEM_PROMPT,
        fusion_module.REFINEMENT_SYSTEM_PROMPT,
        fusion_module.RANKER_SYSTEM_PROMPT,
        fusion_module.PAIRWISE_RANKER_SYSTEM_PROMPT,
        fusion_module.SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT,
    ],
)
def test_candidate_consuming_prompts_mark_content_untrusted(prompt: str) -> None:
    assert "untrusted data" in prompt
    assert "instructions" in prompt


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
    assert result.judge_analysis is None
    assert result.plan is not None
    assert result.plan.rationale == "Rationale suppressed by configuration."


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
async def test_pairwise_rank_fuse_parse_success() -> None:
    ranker = QueueProvider(
        provider_config("ranker"),
        ['{"winner": 2}', '{"winner": 1}', '{"winner": 1}'],
    )
    providers = {
        "a": StaticProvider(provider_config("a"), "Candidate A"),
        "b": StaticProvider(provider_config("b"), "Candidate B best"),
        "c": StaticProvider(provider_config("c"), "Candidate C"),
        "ranker": ranker,
        "judge": StaticProvider(provider_config("judge"), "Fused ranked final."),
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(
            panel=["a", "b", "c"],
            ranker_provider="ranker",
            judge_provider="judge",
            rank_top_k=2,
            pairwise_rank_max_pairs=3,
            pairwise_rank_mode="pairwise",
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Rank and fuse")],
        strategy="pairwise_rank_fuse",
    )

    assert result.final == "Fused ranked final."
    assert result.strategy == "pairwise_rank_fuse"
    assert len(ranker.requests) == 3
    ranking = json.loads(result.workflow_outputs["ranking"])
    assert ranking["parsed"] is True
    assert ranking["wins"][0] == {"candidate": 2, "wins": 2}
    assert result.trace[-2].stage == "ranking_summary"


@pytest.mark.asyncio
async def test_pairwise_rank_fuse_parse_failure_falls_back_to_candidate_order() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Candidate A"),
        "b": StaticProvider(provider_config("b"), "Candidate B"),
        "ranker": StaticProvider(provider_config("ranker"), "not parseable"),
        "judge": StaticProvider(provider_config("judge"), "Fused fallback final."),
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(
            panel=["a", "b"],
            ranker_provider="ranker",
            judge_provider="judge",
            rank_top_k=2,
            pairwise_rank_mode="score",
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Rank and fuse")],
        strategy="pairwise_rank_fuse",
    )

    assert result.final == "Fused fallback final."
    assert result.judge_analysis.startswith("Ranking parse failed; used candidate order.")
    ranking = json.loads(result.workflow_outputs["ranking"])
    assert ranking["parsed"] is False


@pytest.mark.asyncio
async def test_pairwise_rank_fuse_respects_max_pair_limit() -> None:
    ranker = QueueProvider(
        provider_config("ranker"),
        ['{"winner": 1}', '{"winner": 1}', '{"winner": 1}'],
    )
    providers = {
        "a": StaticProvider(provider_config("a"), "Candidate A"),
        "b": StaticProvider(provider_config("b"), "Candidate B"),
        "c": StaticProvider(provider_config("c"), "Candidate C"),
        "ranker": ranker,
        "judge": StaticProvider(provider_config("judge"), "Fused top."),
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(
            panel=["a", "b", "c"],
            ranker_provider="ranker",
            judge_provider="judge",
            pairwise_rank_mode="pairwise",
            pairwise_rank_max_pairs=1,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Rank")],
        strategy="pairwise_rank_fuse",
    )

    assert len(ranker.requests) == 1
    ranking = json.loads(result.workflow_outputs["ranking"])
    assert ranking["pairs_compared"] == 1
    assert ranking["max_pairs"] == 1


@pytest.mark.asyncio
async def test_semantic_vote_groups_equivalent_answers_with_llm() -> None:
    equivalence = QueueProvider(provider_config("equiv"), ['{"equivalent": true}', "yes"])
    providers = {
        "a": StaticProvider(provider_config("a"), "4"),
        "b": StaticProvider(provider_config("b"), "four"),
        "c": StaticProvider(provider_config("c"), "the answer is 4"),
        "equiv": equivalence,
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(
            panel=["a", "b", "c"],
            vote_equivalence_provider="equiv",
            semantic_vote_mode="llm_equivalence",
            semantic_vote_max_pairs=4,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="What is 2+2?")],
        strategy="semantic_vote",
    )

    assert result.final == "the answer is 4"
    assert len(equivalence.requests) == 2
    summary = json.loads(result.workflow_outputs["semantic_vote_summary"])
    assert summary["winning_group_size"] == 3
    assert summary["pairs_compared"] == 2


@pytest.mark.asyncio
async def test_semantic_vote_respects_max_total_calls() -> None:
    equivalence = QueueProvider(provider_config("equiv"), ['{"equivalent": true}'])
    providers = {
        "a": StaticProvider(provider_config("a"), "4"),
        "b": StaticProvider(provider_config("b"), "four"),
        "c": StaticProvider(provider_config("c"), "the answer is 4"),
        "equiv": equivalence,
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(
            panel=["a", "b", "c"],
            vote_equivalence_provider="equiv",
            semantic_vote_mode="llm_equivalence",
            max_total_calls=3,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="What is 2+2?")],
        strategy="semantic_vote",
        max_total_calls=3,
    )

    assert len(equivalence.requests) == 0
    assert len(result.candidates) == 3
    assert "pairs_compared" in (result.trace[-1].note or "")


@pytest.mark.asyncio
async def test_semantic_vote_falls_back_to_exact_voting_without_equivalence_provider() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Answer: yes"),
        "b": StaticProvider(provider_config("b"), "Answer: yes"),
        "c": StaticProvider(provider_config("c"), "Answer: no"),
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(panel=["a", "b", "c"], semantic_vote_mode="llm_equivalence"),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="yes or no")],
        strategy="semantic_vote",
    )

    assert result.final == "Answer: yes"
    summary = json.loads(result.workflow_outputs["semantic_vote_summary"])
    assert summary["mode"] == "rule_only"
    assert summary["groups"] == 2


@pytest.mark.asyncio
async def test_pairwise_rank_fuse_candidate_outputs_are_redacted() -> None:
    providers = {
        "a": StaticProvider(provider_config("a"), "Secret candidate."),
        "ranker": StaticProvider(provider_config("ranker"), "not parseable"),
        "judge": StaticProvider(provider_config("judge"), "Final."),
    }
    config = AppConfig(
        providers=[provider_config(name) for name in providers],
        fusion=FusionConfig(
            panel=["a"],
            ranker_provider="ranker",
            judge_provider="judge",
            include_candidate_outputs=False,
        ),
    )

    result = await FusionEngine(config, providers=providers).run(
        [ChatMessage(role="user", content="Rank")],
        strategy="pairwise_rank_fuse",
    )

    assert result.final == "Secret candidate."
    assert result.candidates[0].content == ""


@pytest.mark.asyncio
async def test_uncertainty_cascade_high_confidence_returns_cheap_answer() -> None:
    cheap = QueueProvider(provider_config("cheap"), ['{"answer":"Cheap answer","confidence":0.95}'])
    strong = QueueProvider(provider_config("strong"), ['{"answer":"Strong answer","confidence":0.99}'])
    config = AppConfig(
        providers=[provider_config("cheap"), provider_config("strong")],
        fusion=FusionConfig(
            panel=["cheap", "strong"],
            cascade_providers=["cheap", "strong"],
            cascade_confidence_threshold=0.8,
        ),
    )

    result = await FusionEngine(
        config,
        providers={"cheap": cheap, "strong": strong},
    ).run([ChatMessage(role="user", content="Simple question")], strategy="uncertainty_cascade")

    assert result.final == "Cheap answer"
    assert len(cheap.requests) == 1
    assert strong.requests == []
    assert result.trace[-1].stage == "cascade_decision"
    assert '"confidence": 0.95' in (result.trace[-1].note or "")
    assert '"escalation_reason": null' in (result.trace[-1].note or "")


@pytest.mark.asyncio
async def test_uncertainty_cascade_low_confidence_escalates() -> None:
    cheap = QueueProvider(provider_config("cheap"), ['{"answer":"Cheap answer","confidence":0.3}'])
    strong = QueueProvider(provider_config("strong"), ['{"answer":"Strong answer","confidence":0.9}'])
    config = AppConfig(
        providers=[provider_config("cheap"), provider_config("strong")],
        fusion=FusionConfig(
            panel=["cheap", "strong"],
            cascade_providers=["cheap", "strong"],
            cascade_confidence_threshold=0.8,
        ),
    )

    result = await FusionEngine(
        config,
        providers={"cheap": cheap, "strong": strong},
    ).run([ChatMessage(role="user", content="Simple question")], strategy="uncertainty_cascade")

    assert result.final == "Strong answer"
    assert len(cheap.requests) == 1
    assert len(strong.requests) == 1
    assert "low_confidence" in (result.trace[1].note or "")


@pytest.mark.asyncio
async def test_uncertainty_cascade_disagreement_escalates() -> None:
    cheap = QueueProvider(
        provider_config("cheap"),
        [
            '{"answer":"A","confidence":0.95}',
            '{"answer":"B","confidence":0.95}',
        ],
    )
    strong = QueueProvider(provider_config("strong"), ['{"answer":"Strong","confidence":0.9}'])
    config = AppConfig(
        providers=[provider_config("cheap"), provider_config("strong")],
        fusion=FusionConfig(
            panel=["cheap", "strong"],
            cascade_providers=["cheap", "strong"],
            cascade_confidence_threshold=0.8,
            cascade_consistency_samples=2,
            cascade_escalate_on_disagreement=True,
        ),
    )

    result = await FusionEngine(
        config,
        providers={"cheap": cheap, "strong": strong},
    ).run([ChatMessage(role="user", content="Simple question")], strategy="uncertainty_cascade")

    assert result.final == "Strong"
    assert len(cheap.requests) == 2
    assert "sample_disagreement" in (result.trace[2].note or "")


@pytest.mark.asyncio
async def test_uncertainty_cascade_respects_max_total_calls() -> None:
    cheap = QueueProvider(provider_config("cheap"), ['{"answer":"Cheap","confidence":0.2}'])
    strong = QueueProvider(provider_config("strong"), ['{"answer":"Strong","confidence":0.9}'])
    config = AppConfig(
        providers=[provider_config("cheap"), provider_config("strong")],
        fusion=FusionConfig(
            panel=["cheap", "strong"],
            cascade_providers=["cheap", "strong"],
            cascade_confidence_threshold=0.8,
            max_total_calls=1,
        ),
    )

    result = await FusionEngine(
        config,
        providers={"cheap": cheap, "strong": strong},
    ).run(
        [ChatMessage(role="user", content="Simple question")],
        strategy="uncertainty_cascade",
        max_total_calls=1,
    )

    assert result.final == "Cheap"
    assert len(cheap.requests) == 1
    assert strong.requests == []
    assert result.judge_analysis == "Cascade budget or provider list exhausted; returned best usable answer."


@pytest.mark.asyncio
async def test_adaptive_heuristic_can_select_uncertainty_cascade() -> None:
    config = AppConfig(
        providers=[provider_config("cheap"), provider_config("strong")],
        fusion=FusionConfig(
            panel=["cheap", "strong"],
            cascade_providers=["cheap", "strong"],
            adaptive_use_model_planner=False,
        ),
    )
    engine = FusionEngine(
        config,
        providers={
            "cheap": StaticProvider(provider_config("cheap"), "Cheap"),
            "strong": StaticProvider(provider_config("strong"), "Strong"),
        },
    )

    plan, trace = await engine.plan([ChatMessage(role="user", content="What is 2+2?")])

    assert plan.strategy == "uncertainty_cascade"
    assert plan.source == "heuristic"
    assert trace == []


@pytest.mark.asyncio
async def test_adaptive_heuristic_can_select_self_moa() -> None:
    config = AppConfig(
        providers=[provider_config("best")],
        fusion=FusionConfig(
            panel=["best"],
            self_moa_provider="best",
            self_moa_samples=2,
            max_total_calls=4,
            adaptive_use_model_planner=False,
        ),
    )
    engine = FusionEngine(config, providers={"best": StaticProvider(provider_config("best"), "A")})

    plan, trace = await engine.plan([ChatMessage(role="user", content="Solve this code bug")])

    assert plan.strategy == "self_moa"
    assert plan.samples_per_provider == 2
    assert trace == []


def test_invalid_cascade_provider_is_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="Cascade providers reference unknown providers: missing"):
        AppConfig(
            providers=[provider_config("cheap")],
            fusion=FusionConfig(panel=["cheap"], cascade_providers=["cheap", "missing"]),
        )


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
async def test_request_cannot_raise_configured_call_ceiling_or_schedule_unbounded_work() -> None:
    provider = QueueProvider(provider_config("a"), ["A", "B", "unused"])
    config = AppConfig(
        providers=[provider_config("a")],
        fusion=FusionConfig(panel=["a"], max_total_calls=2),
    )

    result = await FusionEngine(config, providers={"a": provider}).run(
        [ChatMessage(role="user", content="Test")],
        strategy="best_of_n",
        samples_per_provider=1_000_000_000,
        max_total_calls=1_000_000_000,
    )

    assert result.plan is not None
    assert result.plan.max_total_calls == 2
    assert len(provider.requests) == 2
    assert sum(step.model_call for step in result.trace) == 2


@pytest.mark.asyncio
async def test_tight_candidate_budget_schedules_providers_round_robin() -> None:
    provider_a = QueueProvider(provider_config("a"), ["A"])
    provider_b = QueueProvider(provider_config("b"), ["B"])
    config = AppConfig(
        providers=[provider_config("a"), provider_config("b")],
        fusion=FusionConfig(panel=["a", "b"], max_total_calls=2),
    )

    await FusionEngine(config, providers={"a": provider_a, "b": provider_b}).run(
        [ChatMessage(role="user", content="Test")],
        strategy="majority_vote",
        samples_per_provider=2,
    )

    assert len(provider_a.requests) == 1
    assert len(provider_b.requests) == 1


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
    assert result.judge_analysis is None
    assert result.plan is not None
    assert result.plan.rationale == "Rationale suppressed by configuration."
