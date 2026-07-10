from __future__ import annotations

from pathlib import Path

import pytest

from openfusion.config import AppConfig, FusionConfig, ProviderConfig
from openfusion.evaluation import (
    EvalCase,
    EvalCaseResult,
    _metrics,
    compare_strategies,
    evaluate_cases,
    extract_answer,
    is_exact_match,
    load_jsonl,
    normalize_answer,
)
from openfusion.fusion import FusionEngine
from openfusion.providers import ModelProvider, StaticProvider
from openfusion.schema import CandidateResult, ProviderRequest, Usage


def test_answer_normalization_and_regex() -> None:
    assert normalize_answer(" **Paris!** ") == "paris"
    assert extract_answer("Reasoning\nFinal answer: B", r"Final answer:\s*(\w+)") == "B"
    assert is_exact_match("Reasoning\nFinal answer: B", ["B"], r"Final answer:\s*(\w+)")


def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"id":"one","prompt":"2+2?","reference":"4"}\n',
        encoding="utf-8",
    )
    cases = load_jsonl(path)
    assert cases[0].id == "one"
    assert cases[0].reference == "4"


def test_load_jsonl_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "eval-bom.jsonl"
    path.write_text(
        '\ufeff{"id":"one","prompt":"2+2?","reference":"4"}\n',
        encoding="utf-8",
    )

    assert load_jsonl(path)[0].id == "one"


class EvalQueueProvider(ModelProvider):
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
            content=response,
            ok=bool(response),
            error=None if response else "no scripted response",
            latency_ms=10,
            usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


class RaisingProvider(ModelProvider):
    async def chat(self, request: ProviderRequest) -> CandidateResult:  # noqa: ARG002
        raise RuntimeError("provider exploded")


def provider_config(name: str) -> ProviderConfig:
    return ProviderConfig(name=name, base_url=f"http://{name}", model=f"model-{name}")


@pytest.mark.asyncio
async def test_exact_match_report_includes_new_metrics(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    provider = EvalQueueProvider(provider_config("local"), ["4", "B"])
    config = AppConfig(
        providers=[provider_config("local")],
        fusion=FusionConfig(panel=["local"]),
    )

    summary = await evaluate_cases(
        FusionEngine(config, providers={"local": provider}),
        cases,
        strategy="fallback",
        max_total_calls=2,
    )

    assert summary.total == 2
    assert summary.correct == 2
    assert summary.metrics.total_examples == 2
    assert summary.metrics.accuracy == 1.0
    assert summary.metrics.total_calls == 2
    assert summary.metrics.avg_calls_per_example == 1.0
    assert summary.metrics.total_latency_ms >= 0
    assert summary.metrics.prompt_tokens == 4
    assert summary.metrics.completion_tokens == 6
    assert summary.metrics.total_tokens == 10
    assert summary.metrics.accuracy_ci95_low < summary.metrics.accuracy
    assert summary.metrics.accuracy_ci95_high == 1.0
    assert summary.metrics.accuracy_per_call == 1.0
    assert summary.metrics.accuracy_per_1k_tokens == 200.0
    assert summary.metrics.strategy_failures == 0


@pytest.mark.asyncio
async def test_equal_budget_compare_mode_enforces_max_total_calls(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    provider = EvalQueueProvider(provider_config("local"), ["4", "B", "4", "B", "4", "B"])
    config = AppConfig(
        providers=[provider_config("local")],
        fusion=FusionConfig(panel=["local"], self_moa_provider="local", self_moa_samples=1),
    )

    report = await compare_strategies(
        FusionEngine(config, providers={"local": provider}),
        cases,
        strategies=["fallback", "self_moa"],
        max_total_calls=1,
    )

    assert report.max_total_calls == 1
    assert [summary.strategy for summary in report.summaries] == ["fallback", "self_moa"]
    assert all(result.calls <= 1 for summary in report.summaries for result in summary.results)


@pytest.mark.asyncio
async def test_llm_pairwise_grader_works_with_fake_provider(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    local = EvalQueueProvider(provider_config("local"), ["4", "4"])
    grader = StaticProvider(provider_config("grader"), '{"winner":"A"}')
    config = AppConfig(
        providers=[provider_config("local"), provider_config("grader")],
        fusion=FusionConfig(panel=["local"]),
    )

    baseline = await evaluate_cases(
        FusionEngine(config, providers={"local": local, "grader": grader}),
        cases[:1],
        strategy="fallback",
    )
    local.responses.extend(["4"])
    summary = await evaluate_cases(
        FusionEngine(config, providers={"local": local, "grader": grader}),
        cases[:1],
        strategy="fallback",
        grader="llm_pairwise",
        grader_provider="grader",
        baseline_results={result.id: result for result in baseline.results},
    )

    assert summary.results[0].grade == "win"
    assert summary.metrics.win_rate_vs_baseline == 1.0


@pytest.mark.asyncio
async def test_swap_pairwise_grader_requires_order_consistency(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    local = EvalQueueProvider(provider_config("local"), ["4", "4"])
    grader = EvalQueueProvider(
        provider_config("grader"),
        ['{"winner":"A"}', '{"winner":"B"}'],
    )
    config = AppConfig(
        providers=[provider_config("local"), provider_config("grader")],
        fusion=FusionConfig(panel=["local"]),
    )
    engine = FusionEngine(config, providers={"local": local, "grader": grader})
    baseline = await evaluate_cases(engine, cases[:1], strategy="fallback")
    summary = await evaluate_cases(
        engine,
        cases[:1],
        strategy="fallback",
        grader="llm_pairwise_swap",
        grader_provider="grader",
        baseline_results={result.id: result for result in baseline.results},
    )

    result = summary.results[0]
    assert result.correct is True
    assert result.grade == "win"
    assert result.grader_calls == 2
    assert result.grader_verdicts == ["win", "win"]
    assert result.grader_order_consistent is True
    assert summary.metrics.grader_position_consistency_rate == 1.0
    assert summary.metrics.total_grader_calls == 2


@pytest.mark.asyncio
async def test_swap_pairwise_grader_abstains_on_position_inconsistency(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    local = EvalQueueProvider(provider_config("local"), ["4", "4"])
    grader = EvalQueueProvider(
        provider_config("grader"),
        ['{"winner":"A"}', '{"winner":"A"}'],
    )
    config = AppConfig(
        providers=[provider_config("local"), provider_config("grader")],
        fusion=FusionConfig(panel=["local"]),
    )
    engine = FusionEngine(config, providers={"local": local, "grader": grader})
    baseline = await evaluate_cases(engine, cases[:1], strategy="fallback")
    summary = await evaluate_cases(
        engine,
        cases[:1],
        strategy="fallback",
        grader="llm_pairwise_swap",
        grader_provider="grader",
        baseline_results={result.id: result for result in baseline.results},
    )

    result = summary.results[0]
    assert result.correct is False
    assert result.grade == "inconsistent"
    assert result.grader_verdicts == ["win", "loss"]
    assert result.grader_order_consistent is False
    assert summary.metrics.grader_inconsistencies == 1
    assert summary.metrics.grader_position_consistency_rate == 0.0


@pytest.mark.asyncio
async def test_grader_calls_share_the_per_case_call_budget(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    local = EvalQueueProvider(provider_config("local"), ["4", "4"])
    grader = EvalQueueProvider(
        provider_config("grader"),
        ['{"winner":"A"}', '{"winner":"B"}'],
    )
    config = AppConfig(
        providers=[provider_config("local"), provider_config("grader")],
        fusion=FusionConfig(panel=["local"], max_total_calls=2),
    )
    engine = FusionEngine(config, providers={"local": local, "grader": grader})
    baseline = await evaluate_cases(engine, cases[:1], strategy="fallback")
    summary = await evaluate_cases(
        engine,
        cases[:1],
        strategy="fallback",
        max_total_calls=2,
        grader="llm_pairwise_swap",
        grader_provider="local",
        baseline_results={result.id: result for result in baseline.results},
    )

    result = summary.results[0]
    assert result.grade == "abstain"
    assert result.grader_calls == 0
    assert result.grader_self_judged is False
    assert grader.requests == []
    assert summary.metrics.total_model_calls_including_grader == 1
    assert summary.metrics.grader_self_judged_examples == 0


@pytest.mark.asyncio
async def test_malformed_pairwise_grade_is_abstention_not_tie(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    local = EvalQueueProvider(provider_config("local"), ["4", "4"])
    grader = StaticProvider(provider_config("grader"), "not valid JSON")
    config = AppConfig(
        providers=[provider_config("local"), provider_config("grader")],
        fusion=FusionConfig(panel=["local"]),
    )
    engine = FusionEngine(config, providers={"local": local, "grader": grader})
    baseline = await evaluate_cases(engine, cases[:1], strategy="fallback")
    summary = await evaluate_cases(
        engine,
        cases[:1],
        strategy="fallback",
        grader="llm_pairwise",
        grader_provider="grader",
        baseline_results={result.id: result for result in baseline.results},
    )

    assert summary.results[0].correct is False
    assert summary.results[0].grade == "abstain"
    assert summary.metrics.grader_abstentions == 1
    assert summary.metrics.tie_rate_vs_baseline == 0.0


@pytest.mark.asyncio
async def test_self_judge_disclosure_includes_workflow_synthesizer(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    local = EvalQueueProvider(provider_config("local"), ["4", "draft"])
    grader = EvalQueueProvider(
        provider_config("grader"),
        ["fused answer", '{"winner":"A"}'],
    )
    config = AppConfig(
        providers=[provider_config("local"), provider_config("grader")],
        fusion=FusionConfig(panel=["local"], judge_provider="grader"),
    )
    engine = FusionEngine(config, providers={"local": local, "grader": grader})
    baseline = await evaluate_cases(engine, cases[:1], strategy="fallback")
    summary = await evaluate_cases(
        engine,
        cases[:1],
        strategy="parallel_synthesis",
        grader="llm_pairwise",
        grader_provider="grader",
        baseline_results={result.id: result for result in baseline.results},
    )

    assert summary.results[0].grader_self_judged is True
    assert summary.metrics.grader_self_judged_examples == 1


@pytest.mark.asyncio
async def test_configured_pricing_produces_cost_complete_metrics(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    priced = ProviderConfig(
        name="local",
        base_url="http://local",
        model="model-local",
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
    )
    provider = EvalQueueProvider(priced, ["4", "B"])
    config = AppConfig(
        providers=[priced],
        fusion=FusionConfig(panel=["local"]),
    )

    summary = await evaluate_cases(
        FusionEngine(config, providers={"local": provider}),
        cases,
        strategy="fallback",
    )

    assert summary.metrics.priced_examples == 2
    assert summary.metrics.total_estimated_cost_usd == pytest.approx(0.000016)
    assert summary.metrics.avg_estimated_cost_usd == pytest.approx(0.000008)
    assert summary.metrics.cost_per_correct_usd == pytest.approx(0.000008)


@pytest.mark.asyncio
async def test_handled_workflow_failure_is_counted_in_metrics(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    provider = EvalQueueProvider(provider_config("local"), [])
    config = AppConfig(
        providers=[provider_config("local")],
        fusion=FusionConfig(panel=["local"]),
    )

    summary = await evaluate_cases(
        FusionEngine(config, providers={"local": provider}),
        cases[:1],
        strategy="fallback",
    )

    assert summary.results[0].error == "No provider produced a usable answer."
    assert summary.results[0].correct is False
    assert summary.metrics.strategy_failures == 1
    assert summary.metrics.total_failed_model_calls == 1


@pytest.mark.asyncio
async def test_failed_workflow_sentinel_cannot_match_reference() -> None:
    provider = EvalQueueProvider(provider_config("local"), [])
    config = AppConfig(
        providers=[provider_config("local")],
        fusion=FusionConfig(panel=["local"]),
    )
    cases = [
        EvalCase(
            id="failure-sentinel",
            prompt="fail",
            reference="No provider produced a usable answer.",
        )
    ]

    summary = await evaluate_cases(
        FusionEngine(config, providers={"local": provider}),
        cases,
        strategy="fallback",
    )

    assert summary.correct == 0
    assert summary.results[0].grade == "incorrect"
    assert summary.results[0].grader_error == (
        "Generation failed: No provider produced a usable answer."
    )


@pytest.mark.asyncio
async def test_raising_provider_preserves_attempted_call_telemetry() -> None:
    config_value = provider_config("raising")
    engine = FusionEngine(
        AppConfig(
            providers=[config_value],
            fusion=FusionConfig(panel=["raising"]),
        ),
        providers={"raising": RaisingProvider(config_value)},
    )

    summary = await evaluate_cases(
        engine,
        [EvalCase(id="raising", prompt="fail", reference="anything")],
        strategy="fallback",
    )

    assert summary.results[0].calls == 1
    assert summary.results[0].failed_model_calls == 1
    assert summary.metrics.total_calls == 1
    assert summary.metrics.total_failed_model_calls == 1
    assert summary.metrics.strategy_failures == 1


@pytest.mark.asyncio
async def test_llm_rubric_grader_works_with_fake_provider(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    local = EvalQueueProvider(provider_config("local"), ["not exact"])
    grader = StaticProvider(provider_config("grader"), '{"grade":"correct"}')
    config = AppConfig(
        providers=[provider_config("local"), provider_config("grader")],
        fusion=FusionConfig(panel=["local"]),
    )

    summary = await evaluate_cases(
        FusionEngine(config, providers={"local": local, "grader": grader}),
        cases[:1],
        strategy="fallback",
        grader="llm_rubric",
        grader_provider="grader",
    )

    assert summary.correct == 1
    assert summary.results[0].grader == "llm_rubric"


@pytest.mark.asyncio
async def test_llm_rubric_without_provider_abstains(tmp_path: Path) -> None:
    cases = load_jsonl(_dataset(tmp_path))
    local = EvalQueueProvider(provider_config("local"), ["4"])
    config = AppConfig(
        providers=[provider_config("local")],
        fusion=FusionConfig(panel=["local"]),
    )

    summary = await evaluate_cases(
        FusionEngine(config, providers={"local": local}),
        cases[:1],
        strategy="fallback",
        grader="llm_rubric",
    )

    assert summary.correct == 0
    assert summary.results[0].grade == "abstain"
    assert summary.results[0].grader_calls == 0
    assert summary.results[0].grader_error == "Rubric grading requires a grader provider."


@pytest.mark.asyncio
async def test_grading_error_preserves_generation_telemetry() -> None:
    local = EvalQueueProvider(provider_config("local"), ["4"])
    config = AppConfig(
        providers=[provider_config("local")],
        fusion=FusionConfig(panel=["local"]),
    )

    summary = await evaluate_cases(
        FusionEngine(config, providers={"local": local}),
        [EvalCase(id="invalid-regex", prompt="2+2?", reference="4", answer_regex="(")],
        strategy="fallback",
        grader="regex",
    )

    result = summary.results[0]
    assert result.output == "4"
    assert result.calls == 1
    assert result.total_tokens == 5
    assert result.grade == "abstain"
    assert result.grader_error is not None
    assert result.grader_error.startswith("Grading failed with")


def test_json_output_writes_valid_schema(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    cases = load_jsonl(_dataset(tmp_path))
    assert cases
    payload = {
        "strategy": "fallback",
        "total": 1,
        "correct": 1,
        "accuracy": 1.0,
        "results": [],
        "metrics": {"total_examples": 1, "accuracy": 1.0},
    }
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    loaded = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["total_examples"] == 1


def test_pairwise_rates_are_conditional_on_decided_examples() -> None:
    metrics = _metrics(
        [
            EvalCaseResult(
                id="win",
                correct=True,
                output="a",
                references=["a"],
                strategy="test",
                grade="win",
            ),
            EvalCaseResult(
                id="abstain",
                correct=False,
                output="b",
                references=["a"],
                strategy="test",
                grade="abstain",
            ),
        ],
        accuracy=0.5,
        has_baseline=True,
    )

    assert metrics.decided_pairwise_examples == 1
    assert metrics.pairwise_decision_rate == 0.5
    assert metrics.win_rate_vs_baseline == 1.0
    assert metrics.grader_abstention_rate == 0.5


def _dataset(tmp_path: Path) -> str:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"id":"math","prompt":"2+2?","reference":"4"}\n'
        '{"id":"mcq","prompt":"Answer: A or B","reference":"B","answer_regex":"Answer:\\\\s*([AB])"}\n',
        encoding="utf-8",
    )
    return str(path)
