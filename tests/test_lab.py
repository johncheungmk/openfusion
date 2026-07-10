from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openfusion.cli import app
from openfusion.config import AppConfig, FusionConfig, ProviderConfig
from openfusion.fusion import FusionEngine
from openfusion.lab import (
    LAB_RESULT_SCHEMA_VERSION,
    BaselineResultSummary,
    LabConfig,
    LabEngine,
    LabExample,
    LabExampleResult,
    LabExperiment,
    LabMetricSummary,
    LabModel,
    LabRecommendationSettings,
    LabResultCard,
    LabStrategy,
    StrategyResultSummary,
    _grade_lab_output,
    _latency_ms,
    _metrics,
    _panel_complementarity_report,
    _strategy_comparisons,
    build_engine_plan,
    export_result_card,
    lab_config_to_app_config,
    lab_config_to_yaml,
    load_lab_dataset,
    load_result_card,
    recommend_from_card,
    recommend_from_summaries,
    run_lab_experiment,
    search_huggingface_models,
)
from openfusion.metrics import wilson_interval
from openfusion.providers import ModelProvider
from openfusion.schema import CandidateResult, ChatMessage, FusionResult, ProviderRequest, Usage


class LabFakeProvider(ModelProvider):
    def __init__(
        self,
        config,
        *,
        answers: dict[str, str],
        latency_ms: int = 10,
        delay_seconds: float = 0.0,
    ):
        super().__init__(config)
        self.answers = answers
        self.latency_ms = latency_ms
        self.delay_seconds = delay_seconds
        self.requests: list[ProviderRequest] = []

    async def chat(self, request: ProviderRequest) -> CandidateResult:
        self.requests.append(request)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        prompt = "\n".join(str(message.content or "") for message in request.messages)
        content = "wrong"
        for marker, answer in self.answers.items():
            if marker in prompt:
                content = answer
                break
        return CandidateResult(
            provider=self.config.name,
            model=self.config.model,
            weight=self.config.weight,
            content=content,
            ok=True,
            latency_ms=self.latency_ms,
            usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


def test_lab_config_validation(tmp_path: Path) -> None:
    path = _write_lab_yaml(tmp_path)

    config = LabConfig.load(path)

    assert config.experiment.name == "unit-lab"
    assert [engine.name for engine in config.engines] == ["ollama-one", "vllm-one", "tgi-one"]
    assert [strategy.name for strategy in config.strategies] == ["fallback", "majority_vote"]


def test_invalid_lab_config_errors(tmp_path: Path) -> None:
    path = _write_lab_yaml(tmp_path, model_engine="missing")

    with pytest.raises(ValueError, match="unknown engines"):
        LabConfig.load(path)


def test_generated_provider_config_and_fusion_defaults(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path))

    app_config = lab_config_to_app_config(config)
    generated = AppConfig.model_validate(json.loads(json.dumps(app_config.model_dump(mode="json"))))

    assert [provider.name for provider in generated.providers] == ["local-a", "local-b"]
    assert generated.providers[0].base_url == "http://127.0.0.1:11434/v1"
    assert generated.providers[0].api_key_env is None
    assert generated.fusion.panel == ["local-a", "local-b"]
    assert generated.fusion.judge_provider == "local-a"
    assert generated.fusion.ranker_provider == "local-a"
    assert generated.fusion.fuser_provider == "local-a"
    assert generated.fusion.max_total_calls == 6
    assert generated.fusion.max_tokens == 64
    assert "providers:" in lab_config_to_yaml(config)


def test_lab_model_pricing_validation_and_provider_propagation(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path))
    priced_model = config.models[0].model_copy(
        update={
            "input_cost_per_million_tokens_usd": 0.5,
            "output_cost_per_million_tokens_usd": 1.5,
        }
    )
    config = config.model_copy(update={"models": [priced_model, *config.models[1:]]})

    provider = lab_config_to_app_config(config).providers[0]

    assert provider.input_cost_per_million_tokens_usd == pytest.approx(0.5)
    assert provider.output_cost_per_million_tokens_usd == pytest.approx(1.5)
    with pytest.raises(ValueError, match="zero or greater"):
        LabModel(
            provider_name="invalid",
            engine="ollama-one",
            model="model",
            input_cost_per_million_tokens_usd=-0.01,
        )


def test_lab_engine_rejects_credentials_in_base_url() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        LabEngine(
            name="unsafe",
            base_url="https://alice:supersecret@example.test/v1",
        )
    with pytest.raises(ValueError, match="query string or fragment"):
        LabEngine(
            name="unsafe",
            base_url="https://example.test/v1?api_key=supersecret",
        )


def test_lab_strategy_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="self_moa_sampels"):
        LabStrategy(name="self_moa", self_moa_sampels=3)


def test_v1_result_card_loads_with_v2_field_defaults(tmp_path: Path) -> None:
    result_path = tmp_path / "v1-results.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "openfusion-lab-result-v1",
                "openfusion_version": "0.5.2",
                "experiment": {"name": "legacy"},
                "timestamp": "2026-07-01T00:00:00+00:00",
                "platform": {},
                "python_version": "3.10",
                "dataset": {},
                "engines": [],
                "models": [],
                "strategies": [
                    {
                        "strategy": "fallback",
                        "metrics": {
                            "total_examples": 1,
                            "correct": 1,
                            "accuracy": 1.0,
                        },
                    }
                ],
                "recommendations": {},
            }
        ),
        encoding="utf-8",
    )

    card = load_result_card(result_path)

    assert card.schema_version == "openfusion-lab-result-v1"
    assert card.panel_complementarity is None
    assert card.strategies[0].metrics.accuracy_ci95_low == 0.0
    assert card.strategies[0].metrics.total_estimated_cost_usd is None

    exported_path = tmp_path / "v1-exported.json"
    export_result_card(result_path, exported_path)
    exported = json.loads(exported_path.read_text(encoding="utf-8"))
    assert exported["schema_version"] == "openfusion-lab-result-v1"
    assert "panel_complementarity" not in exported
    assert "accuracy_ci95_low" not in exported["strategies"][0]["metrics"]


def test_loading_prompt_style_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"id":"math","prompt":"2+2?","reference":"4"}\n', encoding="utf-8")

    examples = load_lab_dataset(dataset)

    assert examples[0].messages[0].content == "2+2?"
    assert examples[0].references == ["4"]


def test_loading_utf8_bom_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "bom.jsonl"
    dataset.write_text('\ufeff{"id":"math","prompt":"2+2?","reference":"4"}\n', encoding="utf-8")

    examples = load_lab_dataset(dataset)

    assert examples[0].id == "math"
    assert examples[0].references == ["4"]


def test_loading_chat_style_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        '{"id":"mcq","messages":[{"role":"user","content":"Pick A or B"}],'
        '"answer":"B","answer_regex":"\\\\b([AB])\\\\b"}\n',
        encoding="utf-8",
    )

    examples = load_lab_dataset(dataset)

    assert examples[0].messages[0].content == "Pick A or B"
    assert examples[0].references == ["B"]
    assert examples[0].answer_regex == r"\b([AB])\b"


@pytest.mark.asyncio
async def test_lab_run_with_fake_providers_metrics_and_result_card(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path))
    app_config = lab_config_to_app_config(config)
    providers = {
        provider.name: LabFakeProvider(
            provider,
            answers={"2+2": "4", "FIFO": "B", "3 * 4": "C"},
            latency_ms=10 if provider.name == "local-a" else 20,
            delay_seconds=0.01 if provider.name == "local-a" else 0.02,
        )
        for provider in app_config.providers
    }

    card = await run_lab_experiment(config, lab_path=tmp_path / "lab.yaml", providers=providers)

    fallback = next(summary for summary in card.strategies if summary.strategy == "fallback")
    majority = next(summary for summary in card.strategies if summary.strategy == "majority_vote")
    baseline_a = next(summary for summary in card.baselines if summary.name == "baseline/local-a")
    assert card.schema_version == LAB_RESULT_SCHEMA_VERSION
    assert [baseline.name for baseline in card.baselines] == ["baseline/local-a", "baseline/local-b"]
    assert baseline_a.provider == "local-a"
    assert baseline_a.model == "llama3.2:3b"
    assert baseline_a.metrics.correct == 3
    assert baseline_a.metrics.accuracy_ci95_low < baseline_a.metrics.accuracy
    assert baseline_a.metrics.accuracy_ci95_high == pytest.approx(1.0)
    assert card.panel_complementarity is not None
    assert card.panel_complementarity.oracle_accuracy == pytest.approx(1.0)
    assert card.panel_complementarity.oracle_gain_pp == pytest.approx(0.0)
    assert card.first_provider_baseline == baseline_a
    assert card.fallback_baseline == fallback
    assert card.best_single_model_by_accuracy in card.baselines
    assert card.best_single_model_by_latency == baseline_a
    assert card.best_single_model_baseline == card.best_single_model_by_accuracy
    assert fallback.metrics.total_examples == 3
    assert fallback.metrics.correct == 3
    assert fallback.metrics.total_calls == 3
    assert fallback.metrics.total_prompt_tokens == 6
    assert fallback.metrics.total_completion_tokens == 9
    assert fallback.metrics.total_tokens == 15
    assert 5 <= fallback.metrics.avg_latency_ms < 500
    assert fallback.metrics.p50_latency_ms >= 5
    assert fallback.metrics.p95_latency_ms < 500
    assert fallback.metrics.p99_latency_ms < 500
    assert fallback.metrics.accuracy_per_call == pytest.approx(1.0)
    assert fallback.metrics.accuracy_per_1k_tokens == pytest.approx(200.0)
    assert majority.metrics.total_calls > fallback.metrics.total_calls
    assert majority.metrics.avg_calls_per_example <= config.experiment.max_total_calls
    assert {comparison.strategy for comparison in card.strategy_comparisons} == {
        "fallback",
        "majority_vote",
    }
    majority_comparison = next(
        comparison for comparison in card.strategy_comparisons if comparison.strategy == "majority_vote"
    )
    assert majority_comparison.accuracy_delta_vs_fallback_pp == pytest.approx(0.0)
    assert majority_comparison.latency_ratio_vs_best_single is not None
    assert majority_comparison.calls_ratio_vs_best_single is not None
    assert majority_comparison.tokens_ratio_vs_best_single is not None
    assert card.recommendations.best_accuracy in {"fallback", "majority_vote"}
    assert card.recommendations.best_latency == "fallback"
    assert card.recommendations.best_efficiency == "fallback"
    assert card.recommendations.best_balanced == "fallback"
    assert card.recommendations.configured_objective == "balanced"
    assert card.recommendations.recommended_strategy == "fallback"
    assert card.recommendation_settings == config.recommendation
    assert all("base_url" not in engine for engine in card.engines)
    assert card.recommendations.by_objective["best_accuracy"] in {"fallback", "majority_vote"}
    assert card.recommendations.explanations_by_objective["best_accuracy"]
    assert card.dataset_hash
    assert card.config_hash


@pytest.mark.asyncio
async def test_lab_run_aggregates_configured_estimated_costs(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path))
    priced_models = [
        model.model_copy(
            update={
                "input_cost_per_million_tokens_usd": 1.0,
                "output_cost_per_million_tokens_usd": 2.0,
            }
        )
        for model in config.models
    ]
    config = config.model_copy(update={"models": priced_models})
    app_config = lab_config_to_app_config(config)
    providers = {
        provider.name: LabFakeProvider(
            provider,
            answers={"2+2": "4", "FIFO": "B", "3 * 4": "C"},
        )
        for provider in app_config.providers
    }

    card = await run_lab_experiment(
        config,
        lab_path=tmp_path / "lab.yaml",
        providers=providers,
    )

    fallback = next(summary for summary in card.strategies if summary.strategy == "fallback")
    assert fallback.metrics.priced_examples == 3
    assert fallback.metrics.total_estimated_cost_usd == pytest.approx(3 * 8e-6)
    assert fallback.metrics.avg_estimated_cost_usd == pytest.approx(8e-6)
    assert fallback.metrics.cost_per_correct_usd == pytest.approx(8e-6)
    assert all(baseline.metrics.priced_examples == 3 for baseline in card.baselines)


@pytest.mark.asyncio
async def test_lab_does_not_relabel_first_strategy_as_fallback(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path)).model_copy(
        update={"strategies": [LabStrategy(name="majority_vote")]}
    )
    app_config = lab_config_to_app_config(config)
    providers = {
        provider.name: LabFakeProvider(
            provider,
            answers={"2+2": "4", "FIFO": "B", "3 * 4": "C"},
        )
        for provider in app_config.providers
    }

    card = await run_lab_experiment(
        config,
        lab_path=tmp_path / "lab.yaml",
        providers=providers,
    )

    assert card.fallback_baseline is None
    assert card.strategies[0].baseline_strategy is None
    assert card.strategy_comparisons[0].accuracy_delta_vs_fallback_pp is None


@pytest.mark.asyncio
async def test_lab_best_single_baseline_selection_by_accuracy_and_latency(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path))
    app_config = lab_config_to_app_config(config)
    providers = {}
    for provider in app_config.providers:
        answers = (
            {"2+2": "4", "FIFO": "wrong", "3 * 4": "wrong"}
            if provider.name == "local-a"
            else {"2+2": "4", "FIFO": "B", "3 * 4": "C"}
        )
        providers[provider.name] = LabFakeProvider(
            provider,
            answers=answers,
            latency_ms=5 if provider.name == "local-a" else 25,
            delay_seconds=0.005 if provider.name == "local-a" else 0.025,
        )

    card = await run_lab_experiment(config, lab_path=tmp_path / "lab.yaml", providers=providers)

    assert card.best_single_model_by_accuracy
    assert card.best_single_model_by_accuracy.name == "baseline/local-b"
    assert card.best_single_model_by_accuracy.metrics.accuracy == pytest.approx(1.0)
    assert card.best_single_model_by_latency
    assert card.best_single_model_by_latency.name == "baseline/local-a"


def test_wilson_p50_p95_p99_and_cost_metrics() -> None:
    metrics = _metrics(
        [
            LabExampleResult(
                id="1", correct=True, latency_ms=10, estimated_cost_usd=0.01
            ),
            LabExampleResult(
                id="2", correct=True, latency_ms=20, estimated_cost_usd=0.02
            ),
            LabExampleResult(
                id="3", correct=True, latency_ms=30, estimated_cost_usd=0.03
            ),
            LabExampleResult(
                id="4", correct=False, latency_ms=40, estimated_cost_usd=0.04
            ),
        ],
        None,
    )
    expected_ci = wilson_interval(3, 4)

    assert metrics.accuracy_ci95_low == pytest.approx(expected_ci[0])
    assert metrics.accuracy_ci95_high == pytest.approx(expected_ci[1])
    assert metrics.p50_latency_ms == 25
    assert metrics.p95_latency_ms == 40
    assert metrics.p99_latency_ms == 40
    assert metrics.priced_examples == 4
    assert metrics.total_estimated_cost_usd == pytest.approx(0.1)
    assert metrics.avg_estimated_cost_usd == pytest.approx(0.025)
    assert metrics.cost_per_correct_usd == pytest.approx(0.1 / 3)

    partially_priced = _metrics(
        [
            LabExampleResult(id="1", correct=True, estimated_cost_usd=0.01),
            LabExampleResult(id="2", correct=False),
        ],
        None,
    )
    assert partially_priced.priced_examples == 1
    assert partially_priced.total_estimated_cost_usd is None
    assert partially_priced.avg_estimated_cost_usd is None
    assert partially_priced.cost_per_correct_usd is None


def test_efficiency_metrics_are_dataset_size_invariant() -> None:
    one_result = [
        LabExampleResult(id="1", correct=True, calls=1, total_tokens=10),
    ]
    repeated_results = [
        LabExampleResult(id=str(index), correct=True, calls=1, total_tokens=10)
        for index in range(10)
    ]

    one = _metrics(one_result, None)
    repeated = _metrics(repeated_results, None)

    assert one.accuracy_per_call == pytest.approx(1.0)
    assert repeated.accuracy_per_call == pytest.approx(one.accuracy_per_call)
    assert one.accuracy_per_1k_tokens == pytest.approx(100.0)
    assert repeated.accuracy_per_1k_tokens == pytest.approx(one.accuracy_per_1k_tokens)


def test_panel_complementarity_report() -> None:
    by_baseline = {
        "baseline/a": [
            LabExampleResult(id="1", correct=True),
            LabExampleResult(id="2", correct=True),
            LabExampleResult(id="3", correct=False),
            LabExampleResult(id="4", correct=False),
        ],
        "baseline/b": [
            LabExampleResult(id="1", correct=True),
            LabExampleResult(id="2", correct=False),
            LabExampleResult(id="3", correct=True),
            LabExampleResult(id="4", correct=False),
        ],
        "baseline/c": [
            LabExampleResult(id="1", correct=False),
            LabExampleResult(id="2", correct=False),
            LabExampleResult(id="3", correct=True),
            LabExampleResult(id="4", correct=False),
        ],
    }

    report = _panel_complementarity_report(by_baseline)

    assert report is not None
    assert report.providers == ["a", "b", "c"]
    assert report.oracle_accuracy == pytest.approx(0.75)
    assert (report.oracle_accuracy_ci95_low, report.oracle_accuracy_ci95_high) == pytest.approx(
        wilson_interval(3, 4)
    )
    assert report.best_single_provider == "a"
    assert report.best_single_accuracy == pytest.approx(0.5)
    assert report.oracle_gain_pp == pytest.approx(25.0)
    assert report.all_model_cofailure_rate == pytest.approx(0.25)
    assert (
        report.all_model_cofailure_rate_ci95_low,
        report.all_model_cofailure_rate_ci95_high,
    ) == pytest.approx(wilson_interval(1, 4))
    pair_ab = next(
        pair
        for pair in report.pairwise
        if (pair.provider_a, pair.provider_b) == ("a", "b")
    )
    assert pair_ab.correctness_disagreement_rate == pytest.approx(0.5)
    assert pair_ab.both_wrong_rate == pytest.approx(0.25)
    assert report.marginal_oracle_contribution == pytest.approx(
        {"a": 0.25, "b": 0.0, "c": 0.0}
    )


@pytest.mark.asyncio
async def test_parallel_strategy_latency_uses_wall_clock_not_trace_sum() -> None:
    provider_configs = [
        ProviderConfig(name=name, base_url=f"http://{name}", model=f"model-{name}")
        for name in ("a", "b")
    ]
    providers = {
        provider.name: LabFakeProvider(
            provider,
            answers={"question": "answer"},
            latency_ms=1000,
            delay_seconds=0.02,
        )
        for provider in provider_configs
    }
    engine = FusionEngine(
        AppConfig(
            providers=provider_configs,
            fusion=FusionConfig(
                panel=["a", "b"],
                judge_provider="a",
                max_parallel=2,
                max_total_calls=3,
            ),
        ),
        providers=providers,
    )
    started = time.perf_counter()
    result = await engine.run(
        [ChatMessage(role="user", content="question")],
        strategy="parallel_synthesis",
    )
    wall_latency_ms = _latency_ms(result, started)
    trace_latency_ms = sum(step.latency_ms or 0 for step in result.trace)

    assert trace_latency_ms == 3000
    assert 20 <= wall_latency_ms < 1000
    assert wall_latency_ms < trace_latency_ms
    await engine.aclose()


def test_recommendation_best_accuracy_latency_efficiency_balanced_and_warnings() -> None:
    summaries = [
        _summary("fallback", accuracy=0.75, latency=10, calls=4, failures=0),
        _summary("parallel_synthesis", accuracy=1.0, latency=50, calls=12, failures=0),
        _summary("uncertainty_cascade", accuracy=0.75, latency=12, calls=5, failures=0),
    ]

    recommendation = recommend_from_summaries(
        summaries,
        baseline_strategy="fallback",
        settings=LabRecommendationSettings(max_latency_ms=40),
    )

    assert recommendation.best_accuracy == "parallel_synthesis"
    assert recommendation.best_latency == "fallback"
    assert recommendation.best_efficiency == "fallback"
    assert recommendation.best_balanced in {"fallback", "uncertainty_cascade"}
    assert recommendation.configured_objective == "balanced"
    assert recommendation.recommended_strategy in {"fallback", "uncertainty_cascade"}
    assert any("slower than fallback" in warning for warning in recommendation.warnings)


def test_balanced_score_zero_beats_negative_score() -> None:
    zero = _summary("zero", accuracy=0.0, latency=10, calls=4)
    negative = _summary(
        "negative",
        accuracy=0.0,
        latency=30,
        calls=8,
        failures=1,
    )

    recommendation = recommend_from_summaries(
        [zero, negative],
        baseline_strategy="zero",
        settings=LabRecommendationSettings(),
    )

    assert zero.balanced_score == pytest.approx(0.0)
    assert negative.balanced_score is not None and negative.balanced_score < 0
    assert recommendation.best_balanced == "zero"
    assert recommendation.recommended_strategy is None


@pytest.mark.parametrize("max_latency_ms", [None, 200])
def test_primary_recommendation_excludes_all_failure_strategy(
    max_latency_ms: int | None,
) -> None:
    safe = _summary("fallback", accuracy=1.0, latency=100, calls=4)
    failed = _summary(
        "parallel_synthesis",
        accuracy=0.0,
        latency=1,
        calls=4,
        failures=4,
    )

    recommendation = recommend_from_summaries(
        [safe, failed],
        baseline_strategy="fallback",
        settings=LabRecommendationSettings(
            objective="latency",
            max_latency_ms=max_latency_ms,
        ),
    )

    assert recommendation.best_latency == "parallel_synthesis"
    assert recommendation.recommended_strategy == "fallback"


def test_failed_lab_result_cannot_match_reference_sentinel() -> None:
    result = FusionResult(
        strategy="fallback",
        final="No provider produced a usable answer.",
        ok=False,
        error="No provider produced a usable answer.",
    )
    example = LabExample(
        id="failure-sentinel",
        messages=[ChatMessage(role="user", content="fail")],
        references=["No provider produced a usable answer."],
    )

    assert _grade_lab_output("exact", result, example) is False


def test_result_card_recommendation_settings_round_trip(tmp_path: Path) -> None:
    settings = LabRecommendationSettings(
        objective="accuracy",
        max_latency_ms=40,
        prefer_lower_calls=False,
    )
    summaries = [
        _summary("fallback", accuracy=0.75, latency=10, calls=4),
        _summary("parallel_synthesis", accuracy=1.0, latency=50, calls=12),
    ]
    card = LabResultCard(
        experiment=LabExperiment(name="settings-round-trip"),
        timestamp="2026-07-10T00:00:00+00:00",
        platform={},
        python_version="3.13",
        dataset={},
        engines=[],
        models=[],
        strategies=summaries,
        recommendations=recommend_from_summaries(
            summaries,
            baseline_strategy="fallback",
            settings=settings,
        ),
        recommendation_settings=settings,
    )
    path = tmp_path / "card.json"
    path.write_text(card.model_dump_json(), encoding="utf-8")

    loaded = load_result_card(path)
    regenerated = recommend_from_card(loaded)

    assert loaded.recommendation_settings == settings
    assert regenerated.configured_objective == "accuracy"
    assert regenerated.recommended_strategy == "fallback"


def test_warning_when_fusion_does_not_beat_fallback() -> None:
    summaries = [
        _summary("fallback", accuracy=1.0, latency=10, calls=2),
        _summary("semantic_vote", accuracy=1.0, latency=20, calls=4),
    ]

    recommendation = recommend_from_summaries(
        summaries,
        baseline_strategy="fallback",
        settings=LabRecommendationSettings(),
    )

    assert "Fusion did not beat the fallback baseline on accuracy." in recommendation.warnings
    assert not any("semantic_vote is not recommended" in item for item in recommendation.explanations)


def test_strategy_comparison_delta_relative_ratio_and_divide_by_zero() -> None:
    fallback = _summary("fallback", accuracy=0.5, latency=10, calls=4, tokens=40)
    fusion = _summary("parallel_synthesis", accuracy=0.75, latency=25, calls=8, tokens=100)
    best_single = BaselineResultSummary(
        name="baseline/local-b",
        provider="local-b",
        model="qwen3:latest",
        metrics=LabMetricSummary(
            total_examples=4,
            correct=4,
            accuracy=1.0,
            total_calls=4,
            avg_calls_per_example=1.0,
            total_latency_ms=80,
            avg_latency_ms=20,
            total_tokens=50,
            accuracy_per_call=1.0,
            accuracy_per_1k_tokens=80,
        ),
    )

    comparisons = _strategy_comparisons(
        [fallback, fusion],
        fallback_baseline=fallback,
        best_single_model=best_single,
        settings=LabRecommendationSettings(),
    )
    fusion_comparison = next(item for item in comparisons if item.strategy == "parallel_synthesis")

    assert fusion_comparison.accuracy_delta_vs_fallback_pp == pytest.approx(25.0)
    assert fusion_comparison.accuracy_delta_vs_best_single_pp == pytest.approx(-25.0)
    assert fusion_comparison.accuracy_relative_vs_fallback_percent == pytest.approx(50.0)
    assert fusion_comparison.accuracy_relative_vs_best_single_percent == pytest.approx(-25.0)
    assert fusion_comparison.latency_ratio_vs_fallback == pytest.approx(2.5)
    assert fusion_comparison.latency_ratio_vs_best_single == pytest.approx(1.25)
    assert fusion_comparison.calls_ratio_vs_fallback == pytest.approx(2.0)
    assert fusion_comparison.calls_ratio_vs_best_single == pytest.approx(2.0)
    assert fusion_comparison.tokens_ratio_vs_fallback == pytest.approx(2.5)
    assert fusion_comparison.tokens_ratio_vs_best_single == pytest.approx(2.0)

    zero_fallback = _summary("fallback", accuracy=0.0, latency=0, calls=0, tokens=0)
    zero_comparison = _strategy_comparisons(
        [zero_fallback, fusion],
        fallback_baseline=zero_fallback,
        best_single_model=BaselineResultSummary(
            name="baseline/zero",
            provider="zero",
            model="zero",
            metrics=zero_fallback.metrics,
        ),
        settings=LabRecommendationSettings(),
    )[1]

    assert zero_comparison.accuracy_relative_vs_fallback_percent is None
    assert zero_comparison.latency_ratio_vs_fallback is None
    assert zero_comparison.calls_ratio_vs_fallback is None
    assert zero_comparison.tokens_ratio_vs_fallback is None


def test_recommendation_wording_is_objective_specific() -> None:
    fallback = _summary("fallback", accuracy=1.0, latency=20, calls=4)
    cascade = _summary("uncertainty_cascade", accuracy=0.75, latency=5, calls=4)
    best_single = BaselineResultSummary(
        name="baseline/local-a",
        provider="local-a",
        model="llama3.2:3b",
        metrics=fallback.metrics,
    )

    recommendation = recommend_from_summaries(
        [fallback, cascade],
        baseline_strategy="fallback",
        settings=LabRecommendationSettings(),
        baselines=[best_single],
        fallback_baseline=fallback,
        best_single_model=best_single,
    )

    assert recommendation.best_latency == "uncertainty_cascade"
    assert any(
        "uncertainty_cascade" in item and "not for accuracy improvement" in item
        for item in recommendation.explanations_by_objective["best_latency"]
    )
    assert "Fusion is not recommended for accuracy on this dataset." in recommendation.warnings
    assert not any(
        "uncertainty_cascade is not recommended" in item
        for item in recommendation.explanations
    )


@pytest.mark.asyncio
async def test_result_card_contains_no_secrets(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path, api_key_env="SECRET_ENV"))
    app_config = lab_config_to_app_config(config)
    providers = {
        provider.name: LabFakeProvider(provider, answers={"2+2": "4", "FIFO": "B", "3 * 4": "C"})
        for provider in app_config.providers
    }

    card = await run_lab_experiment(config, lab_path=tmp_path / "lab.yaml", providers=providers)
    payload = card.model_dump_json()

    assert "SECRET_ENV" not in payload
    assert "api_key" not in payload.lower()
    assert "Return only the answer" not in payload


def test_search_models_uses_mocked_httpx_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, Any]]:
            return [
                {
                    "modelId": "org/model",
                    "downloads": 10,
                    "likes": 2,
                    "lastModified": "2026-01-01",
                    "pipeline_tag": "text-generation",
                    "tags": ["license:apache-2.0"],
                }
            ]

    def fake_get(url: str, *, params: dict[str, Any], timeout: int) -> Response:
        calls.append({"url": url, "params": params, "timeout": timeout})
        return Response()

    monkeypatch.setattr("openfusion.lab.httpx.get", fake_get)

    results = search_huggingface_models(
        query="llama",
        limit=1,
        license="apache-2.0",
        sort="likes",
    )

    assert calls[0]["url"] == "https://huggingface.co/api/models"
    assert calls[0]["params"]["search"] == "llama"
    assert calls[0]["params"]["filter"] == "license:apache-2.0"
    assert results[0]["modelId"] == "org/model"
    assert results[0]["license"] == "apache-2.0"


def test_engine_plan_prints_ollama_vllm_tgi_guidance(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path))

    plan = build_engine_plan(config)

    assert "ollama pull llama3.2:3b" in plan
    assert "vllm.entrypoints.openai.api_server" in plan
    assert "docker run --gpus all" in plan
    assert "Start each engine manually" in plan


def test_lab_cli_engine_plan(tmp_path: Path) -> None:
    runner = CliRunner()
    path = _write_lab_yaml(tmp_path)

    result = runner.invoke(app, ["lab", "engine-plan", str(path)])

    assert result.exit_code == 0
    assert "ollama pull llama3.2:3b" in result.output


@pytest.mark.asyncio
async def test_lab_cli_recommend_prints_baseline_and_strategy_tables(tmp_path: Path) -> None:
    config = LabConfig.load(_write_lab_yaml(tmp_path))
    app_config = lab_config_to_app_config(config)
    providers = {
        provider.name: LabFakeProvider(
            provider,
            answers={"2+2": "4", "FIFO": "B", "3 * 4": "C"},
            latency_ms=10 if provider.name == "local-a" else 20,
        )
        for provider in app_config.providers
    }
    card = await run_lab_experiment(config, lab_path=tmp_path / "lab.yaml", providers=providers)
    results_path = tmp_path / "results.json"
    results_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(app, ["lab", "recommend", str(results_path)])

    assert result.exit_code == 0
    assert "Single-model baselines" in result.output
    assert "Strategy comparison" in result.output
    assert "best_accuracy" in result.output
    assert "baseline" in result.output


def test_docs_examples_load_successfully() -> None:
    for path in (
        Path("examples/minibench_local_10.jsonl"),
        Path("examples/open_ended_synthesis_10.jsonl"),
    ):
        examples = load_lab_dataset(path)
        assert len(examples) == 10
        assert all(example.id for example in examples)


def _summary(
    strategy: str,
    *,
    accuracy: float,
    latency: int,
    calls: int,
    tokens: int = 40,
    failures: int = 0,
) -> StrategyResultSummary:
    correct = int(accuracy * 4)
    return StrategyResultSummary(
        strategy=strategy,
        metrics=LabMetricSummary(
            total_examples=4,
            correct=correct,
            accuracy=accuracy,
            total_calls=calls,
            avg_calls_per_example=calls / 4,
            total_latency_ms=latency * 4,
            avg_latency_ms=latency,
            total_tokens=tokens,
            accuracy_per_call=correct / calls if calls else 0.0,
            accuracy_per_1k_tokens=correct / (tokens / 1000) if tokens else 0.0,
            failures=failures,
        ),
    )


def _write_lab_yaml(
    tmp_path: Path,
    *,
    model_engine: str = "ollama-one",
    api_key_env: str | None = None,
) -> Path:
    dataset = tmp_path / "minibench.jsonl"
    dataset.write_text(
        '{"id":"math","prompt":"Return only the answer: 2+2","reference":"4"}\n'
        '{"id":"fifo","messages":[{"role":"user","content":"FIFO structure? A Stack B Queue"}],'
        '"answer":"B","answer_regex":"\\\\b([AB])\\\\b"}\n'
        '{"id":"times","messages":[{"role":"user","content":"Answer only A, B, C, or D. 3 * 4? A 7 B 8 C 12 D 15"}],'
        '"answer":"C","answer_regex":"\\\\b([ABCD])\\\\b"}\n',
        encoding="utf-8",
    )
    api_key_line = f"    api_key_env: {api_key_env}\n" if api_key_env else ""
    path = tmp_path / "lab.yaml"
    path.write_text(
        f"""
experiment:
  name: unit-lab
  seed: 1
  max_examples: 3
  max_total_calls: 6
  max_tokens: 64
  temperature: 0.0
dataset:
  path: minibench.jsonl
  name: unit-minibench
  split: test
  answer_mode: exact_or_regex
engines:
  - name: ollama-one
    type: ollama
    base_url: http://127.0.0.1:11434/v1
    launch: manual
{api_key_line}  - name: vllm-one
    type: vllm
    base_url: http://127.0.0.1:8001/v1
    launch: manual
  - name: tgi-one
    type: tgi
    base_url: http://127.0.0.1:8080/v1
    launch: manual
models:
  - provider_name: local-a
    engine: {model_engine}
    model: llama3.2:3b
    weight: 1.0
    timeout_seconds: 300
  - provider_name: local-b
    engine: vllm-one
    model: gpt-oss:latest
    weight: 1.2
    timeout_seconds: 300
strategies:
  - name: fallback
  - name: majority_vote
recommendation:
  objective: balanced
  max_latency_ms: 120000
  prefer_lower_calls: true
""".strip(),
        encoding="utf-8",
    )
    return path
