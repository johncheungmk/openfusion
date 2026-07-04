from __future__ import annotations

from pathlib import Path

import pytest

from openfusion.config import AppConfig, FusionConfig, ProviderConfig
from openfusion.evaluation import (
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
    assert summary.metrics.total_latency_ms == 20
    assert summary.metrics.prompt_tokens == 4
    assert summary.metrics.completion_tokens == 6
    assert summary.metrics.total_tokens == 10
    assert summary.metrics.accuracy_per_call == 0.5
    assert summary.metrics.accuracy_per_1k_tokens == 100.0
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


def _dataset(tmp_path: Path) -> str:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"id":"math","prompt":"2+2?","reference":"4"}\n'
        '{"id":"mcq","prompt":"Answer: A or B","reference":"B","answer_regex":"Answer:\\\\s*([AB])"}\n',
        encoding="utf-8",
    )
    return str(path)
