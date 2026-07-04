from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openfusion.cli import app
from openfusion.config import AppConfig
from openfusion.lab import (
    LAB_RESULT_SCHEMA_VERSION,
    LabConfig,
    LabExampleResult,
    LabMetricSummary,
    LabRecommendationSettings,
    StrategyResultSummary,
    _metrics,
    build_engine_plan,
    lab_config_to_app_config,
    lab_config_to_yaml,
    load_lab_dataset,
    recommend_from_summaries,
    run_lab_experiment,
    search_huggingface_models,
)
from openfusion.providers import ModelProvider
from openfusion.schema import CandidateResult, ProviderRequest, Usage


class LabFakeProvider(ModelProvider):
    def __init__(self, config, *, answers: dict[str, str], latency_ms: int = 10):
        super().__init__(config)
        self.answers = answers
        self.latency_ms = latency_ms
        self.requests: list[ProviderRequest] = []

    async def chat(self, request: ProviderRequest) -> CandidateResult:
        self.requests.append(request)
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


def test_loading_prompt_style_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"id":"math","prompt":"2+2?","reference":"4"}\n', encoding="utf-8")

    examples = load_lab_dataset(dataset)

    assert examples[0].messages[0].content == "2+2?"
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
        )
        for provider in app_config.providers
    }

    card = await run_lab_experiment(config, lab_path=tmp_path / "lab.yaml", providers=providers)

    fallback = next(summary for summary in card.strategies if summary.strategy == "fallback")
    majority = next(summary for summary in card.strategies if summary.strategy == "majority_vote")
    assert card.schema_version == LAB_RESULT_SCHEMA_VERSION
    assert fallback.metrics.total_examples == 3
    assert fallback.metrics.correct == 3
    assert fallback.metrics.total_calls == 3
    assert fallback.metrics.total_prompt_tokens == 6
    assert fallback.metrics.total_completion_tokens == 9
    assert fallback.metrics.total_tokens == 15
    assert fallback.metrics.avg_latency_ms == 10
    assert fallback.metrics.p50_latency_ms == 10
    assert fallback.metrics.p95_latency_ms == 10
    assert fallback.metrics.accuracy_per_call == pytest.approx(1 / 3)
    assert fallback.metrics.accuracy_per_1k_tokens == pytest.approx(1000 / 15)
    assert majority.metrics.total_calls > fallback.metrics.total_calls
    assert majority.metrics.avg_calls_per_example <= config.experiment.max_total_calls
    assert card.recommendations.best_accuracy in {"fallback", "majority_vote"}
    assert card.recommendations.best_latency == "fallback"
    assert card.recommendations.best_efficiency == "fallback"
    assert card.recommendations.best_balanced == "fallback"
    assert card.dataset_hash
    assert card.config_hash


def test_p50_p95_latency_calculation() -> None:
    metrics = _metrics(
        [
            LabExampleResult(id="1", correct=True, latency_ms=10),
            LabExampleResult(id="2", correct=True, latency_ms=20),
            LabExampleResult(id="3", correct=True, latency_ms=30),
            LabExampleResult(id="4", correct=True, latency_ms=40),
        ],
        None,
    )

    assert metrics.p50_latency_ms == 25
    assert metrics.p95_latency_ms == 40


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
    assert any("slower than fallback" in warning for warning in recommendation.warnings)


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
    assert any("semantic_vote is not recommended" in item for item in recommendation.explanations)


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


def _summary(
    strategy: str,
    *,
    accuracy: float,
    latency: int,
    calls: int,
    failures: int = 0,
) -> StrategyResultSummary:
    return StrategyResultSummary(
        strategy=strategy,
        metrics=LabMetricSummary(
            total_examples=4,
            correct=int(accuracy * 4),
            accuracy=accuracy,
            total_calls=calls,
            avg_calls_per_example=calls / 4,
            total_latency_ms=latency * 4,
            avg_latency_ms=latency,
            total_tokens=40,
            accuracy_per_call=accuracy / calls,
            accuracy_per_1k_tokens=accuracy / 0.04,
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
