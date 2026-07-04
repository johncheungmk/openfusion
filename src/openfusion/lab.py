from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import __version__
from .config import AppConfig, FusionConfig, ProviderConfig
from .evaluation import is_exact_match
from .fusion import FusionEngine, canonical_strategy
from .providers import ModelProvider
from .schema import ChatMessage, FusionResult


LAB_RESULT_SCHEMA_VERSION = "openfusion-lab-result-v1"


class LabExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    seed: int = 0
    max_examples: int | None = None
    max_total_calls: int = 6
    max_tokens: int | None = 256
    temperature: float = 0.2

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("experiment name must not be empty")
        return value

    @field_validator("max_examples", "max_total_calls", "max_tokens")
    @classmethod
    def require_positive_optional_int(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("must be at least 1")
        return value


class LabEngine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = "openai_compatible"
    base_url: str
    launch: Literal["manual"] = "manual"
    api_key_env: str | None = None

    @field_validator("name", "type", "base_url")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def trim_slash(cls, value: str) -> str:
        return value.rstrip("/")


class LabModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    engine: str
    model: str
    weight: float = 1.0
    timeout_seconds: float = 300
    api_key_env: str | None = None

    @field_validator("provider_name", "engine", "model")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("weight", "timeout_seconds")
    @classmethod
    def require_positive_number(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value


class LabStrategy(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    self_moa_provider: str | None = None
    self_moa_samples: int | None = None
    self_moa_mode: Literal["select", "synthesize"] | None = None

    @field_validator("name")
    @classmethod
    def validate_strategy_name(cls, value: str) -> str:
        return canonical_strategy(value)

    @field_validator("self_moa_samples")
    @classmethod
    def require_positive_samples(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("self_moa_samples must be at least 1")
        return value


class LabDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    split: str | None = None
    answer_mode: Literal["exact", "regex", "exact_or_regex"] = "exact_or_regex"

    @field_validator("path", "name")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class LabRecommendationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: Literal["accuracy", "latency", "efficiency", "balanced"] = "balanced"
    max_latency_ms: int | None = None
    prefer_lower_calls: bool = True

    @field_validator("max_latency_ms")
    @classmethod
    def require_positive_latency(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_latency_ms must be at least 1")
        return value


class LabMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_examples: int = 0
    correct: int = 0
    accuracy: float = 0.0
    total_calls: int = 0
    avg_calls_per_example: float = 0.0
    total_latency_ms: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    accuracy_per_call: float = 0.0
    accuracy_per_1k_tokens: float = 0.0
    failures: int = 0
    win_rate_vs_baseline: float | None = None
    tie_rate_vs_baseline: float | None = None
    loss_rate_vs_baseline: float | None = None


class StrategyResultSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    metrics: LabMetricSummary
    baseline_strategy: str | None = None
    balanced_score: float | None = None
    notes: list[str] = Field(default_factory=list)


class LabRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_accuracy: str | None = None
    best_latency: str | None = None
    best_efficiency: str | None = None
    best_balanced: str | None = None
    explanations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LabResultCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LAB_RESULT_SCHEMA_VERSION
    openfusion_version: str = __version__
    experiment: LabExperiment
    timestamp: str
    platform: dict[str, str]
    python_version: str
    dataset: dict[str, Any]
    engines: list[dict[str, Any]]
    models: list[dict[str, Any]]
    strategies: list[StrategyResultSummary]
    recommendations: LabRecommendation
    warnings: list[str] = Field(default_factory=list)
    config_hash: str | None = None
    dataset_hash: str | None = None
    samples: list[dict[str, Any]] | None = None


class LabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: LabExperiment
    dataset: LabDataset
    engines: list[LabEngine]
    models: list[LabModel]
    strategies: list[LabStrategy]
    recommendation: LabRecommendationSettings = Field(default_factory=LabRecommendationSettings)

    @classmethod
    def load(cls, path: str | Path = "lab.yaml") -> "LabConfig":
        return load_lab_config(path)

    @model_validator(mode="after")
    def validate_references(self) -> "LabConfig":
        engine_names = [engine.name for engine in self.engines]
        duplicate_engines = sorted(
            {name for name in engine_names if engine_names.count(name) > 1}
        )
        if duplicate_engines:
            raise ValueError(f"Duplicate engine names: {', '.join(duplicate_engines)}")

        provider_names = [model.provider_name for model in self.models]
        duplicate_providers = sorted(
            {name for name in provider_names if provider_names.count(name) > 1}
        )
        if duplicate_providers:
            raise ValueError(f"Duplicate provider names: {', '.join(duplicate_providers)}")

        known_engines = set(engine_names)
        missing_engines = sorted(
            {model.engine for model in self.models if model.engine not in known_engines}
        )
        if missing_engines:
            raise ValueError(f"Models reference unknown engines: {', '.join(missing_engines)}")

        known_providers = set(provider_names)
        missing_self_moa = sorted(
            {
                strategy.self_moa_provider
                for strategy in self.strategies
                if strategy.self_moa_provider and strategy.self_moa_provider not in known_providers
            }
        )
        if missing_self_moa:
            raise ValueError(
                f"Strategies reference unknown self_moa providers: {', '.join(missing_self_moa)}"
            )

        if not self.engines:
            raise ValueError("At least one engine is required")
        if not self.models:
            raise ValueError("At least one model is required")
        if not self.strategies:
            raise ValueError("At least one strategy is required")
        return self


class LabExample(BaseModel):
    id: str
    messages: list[ChatMessage]
    references: list[str]
    answer_regex: str | None = None
    subject: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LabExampleResult(BaseModel):
    id: str
    correct: bool
    output: str = ""
    error: str | None = None
    calls: int = 0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def load_lab_config(path: str | Path = "lab.yaml") -> LabConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Lab config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Lab config must contain a YAML object: {config_path}")
    return LabConfig.model_validate(raw)


def lab_config_to_app_config(config: LabConfig) -> AppConfig:
    engines_by_name = {engine.name: engine for engine in config.engines}
    provider_configs: list[ProviderConfig] = []
    for model in config.models:
        engine = engines_by_name[model.engine]
        provider_configs.append(
            ProviderConfig(
                name=model.provider_name,
                type="openai_compatible",
                enabled=True,
                base_url=engine.base_url,
                api_key_env=model.api_key_env or engine.api_key_env,
                model=model.model,
                timeout_seconds=model.timeout_seconds,
                weight=model.weight,
                headers={},
            )
        )

    provider_names = [provider.name for provider in provider_configs]
    first_provider = provider_names[0]
    strategy_names = [strategy.name for strategy in config.strategies]
    self_moa_provider = next(
        (strategy.self_moa_provider for strategy in config.strategies if strategy.self_moa_provider),
        first_provider,
    )
    self_moa_samples = next(
        (strategy.self_moa_samples for strategy in config.strategies if strategy.self_moa_samples),
        3,
    )
    self_moa_mode = next(
        (strategy.self_moa_mode for strategy in config.strategies if strategy.self_moa_mode),
        "synthesize",
    )

    fusion = FusionConfig(
        default_strategy=strategy_names[0],
        panel=provider_names,
        judge_provider=first_provider,
        critic_provider=first_provider,
        reviser_provider=first_provider,
        planner_provider=first_provider,
        self_moa_provider=self_moa_provider,
        ranker_provider=first_provider,
        fuser_provider=first_provider,
        vote_equivalence_provider=None,
        cascade_providers=provider_names,
        max_parallel=min(4, max(1, len(provider_names))),
        max_total_calls=config.experiment.max_total_calls,
        samples_per_provider=1,
        refinement_rounds=1,
        self_moa_samples=self_moa_samples,
        self_moa_mode=self_moa_mode,
        max_tokens=config.experiment.max_tokens,
        temperature=config.experiment.temperature,
        rank_top_k=min(3, max(1, len(provider_names))),
        pairwise_rank_max_pairs=12,
        semantic_vote_max_pairs=12,
        cascade_max_steps=max(1, len(provider_names)),
        include_candidate_outputs=True,
        include_workflow_outputs=True,
    )
    return AppConfig(providers=provider_configs, fusion=fusion)


def lab_config_to_yaml(config: LabConfig) -> str:
    app_config = lab_config_to_app_config(config)
    return yaml.safe_dump(app_config.model_dump(mode="json"), sort_keys=False)


def write_generated_config(config: LabConfig, out: str | Path) -> None:
    Path(out).write_text(lab_config_to_yaml(config), encoding="utf-8")


def load_lab_dataset(path: str | Path, *, max_examples: int | None = None, seed: int = 0) -> list[LabExample]:
    dataset_path = Path(path)
    examples: list[LabExample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
            examples.append(_parse_lab_example(raw))
        except Exception as exc:  # noqa: BLE001 - line context helps repair datasets
            raise ValueError(f"Invalid lab JSONL at line {line_number}: {exc}") from exc
    if max_examples is not None and len(examples) > max_examples:
        rng = random.Random(seed)
        selected = list(examples)
        rng.shuffle(selected)
        return selected[:max_examples]
    return examples


async def run_lab_experiment(
    config: LabConfig,
    *,
    lab_path: str | Path | None = None,
    providers: dict[str, ModelProvider] | None = None,
    include_samples: bool = False,
) -> LabResultCard:
    app_config = lab_config_to_app_config(config)
    dataset_path = _resolve_dataset_path(config.dataset.path, lab_path)
    examples = load_lab_dataset(
        dataset_path,
        max_examples=config.experiment.max_examples,
        seed=config.experiment.seed,
    )
    engine = FusionEngine(app_config, providers=providers)
    try:
        by_strategy: dict[str, list[LabExampleResult]] = {}
        for strategy in config.strategies:
            by_strategy[strategy.name] = await _run_strategy(engine, config, strategy, examples)
    finally:
        await engine.aclose()

    baseline_strategy = "fallback" if "fallback" in by_strategy else next(iter(by_strategy), None)
    strategy_summaries = _summarize_strategies(by_strategy, baseline_strategy)
    recommendations = recommend_from_summaries(
        strategy_summaries,
        baseline_strategy=baseline_strategy,
        settings=config.recommendation,
    )
    warnings = list(recommendations.warnings)

    return LabResultCard(
        experiment=config.experiment,
        timestamp=datetime.now(timezone.utc).isoformat(),
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        python_version=sys.version.split()[0],
        dataset={
            "name": config.dataset.name,
            "split": config.dataset.split,
            "path": config.dataset.path,
            "answer_mode": config.dataset.answer_mode,
            "total_examples": len(examples),
        },
        engines=[_safe_engine_metadata(engine) for engine in config.engines],
        models=[_safe_model_metadata(model) for model in config.models],
        strategies=strategy_summaries,
        recommendations=recommendations,
        warnings=warnings,
        config_hash=_sha256_text(lab_config_to_yaml(config)),
        dataset_hash=_sha256_file(dataset_path),
        samples=_sample_results(by_strategy) if include_samples else None,
    )


def run_lab_experiment_sync(
    config: LabConfig,
    *,
    lab_path: str | Path | None = None,
    providers: dict[str, ModelProvider] | None = None,
    include_samples: bool = False,
) -> LabResultCard:
    return asyncio.run(
        run_lab_experiment(
            config,
            lab_path=lab_path,
            providers=providers,
            include_samples=include_samples,
        )
    )


def load_result_card(path: str | Path) -> LabResultCard:
    return LabResultCard.model_validate_json(Path(path).read_text(encoding="utf-8"))


def export_result_card(path: str | Path, out: str | Path) -> None:
    card = load_result_card(path)
    Path(out).write_text(card.model_dump_json(indent=2), encoding="utf-8")


def recommend_from_card(card: LabResultCard) -> LabRecommendation:
    return recommend_from_summaries(
        card.strategies,
        baseline_strategy=_baseline_strategy(card.strategies),
        settings=LabRecommendationSettings(),
    )


def recommend_from_summaries(
    summaries: list[StrategyResultSummary],
    *,
    baseline_strategy: str | None,
    settings: LabRecommendationSettings,
) -> LabRecommendation:
    if not summaries:
        return LabRecommendation(
            explanations=[
                "These recommendations apply only to this dataset, model set, hardware, and call budget."
            ]
        )

    baseline = next((summary for summary in summaries if summary.strategy == baseline_strategy), None)
    scored = [
        summary.model_copy(
            update={
                "balanced_score": _balanced_score(
                    summary.metrics,
                    baseline.metrics if baseline else None,
                    settings,
                )
            }
        )
        for summary in summaries
    ]
    for index, summary in enumerate(scored):
        summaries[index].balanced_score = summary.balanced_score

    best_accuracy = max(
        scored,
        key=lambda item: (
            item.metrics.accuracy,
            -item.metrics.failures,
            -item.metrics.avg_latency_ms,
            -item.metrics.avg_calls_per_example,
        ),
    )
    best_latency = min(
        scored,
        key=lambda item: (
            item.metrics.avg_latency_ms if item.metrics.total_examples else float("inf"),
            -item.metrics.accuracy,
        ),
    )
    best_efficiency = max(
        scored,
        key=lambda item: (
            item.metrics.accuracy_per_call,
            item.metrics.accuracy_per_1k_tokens,
            item.metrics.accuracy,
        ),
    )
    best_balanced = max(scored, key=lambda item: item.balanced_score or float("-inf"))

    explanations = [
        _accuracy_explanation(best_accuracy, baseline),
        f"{best_efficiency.strategy} gave the best accuracy per call.",
        f"{best_balanced.strategy} had the best balanced score under the configured penalties.",
        "These recommendations apply only to this dataset, model set, hardware, and call budget.",
    ]
    if best_latency.strategy == "uncertainty_cascade":
        explanations.append("uncertainty_cascade is recommended when latency/calls matter.")
    elif best_latency.strategy:
        explanations.append(f"{best_latency.strategy} had the lowest average latency.")

    warnings: list[str] = []
    if baseline:
        non_baseline = [summary for summary in scored if summary.strategy != baseline.strategy]
        if non_baseline and max(item.metrics.accuracy for item in non_baseline) <= baseline.metrics.accuracy:
            warnings.append("Fusion did not beat the fallback baseline on accuracy.")
            for summary in non_baseline:
                if summary.metrics.accuracy <= baseline.metrics.accuracy:
                    explanations.append(
                        f"{summary.strategy} is not recommended on this dataset because it did not "
                        f"improve over {baseline.strategy}."
                    )
        if baseline.metrics.avg_latency_ms > 0:
            ratio = best_accuracy.metrics.avg_latency_ms / baseline.metrics.avg_latency_ms
            if ratio >= 3.0:
                warnings.append(
                    f"{best_accuracy.strategy} was {ratio:.1f}x slower than {baseline.strategy}."
                )
    if (
        settings.max_latency_ms is not None
        and best_accuracy.metrics.avg_latency_ms > settings.max_latency_ms
    ):
        warnings.append(
            f"{best_accuracy.strategy} exceeded max_latency_ms={settings.max_latency_ms}."
        )

    return LabRecommendation(
        best_accuracy=best_accuracy.strategy,
        best_latency=best_latency.strategy,
        best_efficiency=best_efficiency.strategy,
        best_balanced=best_balanced.strategy,
        explanations=list(dict.fromkeys(explanations)),
        warnings=warnings,
    )


def search_huggingface_models(
    *,
    query: str | None = None,
    limit: int = 10,
    license: str | None = None,
    sort: Literal["downloads", "likes", "lastModified"] = "downloads",
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit, "sort": sort, "direction": -1}
    if query:
        params["search"] = query
    if license:
        params["filter"] = f"license:{license}"
    response = httpx.get("https://huggingface.co/api/models", params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected Hugging Face API response")
    return [
        {
            "modelId": item.get("modelId") or item.get("id"),
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "lastModified": item.get("lastModified"),
            "pipeline_tag": item.get("pipeline_tag"),
            "license": _hf_license(item),
        }
        for item in data[:limit]
        if isinstance(item, dict)
    ]


def build_engine_plan(config: LabConfig) -> str:
    lines = [
        "OpenFusion Lab engine plan",
        "",
        "Start each engine manually, then run lab validation before running experiments.",
        "",
    ]
    models_by_engine: dict[str, list[LabModel]] = {engine.name: [] for engine in config.engines}
    for model in config.models:
        models_by_engine.setdefault(model.engine, []).append(model)

    for engine in config.engines:
        engine_type = engine.type.lower()
        lines.append(f"Engine: {engine.name}")
        lines.append(f"Type: {engine.type}")
        lines.append(f"Base URL: {engine.base_url}")
        if engine_type == "ollama":
            for model in models_by_engine.get(engine.name, []):
                lines.append(f"  ollama pull {model.model}")
            lines.append("  Start Ollama manually and confirm the /v1 endpoint is reachable.")
        elif engine_type == "vllm":
            engine_models = models_by_engine.get(engine.name, [])
            if not engine_models:
                lines.append("  python -m vllm.entrypoints.openai.api_server --model MODEL_ID")
            for model in engine_models:
                lines.append(f"  python -m vllm.entrypoints.openai.api_server --model {model.model}")
            lines.append("  Bind the server to the configured base URL before validation.")
        elif engine_type in {"tgi", "text-generation-inference"}:
            engine_models = models_by_engine.get(engine.name, [])
            if not engine_models:
                lines.append(
                    "  docker run --gpus all -p 8080:80 "
                    "ghcr.io/huggingface/text-generation-inference:latest --model-id MODEL_ID"
                )
            for model in engine_models:
                lines.append(
                    "  docker run --gpus all -p 8080:80 "
                    f"ghcr.io/huggingface/text-generation-inference:latest --model-id {model.model}"
                )
            lines.append("  Use an OpenAI-compatible adapter or endpoint matching the base URL.")
        else:
            lines.append("  Start the OpenAI-compatible service manually for this engine.")
        lines.append("")
    lines.append("Next: openfusion lab validate LAB_YAML")
    return "\n".join(lines)


def _parse_lab_example(raw: dict[str, Any]) -> LabExample:
    if not isinstance(raw, dict):
        raise ValueError("example must be a JSON object")
    example_id = str(raw.get("id") or "").strip()
    if not example_id:
        raise ValueError("example id is required")

    if isinstance(raw.get("messages"), list):
        messages = [ChatMessage.model_validate(message) for message in raw["messages"]]
        reference = raw.get("answer", raw.get("reference"))
    elif isinstance(raw.get("prompt"), str):
        messages = [ChatMessage(role="user", content=raw["prompt"])]
        reference = raw.get("reference", raw.get("answer"))
    else:
        raise ValueError("example must include messages or prompt")

    references = reference if isinstance(reference, list) else [reference]
    rendered_references = [str(item) for item in references if item is not None and str(item)]
    if not rendered_references:
        raise ValueError("example must include answer or reference")

    return LabExample(
        id=example_id,
        messages=messages,
        references=rendered_references,
        answer_regex=raw.get("answer_regex"),
        subject=raw.get("subject"),
        metadata={
            key: value
            for key, value in raw.items()
            if key
            not in {
                "id",
                "messages",
                "prompt",
                "answer",
                "reference",
                "answer_regex",
                "subject",
            }
        },
    )


async def _run_strategy(
    engine: FusionEngine,
    config: LabConfig,
    strategy: LabStrategy,
    examples: list[LabExample],
) -> list[LabExampleResult]:
    results: list[LabExampleResult] = []
    for example in examples:
        started = time.perf_counter()
        try:
            fusion_result = await engine.run(
                messages=example.messages,
                strategy=strategy.name,
                max_tokens=config.experiment.max_tokens,
                max_total_calls=config.experiment.max_total_calls,
                temperature=config.experiment.temperature,
                vote_regex=example.answer_regex,
                self_moa_provider=strategy.self_moa_provider,
                self_moa_samples=strategy.self_moa_samples,
                self_moa_mode=strategy.self_moa_mode,
            )
            correct = _grade_lab_output(config.dataset.answer_mode, fusion_result, example)
            results.append(
                LabExampleResult(
                    id=example.id,
                    correct=correct,
                    output=fusion_result.final,
                    calls=_call_count(fusion_result),
                    latency_ms=_latency_ms(fusion_result, started),
                    prompt_tokens=fusion_result.usage.prompt_tokens,
                    completion_tokens=fusion_result.usage.completion_tokens,
                    total_tokens=fusion_result.usage.total_tokens,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one failed example should not stop the lab
            results.append(
                LabExampleResult(
                    id=example.id,
                    correct=False,
                    error=f"{exc.__class__.__name__}: {exc}",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            )
    return results


def _grade_lab_output(
    answer_mode: Literal["exact", "regex", "exact_or_regex"],
    result: FusionResult,
    example: LabExample,
) -> bool:
    if answer_mode == "exact":
        return is_exact_match(result.final, example.references)
    if answer_mode == "regex":
        return is_exact_match(result.final, example.references, example.answer_regex)
    return is_exact_match(result.final, example.references, example.answer_regex)


def _summarize_strategies(
    by_strategy: dict[str, list[LabExampleResult]],
    baseline_strategy: str | None,
) -> list[StrategyResultSummary]:
    baseline_results = by_strategy.get(baseline_strategy or "")
    baseline_by_id = {result.id: result for result in baseline_results or []}
    summaries: list[StrategyResultSummary] = []
    for strategy, results in by_strategy.items():
        metrics = _metrics(results, baseline_by_id if strategy != baseline_strategy else None)
        summaries.append(
            StrategyResultSummary(
                strategy=strategy,
                metrics=metrics,
                baseline_strategy=baseline_strategy if strategy != baseline_strategy else None,
            )
        )
    return summaries


def _metrics(
    results: list[LabExampleResult],
    baseline_by_id: dict[str, LabExampleResult] | None,
) -> LabMetricSummary:
    total = len(results)
    correct = sum(result.correct for result in results)
    accuracy = correct / total if total else 0.0
    total_calls = sum(result.calls for result in results)
    total_latency = sum(result.latency_ms for result in results)
    total_tokens = sum(result.total_tokens for result in results)
    wins = ties = losses = 0
    if baseline_by_id is not None:
        for result in results:
            baseline = baseline_by_id.get(result.id)
            if baseline is None:
                continue
            if result.correct and not baseline.correct:
                wins += 1
            elif result.correct == baseline.correct:
                ties += 1
            else:
                losses += 1
    return LabMetricSummary(
        total_examples=total,
        correct=correct,
        accuracy=accuracy,
        total_calls=total_calls,
        avg_calls_per_example=total_calls / total if total else 0.0,
        total_latency_ms=total_latency,
        avg_latency_ms=total_latency / total if total else 0.0,
        p50_latency_ms=_percentile([result.latency_ms for result in results], 50),
        p95_latency_ms=_percentile([result.latency_ms for result in results], 95),
        total_prompt_tokens=sum(result.prompt_tokens for result in results),
        total_completion_tokens=sum(result.completion_tokens for result in results),
        total_tokens=total_tokens,
        accuracy_per_call=accuracy / total_calls if total_calls else 0.0,
        accuracy_per_1k_tokens=accuracy / (total_tokens / 1000) if total_tokens else 0.0,
        failures=sum(1 for result in results if result.error),
        win_rate_vs_baseline=(wins / total) if total and baseline_by_id is not None else None,
        tie_rate_vs_baseline=(ties / total) if total and baseline_by_id is not None else None,
        loss_rate_vs_baseline=(losses / total) if total and baseline_by_id is not None else None,
    )


def _balanced_score(
    metrics: LabMetricSummary,
    baseline: LabMetricSummary | None,
    settings: LabRecommendationSettings,
) -> float:
    baseline_latency = max(1.0, baseline.avg_latency_ms if baseline else metrics.avg_latency_ms)
    latency_ratio = metrics.avg_latency_ms / baseline_latency
    call_ratio = metrics.avg_calls_per_example / max(1.0, baseline.avg_calls_per_example if baseline else 1.0)
    latency_penalty = max(0.0, latency_ratio - 1.0) * 0.05
    call_penalty = max(0.0, call_ratio - 1.0) * (0.03 if settings.prefer_lower_calls else 0.01)
    failure_penalty = (metrics.failures / metrics.total_examples) if metrics.total_examples else 0.0
    return metrics.accuracy - latency_penalty - call_penalty - failure_penalty


def _accuracy_explanation(
    best_accuracy: StrategyResultSummary,
    baseline: StrategyResultSummary | None,
) -> str:
    if baseline and baseline.metrics.avg_latency_ms > 0 and best_accuracy.strategy != baseline.strategy:
        ratio = best_accuracy.metrics.avg_latency_ms / baseline.metrics.avg_latency_ms
        return (
            f"{best_accuracy.strategy} had the best accuracy but was "
            f"{ratio:.1f}x slower than {baseline.strategy}."
        )
    return f"{best_accuracy.strategy} had the best accuracy."


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    if percentile == 50:
        return float(statistics.median(sorted_values))
    index = int(round((percentile / 100) * (len(sorted_values) - 1)))
    return float(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _call_count(result: FusionResult) -> int:
    return sum(
        1
        for step in result.trace
        if step.provider and step.status in {"ok", "error"} and step.latency_ms is not None
    )


def _latency_ms(result: FusionResult, started: float) -> int:
    trace_latency = sum(step.latency_ms or 0 for step in result.trace)
    if trace_latency:
        return trace_latency
    return int((time.perf_counter() - started) * 1000)


def _resolve_dataset_path(dataset_path: str, lab_path: str | Path | None) -> Path:
    path = Path(dataset_path)
    if path.is_absolute() or path.exists() or lab_path is None:
        return path
    return Path(lab_path).resolve().parent / path


def _safe_engine_metadata(engine: LabEngine) -> dict[str, Any]:
    return {
        "name": engine.name,
        "type": engine.type,
        "base_url": engine.base_url,
        "launch": engine.launch,
    }


def _safe_model_metadata(model: LabModel) -> dict[str, Any]:
    return {
        "provider_name": model.provider_name,
        "engine": model.engine,
        "model": model.model,
        "weight": model.weight,
        "timeout_seconds": model.timeout_seconds,
    }


def _sample_results(by_strategy: dict[str, list[LabExampleResult]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy, results in by_strategy.items():
        for result in results:
            rows.append(
                {
                    "strategy": strategy,
                    "id": result.id,
                    "correct": result.correct,
                    "error": result.error,
                    "calls": result.calls,
                    "latency_ms": result.latency_ms,
                    "total_tokens": result.total_tokens,
                }
            )
    return rows


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_strategy(summaries: list[StrategyResultSummary]) -> str | None:
    if any(summary.strategy == "fallback" for summary in summaries):
        return "fallback"
    return summaries[0].strategy if summaries else None


def _hf_license(item: dict[str, Any]) -> str | None:
    card_data = item.get("cardData")
    if isinstance(card_data, dict) and card_data.get("license"):
        return str(card_data["license"])
    tags = item.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                return tag.split(":", 1)[1]
    return None
