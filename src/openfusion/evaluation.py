from __future__ import annotations

import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .fusion import FusionEngine
from .schema import ChatMessage, FusionResult


GraderMode = Literal["exact_match", "regex", "llm_pairwise", "llm_rubric"]


class EvalCase(BaseModel):
    id: str
    prompt: str
    reference: str | list[str]
    system: str | None = None
    answer_regex: str | None = None
    rubric: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCaseResult(BaseModel):
    id: str
    correct: bool
    output: str
    references: list[str]
    strategy: str
    error: str | None = None
    calls: int = 0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    baseline_output: str | None = None
    grader: str = "exact_match"
    grade: Literal["win", "tie", "loss", "correct", "incorrect"] | None = None


class EvalMetrics(BaseModel):
    total_examples: int = 0
    accuracy: float = 0.0
    win_rate_vs_baseline: float = 0.0
    tie_rate_vs_baseline: float = 0.0
    loss_rate_vs_baseline: float = 0.0
    total_calls: int = 0
    avg_calls_per_example: float = 0.0
    total_latency_ms: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float | None = None
    accuracy_per_call: float = 0.0
    accuracy_per_1k_tokens: float = 0.0
    strategy_failures: int = 0


class EvalSummary(BaseModel):
    strategy: str
    total: int
    correct: int
    accuracy: float
    results: list[EvalCaseResult]
    metrics: EvalMetrics = Field(default_factory=EvalMetrics)
    grader: GraderMode = "exact_match"
    baseline_strategy: str | None = None


class EvalComparisonReport(BaseModel):
    object: str = "openfusion.eval_comparison"
    baseline_strategy: str
    max_total_calls: int | None = None
    grader: GraderMode = "exact_match"
    summaries: list[EvalSummary]


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[`*_>#]", "", normalized)
    normalized = re.sub(r"[^\w\s.-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip(" .-")


def extract_answer(text: str, answer_regex: str | None) -> str:
    if not answer_regex:
        return text
    matches = list(re.finditer(answer_regex, text, flags=re.IGNORECASE | re.MULTILINE))
    if not matches:
        return text
    match = matches[-1]
    return match.group(1) if match.lastindex else match.group(0)


def is_exact_match(output: str, references: list[str], answer_regex: str | None = None) -> bool:
    normalized_output = normalize_answer(extract_answer(output, answer_regex))
    return any(normalized_output == normalize_answer(reference) for reference in references)


def load_jsonl(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
            cases.append(EvalCase.model_validate(raw))
        except Exception as exc:  # noqa: BLE001 - include line number for dataset repair
            raise ValueError(f"Invalid evaluation JSONL at line {line_number}: {exc}") from exc
    return cases


async def evaluate_cases(
    engine: FusionEngine,
    cases: list[EvalCase],
    strategy: str,
    panel: list[str] | None = None,
    judge_provider: str | None = None,
    max_tokens: int | None = None,
    max_total_calls: int | None = None,
    grader: GraderMode = "exact_match",
    grader_provider: str | None = None,
    baseline_results: dict[str, EvalCaseResult] | None = None,
) -> EvalSummary:
    results: list[EvalCaseResult] = []
    for case in cases:
        messages = _case_messages(case)
        references = case.reference if isinstance(case.reference, list) else [case.reference]
        try:
            fusion_result = await engine.run(
                messages=messages,
                strategy=strategy,
                panel=panel,
                judge_provider=judge_provider,
                max_tokens=max_tokens,
                max_total_calls=max_total_calls,
                vote_regex=case.answer_regex,
            )
            baseline = baseline_results.get(case.id) if baseline_results else None
            correct, grade = await _grade_output(
                engine=engine,
                case=case,
                output=fusion_result.final,
                references=references,
                grader=grader,
                grader_provider=grader_provider,
                baseline_output=baseline.output if baseline else None,
            )
            results.append(
                _case_result(
                    case=case,
                    references=references,
                    strategy=fusion_result.strategy,
                    output=fusion_result.final,
                    fusion_result=fusion_result,
                    correct=correct,
                    grader=grader,
                    grade=grade,
                    baseline_output=baseline.output if baseline else None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one failed case should not abort a benchmark
            results.append(
                EvalCaseResult(
                    id=case.id,
                    correct=False,
                    output="",
                    references=references,
                    strategy=strategy,
                    error=f"{exc.__class__.__name__}: {exc}",
                    grader=grader,
                    grade="incorrect",
                )
            )
    return _summary(strategy, results, grader, baseline_results is not None)


async def compare_strategies(
    engine: FusionEngine,
    cases: list[EvalCase],
    strategies: list[str],
    panel: list[str] | None = None,
    judge_provider: str | None = None,
    max_tokens: int | None = None,
    max_total_calls: int | None = None,
    grader: GraderMode = "exact_match",
    grader_provider: str | None = None,
) -> EvalComparisonReport:
    ordered = _comparison_strategies(strategies, engine)
    baseline_strategy = ordered[0]
    summaries: list[EvalSummary] = []
    baseline = await evaluate_cases(
        engine=engine,
        cases=cases,
        strategy=baseline_strategy,
        panel=panel,
        judge_provider=judge_provider,
        max_tokens=max_tokens,
        max_total_calls=max_total_calls,
        grader="exact_match" if grader in {"llm_pairwise", "llm_rubric"} else grader,
    )
    summaries.append(baseline)
    baseline_results = {result.id: result for result in baseline.results}
    for strategy in ordered[1:]:
        summaries.append(
            await evaluate_cases(
                engine=engine,
                cases=cases,
                strategy=strategy,
                panel=panel,
                judge_provider=judge_provider,
                max_tokens=max_tokens,
                max_total_calls=max_total_calls,
                grader=grader,
                grader_provider=grader_provider,
                baseline_results=baseline_results,
            )
        )
    return EvalComparisonReport(
        baseline_strategy=baseline_strategy,
        max_total_calls=max_total_calls,
        grader=grader,
        summaries=summaries,
    )


def _case_messages(case: EvalCase) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if case.system:
        messages.append(ChatMessage(role="system", content=case.system))
    messages.append(ChatMessage(role="user", content=case.prompt))
    return messages


def _case_result(
    case: EvalCase,
    references: list[str],
    strategy: str,
    output: str,
    fusion_result: FusionResult,
    correct: bool,
    grader: GraderMode,
    grade: Literal["win", "tie", "loss", "correct", "incorrect"],
    baseline_output: str | None = None,
) -> EvalCaseResult:
    return EvalCaseResult(
        id=case.id,
        correct=correct,
        output=output,
        references=references,
        strategy=strategy,
        calls=_call_count(fusion_result),
        latency_ms=_latency_ms(fusion_result),
        prompt_tokens=fusion_result.usage.prompt_tokens,
        completion_tokens=fusion_result.usage.completion_tokens,
        total_tokens=fusion_result.usage.total_tokens,
        baseline_output=baseline_output,
        grader=grader,
        grade=grade,
    )


async def _grade_output(
    engine: FusionEngine,
    case: EvalCase,
    output: str,
    references: list[str],
    grader: GraderMode,
    grader_provider: str | None,
    baseline_output: str | None,
) -> tuple[bool, Literal["win", "tie", "loss", "correct", "incorrect"]]:
    if grader in {"exact_match", "regex"}:
        correct = is_exact_match(output, references, case.answer_regex if grader == "regex" else None)
        return correct, "correct" if correct else "incorrect"
    if grader == "llm_pairwise":
        grade = await _llm_pairwise_grade(engine, case, output, baseline_output, grader_provider)
        return grade in {"win", "tie"}, grade
    grade = await _llm_rubric_grade(engine, case, output, references, grader_provider)
    return grade in {"win", "tie", "correct"}, "correct" if grade in {"win", "tie", "correct"} else "incorrect"


async def _llm_pairwise_grade(
    engine: FusionEngine,
    case: EvalCase,
    output: str,
    baseline_output: str | None,
    grader_provider: str | None,
) -> Literal["win", "tie", "loss"]:
    if not baseline_output or not grader_provider:
        return "tie"
    prompt = (
        "Compare answer A against baseline answer B for the user's request. "
        "Return strict JSON only: {\"winner\":\"A|B|tie\"}.\n\n"
        f"User request:\n{case.prompt}\n\nAnswer A:\n{output}\n\nAnswer B:\n{baseline_output}"
    )
    text = await _call_grader(engine, grader_provider, prompt)
    data = _extract_json_object(text)
    winner = str(data.get("winner", "") if data else "").strip().casefold()
    if winner in {"a", "answer_a", "strategy"}:
        return "win"
    if winner in {"b", "answer_b", "baseline"}:
        return "loss"
    return "tie"


async def _llm_rubric_grade(
    engine: FusionEngine,
    case: EvalCase,
    output: str,
    references: list[str],
    grader_provider: str | None,
) -> Literal["correct", "incorrect"]:
    if not grader_provider:
        return "correct" if is_exact_match(output, references, case.answer_regex) else "incorrect"
    rubric = case.rubric or str(case.metadata.get("rubric") or "Judge correctness against the reference.")
    prompt = (
        "Grade the answer using the rubric. Return strict JSON only: "
        "{\"grade\":\"correct|incorrect\"}.\n\n"
        f"User request:\n{case.prompt}\n\nReference:\n{json.dumps(references)}\n\n"
        f"Rubric:\n{rubric}\n\nAnswer:\n{output}"
    )
    text = await _call_grader(engine, grader_provider, prompt)
    data = _extract_json_object(text)
    grade = str(data.get("grade", "") if data else "").strip().casefold()
    return "correct" if grade in {"correct", "pass", "yes"} else "incorrect"


async def _call_grader(engine: FusionEngine, provider_name: str, prompt: str) -> str:
    result = await engine.run_provider(
        provider_name=provider_name,
        messages=[
            ChatMessage(
                role="system",
                content="You are an evaluation judge. Return only the requested JSON.",
            ),
            ChatMessage(role="user", content=prompt),
        ],
        temperature=0.0,
        max_tokens=120,
    )
    return result.final


def _summary(
    strategy: str,
    results: list[EvalCaseResult],
    grader: GraderMode,
    has_baseline: bool,
) -> EvalSummary:
    correct = sum(result.correct for result in results)
    total = len(results)
    accuracy = (correct / total) if total else 0.0
    return EvalSummary(
        strategy=strategy,
        total=total,
        correct=correct,
        accuracy=accuracy,
        results=results,
        metrics=_metrics(results, accuracy, has_baseline),
        grader=grader,
        baseline_strategy="fallback" if has_baseline else None,
    )


def _metrics(results: list[EvalCaseResult], accuracy: float, has_baseline: bool) -> EvalMetrics:
    total = len(results)
    latencies = [result.latency_ms for result in results]
    total_calls = sum(result.calls for result in results)
    total_tokens = sum(result.total_tokens for result in results)
    wins = sum(1 for result in results if result.grade == "win")
    ties = sum(1 for result in results if result.grade == "tie")
    losses = sum(1 for result in results if result.grade == "loss")
    return EvalMetrics(
        total_examples=total,
        accuracy=accuracy,
        win_rate_vs_baseline=(wins / total) if total and has_baseline else 0.0,
        tie_rate_vs_baseline=(ties / total) if total and has_baseline else 0.0,
        loss_rate_vs_baseline=(losses / total) if total and has_baseline else 0.0,
        total_calls=total_calls,
        avg_calls_per_example=(total_calls / total) if total else 0.0,
        total_latency_ms=sum(latencies),
        avg_latency_ms=(sum(latencies) / total) if total else 0.0,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        prompt_tokens=sum(result.prompt_tokens for result in results),
        completion_tokens=sum(result.completion_tokens for result in results),
        total_tokens=total_tokens,
        estimated_cost=None,
        accuracy_per_call=(accuracy / total_calls) if total_calls else 0.0,
        accuracy_per_1k_tokens=(accuracy / (total_tokens / 1000)) if total_tokens else 0.0,
        strategy_failures=sum(1 for result in results if result.error),
    )


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


def _latency_ms(result: FusionResult) -> int:
    return sum(step.latency_ms or 0 for step in result.trace)


def _comparison_strategies(strategies: list[str], engine: FusionEngine) -> list[str]:
    selected = [strategy.strip() for strategy in strategies if strategy.strip()]
    if not selected:
        selected = ["fallback"]
    if "fallback" not in {strategy.replace("-", "_") for strategy in selected}:
        selected.insert(0, "fallback")
    if engine.config.fusion.self_moa_provider and "self_moa" not in {
        strategy.replace("-", "_") for strategy in selected
    }:
        selected.append("self_moa")
    return list(dict.fromkeys(selected))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
