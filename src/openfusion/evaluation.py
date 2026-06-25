from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .fusion import FusionEngine
from .schema import ChatMessage


class EvalCase(BaseModel):
    id: str
    prompt: str
    reference: str | list[str]
    system: str | None = None
    answer_regex: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCaseResult(BaseModel):
    id: str
    correct: bool
    output: str
    references: list[str]
    strategy: str
    error: str | None = None


class EvalSummary(BaseModel):
    strategy: str
    total: int
    correct: int
    accuracy: float
    results: list[EvalCaseResult]


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
) -> EvalSummary:
    results: list[EvalCaseResult] = []
    for case in cases:
        messages: list[ChatMessage] = []
        if case.system:
            messages.append(ChatMessage(role="system", content=case.system))
        messages.append(ChatMessage(role="user", content=case.prompt))
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
            output = fusion_result.final
            correct = is_exact_match(output, references, case.answer_regex)
            results.append(
                EvalCaseResult(
                    id=case.id,
                    correct=correct,
                    output=output,
                    references=references,
                    strategy=fusion_result.strategy,
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
                )
            )
    correct = sum(result.correct for result in results)
    total = len(results)
    return EvalSummary(
        strategy=strategy,
        total=total,
        correct=correct,
        accuracy=(correct / total) if total else 0.0,
        results=results,
    )
