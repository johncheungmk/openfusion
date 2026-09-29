"""Check paired task-level aggregation without executing models or candidate code."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prospective_analysis", Path(__file__).parents[1] / "examples" / "prospective_analysis.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def records_and_grades():
    records, grades = [], []
    for task in ("a", "b"):
        for seed in (17429, 58103):
            for arm in ("moa", "qwen_repair"):
                records.append(
                    {
                        "id": task,
                        "arm": arm,
                        "seed": seed,
                        "elapsed": 1,
                        "calls": [
                            {
                                "input_tokens": 10,
                                "output_tokens": 10,
                                "elapsed": 0.8,
                                "finish": "stop",
                            }
                        ],
                    }
                )
                grades.append(
                    {
                        "id": task,
                        "arm": arm,
                        "seed": seed,
                        "correct": (task == "a") == (arm == "moa"),
                    }
                )
    return records, grades


def test_seeds_are_clustered_within_tasks():
    records, grades = records_and_grades()
    result = module.analyze(records, grades, resamples=100)
    assert result["n_tasks"] == 2
    assert result["primary"]["accuracy_difference"] == 0
    assert [r["delta"] for r in result["task_outcomes"]] == [1, -1]
    assert result["primary"]["mean_latency_ratio"] == 1
    assert result["primary"]["mean_token_ratio"] == 1


def test_missing_seed_is_not_silently_treated_as_complete():
    records, grades = records_and_grades()
    with pytest.raises(ValueError, match="Incomplete primary task"):
        module.analyze(records[:-1], grades, resamples=10)


def test_missing_grading_result_fails():
    records, grades = records_and_grades()
    with pytest.raises(KeyError):
        module.analyze(records, grades[:-1], resamples=10)
