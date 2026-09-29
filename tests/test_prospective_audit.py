"""Check that independent artifact auditing detects controller violations."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "prospective_audit", Path(__file__).parents[1] / "examples" / "prospective_audit.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def bundle():
    records = []
    for arm in ("single", "qwen_repair", "moa"):
        for seed in (17429, 58103):
            records.append(
                {
                    "id": "t",
                    "arm": arm,
                    "seed": seed,
                    "selected": 0,
                    "registration": {"commit": "registered"},
                    "calls": [
                        {
                            "tag": "qwen",
                            "seed": seed,
                            "temperature": 0,
                            "max_output_tokens": 2048,
                            "visible": {"pass": True},
                        }
                    ],
                }
            )
    return {
        "registration_commit": "registered",
        "tasks": [{"id": "t", "stronger_subset": False}],
        "records": records,
        "grades": [{"id": r["id"], "arm": r["arm"], "seed": r["seed"]} for r in records],
    }


def test_complete_policy_coverage_passes():
    assert module.audit(bundle())["passed"]


def test_audit_detects_missing_run():
    value = bundle()
    value["records"].pop()
    assert not module.audit(value)["passed"]


def test_audit_detects_unregistered_selection():
    value = bundle()
    value["records"][0]["selected"] = 1
    assert any("Selection mismatch" in issue for issue in module.audit(value)["issues"])


def test_audit_detects_hidden_change_to_model_order():
    value = bundle()
    value["records"][-1]["calls"][0]["tag"] = "phi"
    assert any("Model order mismatch" in issue for issue in module.audit(value)["issues"])


def test_audit_detects_inconsistent_grading_boolean():
    value = bundle()
    value["grades"][0].update(
        correct=True, evaluation={"pass": True, "results": [{"pass": False}]}
    )
    assert any("Grading outcome mismatch" in issue for issue in module.audit(value)["issues"])
