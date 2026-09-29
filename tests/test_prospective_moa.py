"""Offline checks for the frozen experiment controller, not model quality."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "prospective_moa", Path(__file__).parents[1] / "examples" / "prospective_moa.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def simulate(arm, passes):
    requests = []
    endpoints = {k: {"model": k} for k in ("qwen", "phi", "gemma", "stronger")}
    item = {"id": "unit", "prompt": "Implement f(x).", "visible": ["assert f(1)==2"]}

    def generate(endpoint, prompt, temperature, seed, max_tokens):
        requests.append((endpoint["model"], prompt, max_tokens))
        return {"text": f"answer-{len(requests)}"}

    def check(code, tests, setup):
        assert tests == item["visible"]
        return {"pass": passes[len(requests) - 1], "results": []}

    row = module.run_policy(item, arm, 42, endpoints, generate, check)
    return row, requests


def test_early_stop_uses_no_alternative_calls():
    for arm in ("single", "qwen_repair", "moa"):
        row, calls = simulate(arm, [True])
        assert len(calls) == 1
        assert row["selected"] == 0


def test_moa_order_and_budget_with_final_anchor_repair():
    row, calls = simulate("moa", [False] * 4)
    assert [c[0] for c in calls] == ["qwen", "phi", "gemma", "qwen"]
    assert "answer-1" in calls[3][1]
    assert "answer-2" not in calls[3][1]
    assert row["selected"] == 3
    assert all(c[2] == 2048 for c in calls)


def test_single_model_repair_uses_only_last_visible_feedback():
    row, calls = simulate("qwen_repair", [False, False, True])
    assert len(calls) == 3
    assert all(c[0] == "qwen" for c in calls)
    assert "answer-2" in calls[2][1]
    assert "answer-1" not in calls[2][1]
    assert row["selected"] == 2


def test_moa_stops_at_passing_alternative():
    row, calls = simulate("moa", [False, True])
    assert [c[0] for c in calls] == ["qwen", "phi"]
    assert row["selected"] == 1


def test_direct_baselines_have_one_call_even_on_failure():
    for arm in ("single", "stronger"):
        row, calls = simulate(arm, [False])
        assert len(calls) == 1
        assert row["selected"] == 0
    assert calls[0][2] == 8192


def test_code_extraction():
    assert module.extract("```python\ndef f(): return 1\n```") == "def f(): return 1"
