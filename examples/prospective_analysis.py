"""Paired, task-clustered analysis of the prospectively frozen workflow study."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def analyze(records, grades, resamples=10000):
    grade = {(r["id"], r["arm"], r["seed"]): r["correct"] for r in grades}
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["id"], r["arm"], r["seed"])
        grouped[r["id"]][r["arm"]].append(
            {
                "correct": float(grade[key]),
                "elapsed": r["elapsed"],
                "calls": len(r["calls"]),
                "tokens": sum(c["input_tokens"] + c["output_tokens"] for c in r["calls"]),
                "output_tokens": sum(c["output_tokens"] for c in r["calls"]),
                "model_seconds": sum(c["elapsed"] for c in r["calls"]),
                "load_seconds": sum(c.get("load_ns", 0) for c in r["calls"]) / 1e9,
                "truncations": sum(c["finish"] == "length" for c in r["calls"]),
            }
        )
    summary = {}
    for arm in sorted({r["arm"] for r in records}):
        values = [v for row in grouped.values() for v in row.get(arm, [])]
        summary[arm] = {
            "runs": len(values),
            "correct": sum(v["correct"] for v in values),
            **{f"mean_{k}": sum(v[k] for v in values) / len(values) for k in values[0]},
            "p50_elapsed": quantile([v["elapsed"] for v in values], 0.5),
            "p95_elapsed": quantile([v["elapsed"] for v in values], 0.95),
        }
    tasks = []
    for ident, row in sorted(grouped.items()):
        if "moa" not in row or "qwen_repair" not in row:
            continue
        if len(row["moa"]) != 2 or len(row["qwen_repair"]) != 2:
            raise ValueError("Incomplete primary task: " + ident)
        averages = {
            arm: {k: sum(v[k] for v in row[arm]) / len(row[arm]) for k in row[arm][0]}
            for arm in ("moa", "qwen_repair")
        }
        tasks.append(
            {
                "id": ident,
                "delta": averages["moa"]["correct"] - averages["qwen_repair"]["correct"],
                **averages,
            }
        )
    n = len(tasks)
    delta = sum(t["delta"] for t in tasks) / n
    rng = random.Random(143092026)
    bootstrap, latency, tokens = [], [], []
    for _ in range(resamples):
        sample = rng.choices(tasks, k=n)
        bootstrap.append(sum(t["delta"] for t in sample) / n)
        latency.append(
            sum(t["moa"]["elapsed"] for t in sample)
            / sum(t["qwen_repair"]["elapsed"] for t in sample)
        )
        tokens.append(
            sum(t["moa"]["tokens"] for t in sample)
            / sum(t["qwen_repair"]["tokens"] for t in sample)
        )
    interval = [quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)]
    latency_ratio = sum(t["moa"]["elapsed"] for t in tasks) / sum(
        t["qwen_repair"]["elapsed"] for t in tasks
    )
    token_ratio = sum(t["moa"]["tokens"] for t in tasks) / sum(
        t["qwen_repair"]["tokens"] for t in tasks
    )
    # Flip the entire task, keeping both seeds together.
    nonzero = [t["delta"] for t in tasks if t["delta"]]
    observed = abs(sum(nonzero))
    exceed = 0
    for _ in range(100000):
        value = abs(sum(v if rng.getrandbits(1) else -v for v in nonzero))
        exceed += value >= observed - 1e-12
    p = (exceed + 1) / 100001
    supported = (
        interval[0] > 0
        and delta >= 0.05
        and latency_ratio <= 1.5
        and token_ratio <= 1.5
        and p < 0.05
    )
    if supported:
        verdict = "supported under the tested conditions"
    elif interval[1] < 0.05:
        verdict = "minimum accuracy benefit not supported under the tested conditions"
    else:
        verdict = "inconclusive against the predefined deployment criteria"
    return {
        "n_tasks": n,
        "arms": summary,
        "primary": {
            "accuracy_difference": delta,
            "cluster_bootstrap_95ci": interval,
            "task_sign_flip_p": p,
            "mean_latency_ratio": latency_ratio,
            "latency_ratio_95ci": [quantile(latency, 0.025), quantile(latency, 0.975)],
            "mean_token_ratio": token_ratio,
            "token_ratio_95ci": [quantile(tokens, 0.025), quantile(tokens, 0.975)],
            "positive_task_differences": sum(t["delta"] > 0 for t in tasks),
            "negative_task_differences": sum(t["delta"] < 0 for t in tasks),
            "verdict": verdict,
        },
        "task_outcomes": tasks,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.bundle.read_text(encoding="utf-8"))
    result = analyze(data["records"], data["grades"])
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "task_outcomes"}, indent=2))
