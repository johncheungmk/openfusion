"""Recompute fixed-budget selection comparisons from recorded test outcomes.

Usage: python examples/moa_ablation_analysis.py examples/results/moa-ablation-2026-09-29.json
No network requests or generated-code execution are performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def paired(rows, a, b):
    wins = sum(r["policies"][a]["correct"] and not r["policies"][b]["correct"] for r in rows)
    losses = sum(not r["policies"][a]["correct"] and r["policies"][b]["correct"] for r in rows)
    n = wins + losses
    p = min(1, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1
    return {"wins": wins, "losses": losses, "exact_mcnemar_p": p}


def analyze(data, split):
    rows = [r for r in data["records"] if r["split"] == split]
    if not rows:
        raise ValueError("No records for requested split")
    for row in rows:
        checks = row["candidate_test_results"]
        for arm, tags in data["protocol"]["arms"].items():
            for count in (1, 2):
                chosen = next((t for t in tags if all(checks[t][:count])), tags[0])
                policy = row["policies"][f"{arm}_check{count}"]
                assert policy["selected"] == chosen, row["id"]
                assert policy["correct"] == all(checks[chosen]), row["id"]
                assert policy["remaining_hidden_correct"] == all(checks[chosen][2:]), row["id"]
    names = list(rows[0]["policies"])
    summary = {
        "split": split,
        "n": len(rows),
        "correct": {n: sum(r["policies"][n]["correct"] for r in rows) for n in names},
        "primary": paired(rows, "mixed_qwen_first_check1", "qwen_three_check1"),
        "phi_anchor": paired(rows, "mixed_phi_first_check1", "phi_three_check1"),
    }
    if all("flagship" in r for r in rows):
        summary["flagship_correct"] = sum(r["flagship"]["correct"] for r in rows)
        summary["flagship_with_length_retry_correct"] = sum(r["flagship_retry"]["correct"] for r in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--split", choices=["humaneval", "retrospective"], default="humaneval")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    print(json.dumps(analyze(data, args.split), indent=2))


if __name__ == "__main__":
    main()
