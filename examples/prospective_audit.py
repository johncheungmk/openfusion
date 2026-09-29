"""Independent structural audit of completed prospective records; no model calls."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(bundle):
    records = bundle["records"]
    grades = bundle["grades"]
    grading_by_key = {(r["id"], r["arm"], r["seed"]): r for r in grades}
    tasks = {r["id"]: r for r in bundle["tasks"]}
    seen = set()
    issues = []
    for r in records:
        key = (r["id"], r["arm"], r["seed"])
        if key in seen:
            issues.append(f"Duplicate record {key}")
        seen.add(key)
        grade = grading_by_key.get(key, {})
        if "evaluation" in grade:
            evaluation = grade["evaluation"]
            expected_pass = bool(evaluation.get("results")) and all(
                result.get("pass") is True for result in evaluation.get("results", [])
            )
            if grade["correct"] != expected_pass or evaluation["pass"] != expected_pass:
                issues.append(f"Grading outcome mismatch {key}")
        if "record_sha256" in grade:
            encoded = json.dumps(r, indent=2, ensure_ascii=False).encode("utf-8")
            if hashlib.sha256(encoded).hexdigest() != grade["record_sha256"]:
                issues.append(f"Generation/grading hash mismatch {key}")
        calls = r["calls"]
        maximum = 1 if r["arm"] in ("single", "stronger") else 4
        if not 1 <= len(calls) <= maximum:
            issues.append(f"Budget violation {key}")
        if r["selected"] != len(calls) - 1:
            issues.append(f"Selection mismatch {key}")
        if any(c["visible"]["pass"] for c in calls[:-1]):
            issues.append(f"Continued after acceptance {key}")
        if len(calls) < maximum and not calls[-1]["visible"]["pass"]:
            issues.append(f"Stopped before budget without acceptance {key}")
        expected = (
            ["qwen", "phi", "gemma", "qwen"]
            if r["arm"] == "moa"
            else ["stronger"]
            if r["arm"] == "stronger"
            else ["qwen"] * maximum
        )
        if [c["tag"] for c in calls] != expected[: len(calls)]:
            issues.append(f"Model order mismatch {key}")
        for step, c in enumerate(calls):
            if c["seed"] != r["seed"] + step:
                issues.append(f"Seed mismatch {key}")
            if c["max_output_tokens"] != (8192 if r["arm"] == "stronger" else 2048):
                issues.append(f"Token cap mismatch {key}")
            if c["temperature"] != (0 if step == 0 else 0.7):
                issues.append(f"Temperature mismatch {key}")
        if (
            not r.get("registration")
            or r["registration"]["commit"] != bundle["registration_commit"]
        ):
            issues.append(f"Registration mismatch {key}")
        if r["id"] not in tasks:
            issues.append(f"Task outside frozen cohort {key}")
        if r["arm"] == "stronger" and not tasks[r["id"]]["stronger_subset"]:
            issues.append(f"Stronger task outside frozen subset {key}")
    expected_keys = {
        (r["id"], arm, seed)
        for r in tasks.values()
        for arm in ("single", "qwen_repair", "moa")
        for seed in (17429, 58103)
    }
    expected_keys |= {(r["id"], "stronger", 17429) for r in tasks.values() if r["stronger_subset"]}
    if seen != expected_keys:
        issues.append(f"Unexpected/missing records: {len(seen ^ expected_keys)}")
    graded = {(r["id"], r["arm"], r["seed"]) for r in grades}
    if graded != seen or len(grades) != len(graded):
        issues.append("Grading records not one-to-one with generation records")
    return {
        "records": len(records),
        "grades": len(grades),
        "tasks": len(tasks),
        "arm_counts": dict(Counter(r["arm"] for r in records)),
        "issues": issues,
        "passed": not issues,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.bundle.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
