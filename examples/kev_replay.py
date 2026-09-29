"""Replay recorded candidate selection; labels are never included in a model request.

python examples/kev_replay.py examples/results/kev-spark-2026-09-29.json --split fresh
Add --base-url http://localhost:8009/v1 to re-run Kev on the same candidates.
This script does not execute generated code.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from openfusion.config import DecisionModelConfig
from openfusion.decisions import DecisionClient


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--split", default="fresh", choices=["dev", "test", "confirm", "fresh"])
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="kev-latest")
    parser.add_argument("--api-key-env")
    parser.add_argument("--policy", choices=["kev", "verified_kev"], default="verified_kev")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    records = [r for r in data["records"] if r["split"] == args.split]
    if not records:
        raise SystemExit("No records for this split")
    client = DecisionClient(DecisionModelConfig(
        base_url=args.base_url or "http://localhost:8009/v1", model=args.model,
        api_key_env=args.api_key_env,
    ))
    rows = []
    for record in records:
        if args.base_url:
            indices = list(range(len(record["candidates"])))
            if args.policy == "verified_kev":
                indices = [i for i in indices if record["visible_pass"][i]] or indices
            if len(indices) == 1:
                index = indices[0]
            else:
                decision = await client.select(
                    record["question"], [record["candidates"][i] for i in indices],
                )
                index = indices[decision.index]
        else:
            index = record["policies"][args.policy]["index"]
        rows.append({"id": record["id"], "index": index,
                     "correct": record["candidate_correct"][index]})
    result = {"policy": args.policy, "split": args.split, "n": len(rows),
              "correct": sum(r["correct"] for r in rows), "rows": rows}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
