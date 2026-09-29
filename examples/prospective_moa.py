"""Prospective sequential code-selection experiment. Run on a Docker-equipped host.

Endpoint configuration is private and supplied separately. Hidden grading is a separate
command that cannot affect generation, selection, retries, or recorded latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

IMAGE = "python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f"
INSTRUCTION = (
    "Implement the requested Python function(s). Python 3.12 and its standard library "
    "are available. Return complete executable Python code in one fenced python "
    "block, including any required imports. Do not include explanations."
)
ARMS = ("single", "qwen_repair", "moa")
SEEDS = (17429, 58103)
SANDBOX = """import contextlib,json,sys
r=json.load(sys.stdin)
ns={"__name__":"__candidate__"}
out=[]
try:
 with open('/dev/null','w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
  exec(compile(r["code"],"candidate.py","exec"),ns)
  exec(r.get("setup",""),ns)
  for test in r["tests"]:
   try:
    exec(compile(test,"test.py","exec"),ns)
    out.append({"pass":True})
   except BaseException as e:
    out.append({"pass":False,"error":type(e).__name__+": "+str(e)[:300]})
except BaseException as e:
 out=[{"pass":False,"error":type(e).__name__+": "+str(e)[:300]} for _ in r["tests"]]
print("OF_RESULT_"+json.dumps({"results":out,"pass":bool(out) and all(x["pass"] for x in out)}))
"""


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else text.strip()


def evaluate(code, tests, setup=""):
    """No mounts, network, GPU, privilege, or persistent writable filesystem."""
    name = "of-eval-" + str(time.time_ns())
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
        "--pids-limit=32",
        "--memory=256m",
        "--cpus=1",
        "--ulimit",
        "nofile=64:64",
        "--ulimit",
        "fsize=1048576:1048576",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        IMAGE,
        "python",
        "-I",
        "-c",
        SANDBOX,
    ]
    start = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            input=json.dumps({"code": code, "tests": tests, "setup": setup}),
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
        lines = [line for line in p.stdout.splitlines() if line.startswith("OF_RESULT_")]
        if not lines:
            # Docker infrastructure errors must not be mistaken for wrong answers.
            if p.returncode == 125:
                raise RuntimeError("Docker infrastructure failure: " + p.stderr[:300])
            result = {"pass": False, "results": [], "error": "execution_error"}
        else:
            result = json.loads(lines[-1][len("OF_RESULT_") :])
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15, check=False)
        result = {"pass": False, "results": [], "error": "execution_timeout"}
    result["elapsed"] = time.perf_counter() - start
    return result


def request(endpoint, prompt, temperature, seed, max_tokens=2048):
    api = endpoint.get("api", "ollama")
    if api == "ollama":
        body = {
            "model": endpoint["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "60m",
            "options": {
                "num_ctx": 8192,
                "num_predict": max_tokens,
                "temperature": temperature,
                "seed": seed,
            },
        }
        suffix = "/api/chat"
    else:
        body = {
            "model": endpoint["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        suffix = "/chat/completions"
        if endpoint.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = endpoint["chat_template_kwargs"]
    start = time.perf_counter()
    req = urllib.request.Request(
        endpoint["url"].rstrip("/") + suffix,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as response:
        raw = json.load(response)
    if api == "ollama":
        result = {
            "text": raw.get("message", {}).get("content") or "",
            "output_tokens": raw.get("eval_count", 0),
            "input_tokens": raw.get("prompt_eval_count", 0),
            "finish": raw.get("done_reason"),
            "load_ns": raw.get("load_duration", 0),
            "generation_ns": raw.get("eval_duration", 0),
            "prefill_ns": raw.get("prompt_eval_duration", 0),
        }
    else:
        # Never retain hidden reasoning text.
        result = {
            "text": raw["choices"][0]["message"].get("content") or "",
            "output_tokens": raw.get("usage", {}).get("completion_tokens", 0),
            "input_tokens": raw.get("usage", {}).get("prompt_tokens", 0),
            "finish": raw["choices"][0].get("finish_reason"),
        }
    result.update(
        elapsed=time.perf_counter() - start,
        model=endpoint["model"],
        temperature=temperature,
        seed=seed,
        max_output_tokens=max_tokens,
    )
    if endpoint.get("chat_template_kwargs"):
        result["chat_template_kwargs"] = endpoint["chat_template_kwargs"]
    return result


def run_policy(item, arm, seed, endpoints, generate=request, check=evaluate):
    # item contains public specification, visible tests and public setup only.
    base = (
        item["prompt"]
        + "\n\nVisible acceptance test:\n"
        + "\n".join(item["visible"])
        + "\n\n"
        + INSTRUCTION
    )
    calls = []
    start = time.perf_counter()
    budget = 1 if arm in ("single", "stronger") else 4
    selected = None
    for step in range(budget):
        tag, temp, prompt = "qwen", (0 if step == 0 else 0.7), base
        if arm == "stronger":
            tag, temp = "stronger", 0
        elif arm == "moa" and step in (1, 2):
            tag = ("phi", "gemma")[step - 1]
        elif step and (arm == "qwen_repair" or step == 3):
            prior = calls[-1] if arm == "qwen_repair" else calls[0]
            prompt += (
                "\n\nPrevious code:\n"
                + prior["text"]
                + "\n\nVisible-test feedback:\n"
                + json.dumps({k: v for k, v in prior["visible"].items() if k != "elapsed"})
                + "\nRepair the implementation to satisfy the specification generally."
            )
        record = generate(
            endpoints[tag], prompt, temp, seed + step, 8192 if arm == "stronger" else 2048
        )
        record["visible"] = check(extract(record["text"]), item["visible"], item.get("setup", ""))
        record["tag"] = tag
        calls.append(record)
        selected = step
        if record["visible"]["pass"]:
            break
    # A failed final repair is still the declared fallback; no hidden test consulted.
    return {
        "id": item["id"],
        "arm": arm,
        "seed": seed,
        "selected": selected,
        "calls": calls,
        "elapsed": time.perf_counter() - start,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def run(args):
    items = read(args.tasks)
    endpoints = read(args.endpoints)
    registration = None
    if not args.development:
        if not args.protocol or not args.registration_commit:
            raise ValueError("Confirmation requires a frozen protocol and its registered commit")
        protocol = read(args.protocol)
        expected = protocol["code_sha256"]["examples/prospective_moa.py"]
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("Runner differs from the frozen protocol")
        if (
            hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest()
            != protocol["artifacts"]["tasks-public.json"]
        ):
            raise ValueError("Task cohort differs from the frozen protocol")
        registration = {
            "commit": args.registration_commit,
            "protocol_sha256": hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
        }
    out = Path(args.output)
    seeds = SEEDS if not args.development else (SEEDS[0],)
    arms = ARMS + (("stronger",) if args.stronger else ())
    jobs = [
        (item, arm, seed)
        for item in items
        for seed in seeds
        for arm in arms
        if arm != "stronger" or (seed == seeds[0] and item.get("stronger_subset", args.development))
    ]
    random.Random(29092026).shuffle(jobs)
    out.mkdir(parents=True, exist_ok=True)
    for idx, (item, arm, seed) in enumerate(jobs):
        path = out / f"{item['id']}-{arm}-{seed}.json"
        if path.exists():
            continue
        # Fail-stop on transport/infrastructure errors, preserve completed jobs.
        value = run_policy(item, arm, seed, endpoints)
        value["registration"] = registration
        dump(path, value)
        print(
            json.dumps(
                {
                    "done": idx + 1,
                    "total": len(jobs),
                    "id": item["id"],
                    "arm": arm,
                    "calls": len(value["calls"]),
                }
            ),
            flush=True,
        )


def grade(args):
    tasks = {r["id"]: r for r in read(args.tasks)}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(args.records).glob("*.json")):
        dest = out / source.name
        if dest.exists():
            continue
        row = read(source)
        item = tasks[row["id"]]
        answer = row["calls"][row["selected"]]["text"]
        result = evaluate(extract(answer), item["tests"], item.get("setup", ""))
        dump(
            dest,
            {
                "id": row["id"],
                "arm": row["arm"],
                "seed": row["seed"],
                "correct": result["pass"],
                "evaluation": result,
                "record_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        )
    print("Grading complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tasks", required=True)
    p.add_argument("--endpoints", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--development", action="store_true")
    p.add_argument("--stronger", action="store_true")
    p.add_argument("--protocol")
    p.add_argument("--registration-commit")
    p = sub.add_parser("grade")
    p.add_argument("--tasks", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    (run if args.action == "run" else grade)(args)


if __name__ == "__main__":
    main()
