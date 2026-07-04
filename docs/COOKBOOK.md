# OpenFusion Lab Cookbook

This cookbook shows a local two-model Lab run that compares each base model, fallback, and fusion strategies. It is a smoke-test workflow, not a benchmark claim.

## A. Install OpenFusion

Windows PowerShell:

```powershell
python -m pip install -U pip
python -m pip install -e ".[dev]"
openfusion --help
```

Linux/macOS Bash:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
openfusion --help
```

## B. Start Ollama and Pull Two Small Models

Windows PowerShell:

```powershell
ollama serve
ollama pull llama3.2:3b
ollama pull qwen3:latest
```

Linux/macOS Bash:

```bash
ollama serve
ollama pull llama3.2:3b
ollama pull qwen3:latest
```

## C. Create a BOM-Safe JSONL Dataset

On Windows, prefer Python for JSONL creation or ensure your editor writes UTF-8 without BOM. OpenFusion Lab accepts UTF-8 with or without BOM, but plain UTF-8 keeps files portable.

Windows PowerShell:

```powershell
@'
from pathlib import Path
rows = [
    '{"id":"math","prompt":"Return only the answer: 2 + 2","reference":"4"}',
    '{"id":"fifo","prompt":"Answer only A, B, C, or D. FIFO means A Stack B Queue C Tree D Heap","answer":"B","answer_regex":"\\\\b([ABCD])\\\\b"}',
]
Path("local-mini.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
'@ | python -
```

Linux/macOS Bash:

```bash
python - <<'PY'
from pathlib import Path
rows = [
    '{"id":"math","prompt":"Return only the answer: 2 + 2","reference":"4"}',
    '{"id":"fifo","prompt":"Answer only A, B, C, or D. FIFO means A Stack B Queue C Tree D Heap","answer":"B","answer_regex":"\\\\b([ABCD])\\\\b"}',
]
Path("local-mini.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
```

## D. Create lab-local-small.yaml

Use the included config:

```bash
examples/lab_local_small.yaml
```

It runs `llama3.2:3b`, `qwen3:latest`, `fallback`, `self_moa`, `semantic_vote`, `parallel_synthesis`, and `uncertainty_cascade`.

## E. Validate the Lab

PowerShell and Bash:

```bash
openfusion lab validate examples/lab_local_small.yaml
openfusion lab engine-plan examples/lab_local_small.yaml
```

## F. Run the Lab

PowerShell:

```powershell
openfusion lab run examples/lab_local_small.yaml --out results.local.json
```

Bash:

```bash
openfusion lab run examples/lab_local_small.yaml --out results.local.json
```

## G. Read the Baseline and Fusion Comparison

```bash
openfusion lab recommend results.local.json
```

Read the single-model baseline table first. Then compare each strategy against fallback and the best single-model baseline.

## H. Export CSV/JSON Result

The built-in export writes normalized JSON:

```bash
openfusion lab export results.local.json --out result-card.json
```

For CSV, parse `baselines`, `strategies`, and `strategy_comparisons` from the JSON with Python or your spreadsheet tool.

## I. Interpret Whether Fusion Helped

For simple MCQ and short exact-answer tasks, fusion may not help. A useful report says whether each strategy improved or regressed in accuracy, latency, calls, and tokens.

Always compare against each base model and the best single-model baseline. Always report latency, calls, and tokens. Do not claim benchmark improvement from a small smoke test.

## J. Run a Second Open-Ended Synthesis Test

```bash
openfusion lab validate examples/lab_open_ended_synthesis.yaml
openfusion lab run examples/lab_open_ended_synthesis.yaml --out results.synthesis.json
openfusion lab recommend results.synthesis.json
```

For open-ended synthesis, use pairwise or rubric evaluation when exact regex checks are too weak.

## K. Troubleshooting

- If Ollama is unreachable, confirm `ollama serve` is running and `http://127.0.0.1:11434/v1` is available.
- If a Windows-created JSONL file fails elsewhere, rewrite it with Python using `encoding="utf-8"`.
- If fusion is slower, check call counts and token totals before interpreting accuracy.
- If fusion does not help, report the negative result plainly.
- If recommendations look objective-specific, read the accuracy, latency, efficiency, and balanced sections separately.
