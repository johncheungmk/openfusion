# `lab.yaml`

`lab.yaml` defines a local model-fusion experiment.

Use it when you want to compare base models and fusion strategies.

```yaml
experiment:
  name: local-small-model-fusion-test
  max_examples: 10
  max_total_calls: 4
  max_tokens: 16
  temperature: 0.2

dataset:
  path: examples/minibench_local_10.jsonl
  name: local-mini-mmlu
  split: smoke
  answer_mode: exact_or_regex

engines:
  - name: ollama-local
    type: ollama
    base_url: http://127.0.0.1:11434/v1
    launch: manual

models:
  - provider_name: ollama-llama32-3b
    engine: ollama-local
    model: llama3.2:3b
    weight: 1.0
    timeout_seconds: 180

  - provider_name: ollama-qwen3
    engine: ollama-local
    model: qwen3:latest
    weight: 1.2
    timeout_seconds: 240

strategies:
  - name: fallback
  - name: self_moa
  - name: semantic_vote
  - name: parallel_synthesis
  - name: uncertainty_cascade
```

Validate it:

```powershell
openfusion lab validate lab.yaml
```
