# OpenFusion Lab

OpenFusion Lab is a local experiment runner for comparing single-model and fusion strategies under the same dataset, model set, hardware, and call budget.

It does not launch model servers. Start engines manually, validate the lab file, run the experiment, then inspect the result card.

## Workflow

1. Choose OpenAI-compatible engines and models in `lab.yaml`.
2. Start each engine manually.
3. Generate a runtime config:

```bash
openfusion lab generate-config examples/lab_llama_gptoss.yaml --out openfusion.lab.generated.yaml
```

4. Validate the lab:

```bash
openfusion lab validate examples/lab_llama_gptoss.yaml
```

5. Run strategies:

```bash
openfusion lab run examples/lab_llama_gptoss.yaml --out results.json
```

6. Read recommendations:

```bash
openfusion lab recommend results.json
```

7. Export a normalized result card:

```bash
openfusion lab export results.json --out result-card.json
```

Result cards use `schema_version: openfusion-lab-result-v1` and omit secrets, provider headers, raw prompts, and raw references by default. Future result cards may support public leaderboard workflows.

## Dataset Format

Chat-style examples:

```json
{"id":"mmlu-mini-001","messages":[{"role":"user","content":"Answer only A, B, C, or D..."}],"answer":"C","answer_regex":"\\b([ABCD])\\b"}
```

Prompt-style examples:

```json
{"id":"math-001","prompt":"Return only the answer: 2 + 2","reference":"4"}
```

The evaluator supports exact match and regex extraction through `answer_mode: exact_or_regex`.

## Model Discovery

`openfusion lab search-models` queries the Hugging Face model API only when called. It does not require a token, download models, or use `trust_remote_code`.

```bash
openfusion lab search-models --query llama --limit 5 --sort downloads
```

## Engine Guidance

Use:

```bash
openfusion lab engine-plan examples/lab_llama_gptoss.yaml
```

The command prints manual launch guidance for Ollama, vLLM, and TGI. OpenFusion Lab v0.5.0 does not auto-launch engines.
