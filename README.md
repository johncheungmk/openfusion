<p align="center">
  <img src="assets/openfusion-banner.svg" alt="OpenFusion — open-source multi-model orchestration and fusion runtime" width="100%">
</p>

# OpenFusion

**OpenFusion v0.2** is an open-source, OpenAI-compatible runtime for combining local and cloud language models through transparent inference-time workflows.

It supports simple routing, but its main purpose is broader: generate independent solutions, vote or rank them, synthesize complementary evidence, run critique–revision, and execute bounded multi-layer refinement. Providers may be Ollama, LM Studio, vLLM, llama.cpp server, LiteLLM, OpenAI, OpenRouter, or any service exposing an OpenAI-compatible `/v1/chat/completions` endpoint.

OpenFusion is **not** weight-level model merging, and v0.2 is **not** a trained reinforcement-learning orchestrator equivalent to Sakana Fugu. It is a readable, configurable foundation for experimenting with multi-model test-time computation.

## What v0.2 adds

- Independent best-of-N sampling and selection.
- Majority and provider-weighted consensus voting.
- Generative parallel synthesis; legacy `panel_judge` remains an alias.
- Critic → reviser workflows with distinct model roles.
- Mixture-of-agents-style layered refinement.
- Adaptive planning using transparent heuristics or an optional constrained model planner.
- A hard per-request model-call budget.
- Public workflow plans and execution traces without hidden chain-of-thought.
- A JSONL exact-match evaluation command.
- Clear timeout errors for slow local models.

## Model gateway versus fusion runtime

A gateway such as LiteLLM focuses on provider access, keys, routing, budgets, load balancing, and observability. OpenFusion focuses on what happens **after one request may involve several model calls**.

A useful production arrangement is:

```text
Application / agent / RAG service
              |
              v
         OpenFusion
  deliberation and synthesis
              |
              v
 LiteLLM or another AI gateway
              |
              v
 local and cloud model providers
```

OpenFusion can also call providers directly without LiteLLM.

## Strategies

| Strategy | Behavior | Typical calls |
|---|---|---:|
| `fallback` | Try providers in order and return the first success. | 1 to panel size |
| `parallel_synthesis` | Independent drafts, then a synthesizer writes a new answer. | drafts + 1 |
| `best_of_n` | Generate alternatives and select one unchanged answer. | candidates + 1 |
| `majority_vote` | Normalize concise answers and choose the largest exact consensus group. | candidates |
| `weighted_vote` | As above, but sum provider weights. | candidates |
| `critique_revision` | Independent drafts → critic feedback → new revised answer. | drafts + 2 |
| `layered_refinement` | Independent layer → one or more refinement layers → synthesis. | multiple layers + 1 |
| `adaptive` | Choose a bounded workflow using heuristics or an optional model planner. | depends on plan |

Voting is most suitable for concise, multiple-choice, classification, or regex-extractable answers. Open-ended prose usually benefits more from synthesis or critique–revision.

## Quick start

### Windows PowerShell

```powershell
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

Copy-Item .env.example .env
openfusion init --path openfusion.yaml
```

### Linux / macOS Bash

```bash
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

cp .env.example .env
openfusion init --path openfusion.yaml
```

## The configuration file you must edit

The command below creates the runtime configuration file:

```text
openfusion.yaml
```

```bash
openfusion init --path openfusion.yaml
```

After running it, edit **`openfusion.yaml` in the repository root**. Do not edit `.env.example`; `.env` contains secrets, while `openfusion.yaml` contains provider names, model names, roles, and workflow settings.

### Local-only Ollama setup

First inspect or install an Ollama model.

PowerShell:

```powershell
ollama list
ollama pull llama3.2:3b
notepad .\openfusion.yaml
```

Linux / macOS:

```bash
ollama list
ollama pull llama3.2:3b
nano openfusion.yaml
```

Make the `model` value in `openfusion.yaml` exactly match `ollama list`:

```yaml
providers:
  - name: local-ollama
    type: openai_compatible
    enabled: true
    base_url: http://localhost:11434/v1
    api_key_env:
    model: llama3.2:3b
    timeout_seconds: 300
    weight: 1.0
    headers: {}

fusion:
  default_strategy: parallel_synthesis
  panel: [local-ollama]
  judge_provider: local-ollama
  critic_provider: local-ollama
  reviser_provider: local-ollama
  planner_provider:

  max_parallel: 2
  max_total_calls: 8
  samples_per_provider: 1
  refinement_rounds: 1

  temperature: 0.2
  judge_temperature: 0.1
  critique_temperature: 0.1
  max_tokens: 256

  require_at_least_successes: 1
  include_candidate_outputs: true
  include_workflow_outputs: true
  judge_candidate_max_chars: 4000
  transcript_max_chars: 12000
  adaptive_use_model_planner: false

server:
  host: 127.0.0.1
  port: 8000
  api_key_env: OPENFUSION_API_KEY
```

For large CPU-only models, `timeout_seconds: 300` and `max_tokens: 128` or `256` are sensible starting points. A panel workflow can call the same model more than once, so it is naturally slower than direct routing.

### Cloud plus local example

Put API keys in `.env`:

```dotenv
OPENAI_API_KEY=replace-me
OPENFUSION_API_KEY=replace-with-a-long-random-token
```

Then enable the cloud provider in `openfusion.yaml`:

```yaml
providers:
  - name: local-ollama
    type: openai_compatible
    enabled: true
    base_url: http://localhost:11434/v1
    api_key_env:
    model: llama3.2:3b
    timeout_seconds: 300
    weight: 1.0

  - name: cloud-openai
    type: openai_compatible
    enabled: true
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    model: gpt-4.1-mini
    timeout_seconds: 120
    weight: 1.2

fusion:
  default_strategy: critique_revision
  panel: [local-ollama, cloud-openai]
  judge_provider: cloud-openai
  critic_provider: cloud-openai
  reviser_provider: cloud-openai
  max_total_calls: 8
  max_tokens: 512
```

## Validate and start

PowerShell:

```powershell
openfusion providers --config .\openfusion.yaml
openfusion strategies
pytest -q
ruff check src tests
openfusion serve --config .\openfusion.yaml --port 8000
```

Linux / macOS:

```bash
openfusion providers --config ./openfusion.yaml
openfusion strategies
pytest -q
ruff check src tests
openfusion serve --config ./openfusion.yaml --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Test direct provider routing first

Direct routing proves that the underlying model works before you add multi-call orchestration.

### Windows PowerShell

```powershell
$headers = @{
  "Authorization" = "Bearer replace-with-a-long-random-token"
  "Content-Type"  = "application/json"
}

$body = @{
  model = "provider/local-ollama/llama3.2:3b"
  messages = @(
    @{
      role = "user"
      content = "Give a RAG deployment plan in three short bullets."
    }
  )
  max_tokens = 128
} | ConvertTo-Json -Depth 10

$response = Invoke-RestMethod `
  -Uri "http://localhost:8000/v1/chat/completions" `
  -Method Post `
  -Headers $headers `
  -Body $body

$response.choices[0].message.content
```

### Linux / macOS

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-a-long-random-token' \
  -d '{
    "model": "provider/local-ollama/llama3.2:3b",
    "messages": [
      {"role": "user", "content": "Give a RAG deployment plan in three short bullets."}
    ],
    "max_tokens": 128
  }'
```

## Run orchestration strategies

CLI:

```bash
openfusion chat "Compare two RAG deployment designs." \
  --config openfusion.yaml \
  --strategy critique_revision \
  --show-trace
```

OpenAI-compatible API:

```json
{
  "model": "openfusion/critique-revision",
  "messages": [
    {"role": "user", "content": "Compare two RAG deployment designs."}
  ],
  "max_tokens": 256,
  "fusion_panel": ["local-ollama", "cloud-openai"],
  "fusion_critic": "cloud-openai",
  "fusion_reviser": "cloud-openai",
  "fusion_max_total_calls": 8
}
```

Available model IDs include:

```text
openfusion/adaptive
openfusion/parallel-synthesis
openfusion/panel-judge            # legacy alias
openfusion/critique-revision
openfusion/layered-refinement
openfusion/best-of-n
openfusion/majority-vote
openfusion/weighted-vote
openfusion/fallback
provider/{provider-name}/{configured-model}
```

## Adaptive planning

Adaptive mode is deliberately constrained. It can choose only enabled providers and built-in strategies, and every request has a hard call budget.

By default it uses transparent heuristics and spends no planning model call:

```bash
openfusion plan "Review three RAG architectures and recommend one." \
  --config openfusion.yaml
```

To use a model-generated JSON plan, configure `planner_provider`, set `adaptive_use_model_planner: true`, or pass `--model-planner`. This consumes one call before workflow execution. Invalid plans fall back to heuristics.

## Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="replace-with-a-long-random-token",
)

completion = client.chat.completions.create(
    model="openfusion/layered-refinement",
    messages=[{"role": "user", "content": "Compare vLLM and Ollama."}],
    max_tokens=256,
    extra_body={
        "fusion_samples_per_provider": 1,
        "fusion_refinement_rounds": 1,
        "fusion_max_total_calls": 8,
    },
)
print(completion.choices[0].message.content)
```

## Evaluation

Do not assume that more agents always improve a task. Measure quality, latency, and cost on a dataset representative of your use case.

Example JSONL:

```jsonl
{"id":"math-1","prompt":"Return only the answer: 2 + 2","reference":"4"}
{"id":"mcq-1","prompt":"Final answer only. A) red B) blue","reference":"B","answer_regex":"(?:Final answer|Answer):\\s*([A-D])"}
```

Run:

```bash
openfusion evaluate examples/eval_sample.jsonl \
  --config openfusion.yaml \
  --strategy weighted_vote \
  --output evaluation-report.json
```

The built-in evaluator is intentionally simple exact match. Add domain-specific graders before publishing performance claims.

## Response metadata

Every response includes an `openfusion` object containing:

- the executed strategy;
- a bounded orchestration plan;
- candidate status and optional candidate text;
- a public execution trace with stages, providers, latency, and errors;
- usage summed across model calls;
- optional public critique or vote summary.

It does not request or expose hidden chain-of-thought.

## Security and cost notes

- Keep secrets in `.env` or the process environment, never YAML.
- Leave the default host at `127.0.0.1` for local use.
- Set a strong `OPENFUSION_API_KEY` before binding to `0.0.0.0`.
- Set `include_candidate_outputs: false` when intermediate model text is sensitive.
- Set `include_workflow_outputs: false` to suppress critique and vote summaries.
- Use `max_total_calls` to cap per-request model calls.
- Model-generated planning cannot invent executable tools or arbitrary code paths.

See [docs/SECURITY.md](docs/SECURITY.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/LITELLM.md](docs/LITELLM.md).

## Research positioning

OpenFusion v0.2 is inspired by self-consistency, LLM-Blender, Mixture-of-Agents, multi-agent debate, OpenRouter Fusion, and Sakana's orchestration research. It implements practical inference workflows, not proprietary training methods or weight merging. See [docs/RESEARCH.md](docs/RESEARCH.md).

## Migration from v0.1

- `panel_judge` still works but is normalized to `parallel_synthesis` in metadata.
- New config fields have defaults, so an existing valid v0.1 config should continue to load.
- `/health` now includes version and strategy names.
- More strategy model IDs appear in `/v1/models`.
- Provider timeout errors are now explicit.

See [docs/MIGRATION_V2.md](docs/MIGRATION_V2.md).

## Development and Codex CLI

The repository includes `AGENTS.md`, which Codex CLI discovers automatically when it starts from the repository root. `CODEX_INSTRUCTIONS.md` provides a longer maintenance checklist, and `docs/CODEX_V2_UPGRADE.md` contains the reproducible v0.1-to-v0.2 upgrade prompt.

Interactive Codex session:

```bash
cd openfusion
codex
```

One-shot review:

```bash
codex "Review this OpenFusion v0.2 tree, run the required checks in AGENTS.md, and fix only verified issues."
```

Manual development checks:

```bash
python -m pip install -e '.[dev]'
python -m compileall -q src tests
ruff check src tests
pytest -q
python -m build
git diff --check
```

## License

MIT
