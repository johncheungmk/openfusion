<p align="center">
  <img src="assets/openfusion-banner.svg" alt="OpenFusion — open-source multi-model orchestration and fusion runtime" width="100%">
</p>

# OpenFusion

**OpenFusion v0.4.0** is an open-source, OpenAI-compatible runtime for combining local and cloud language models through transparent inference-time workflows. It supports simple routing, but its main purpose is broader: generate independent solutions, vote or rank them, synthesize complementary evidence, run critique–revision, and execute bounded multi-layer refinement.

Providers may be Ollama, LM Studio, vLLM, llama.cpp server, LiteLLM, OpenAI, OpenRouter, or any service exposing an OpenAI-compatible `/v1/chat/completions` endpoint.

OpenFusion is **not** the original Together AI Mixture-of-Agents implementation, not weight-level model merging, not a trained Sakana Fugu-style learned orchestrator, and not a replacement for LiteLLM. It is a transparent, self-hostable, local-model-friendly runtime for OpenRouter-Fusion-like and MoA-inspired experiments where the workflow is explicit, bounded, and inspectable.

> Do not claim benchmark gains from OpenFusion, MoA, Self-MoA, voting, ranking, or cascading strategies without task-specific evaluation. More agents and more calls can improve, match, or degrade results depending on model quality, task type, prompts, and budget.

---

## Highlights in v0.4.0

- First-class `self_moa` and `self_moa_seq` strategies.
- Role-diverse panel prompts and optional structured synthesis analysis.
- `pairwise_rank_fuse` for rank-then-fuse workflows.
- `semantic_vote` for concise exact or semantically equivalent answers.
- `uncertainty_cascade` for confidence-aware escalation.
- Independent best-of-N sampling and selection.
- Majority and provider-weighted consensus voting.
- Generative parallel synthesis; legacy `panel_judge` remains an alias.
- Critic → reviser workflows with distinct model roles.
- Mixture-of-agents-style layered refinement.
- Adaptive planning using transparent heuristics or an optional constrained model planner.
- Hard per-request model-call budget.
- Public workflow plans and execution traces without hidden chain-of-thought.
- JSONL evaluation with equal-budget comparison, latency, calls, token totals, and optional LLM pairwise/rubric grading.
- Clear timeout errors for slow local models.

---

## Positioning

| System | Open-source implementation | Self-hostable | Local model support | OpenAI-compatible gateway | Learned orchestrator | Configurable workflow strategies | Transparent traces | Built-in evaluation | Provider/key management focus | Intended role |
|---|---|---|---|---|---|---|---|---|---|---|
| OpenFusion | Yes | Yes | Yes | Yes | No | Yes | Yes | Yes | Basic | Transparent orchestration runtime for local/cloud fusion experiments |
| OpenRouter Fusion | No, managed feature | No | No direct local hosting | Via OpenRouter API | No public learned orchestrator claim | Limited by managed service | Structured analysis surfaced by service | No local built-in evaluator | Managed provider marketplace | Hosted multi-model deliberation product |
| Sakana Fugu | No public implementation | No | No direct local hosting | Product/model endpoint | Yes, positioned as learned orchestration | Not user-configurable as local workflows | Not a local trace runtime | No local built-in evaluator | Not gateway focused | Learned model orchestration system |
| LiteLLM | Yes | Yes | Yes, through configured providers | Yes | No | Routing/gateway policies, not MoA workflows | Gateway logs/observability | No MoA evaluation harness | Strong | Provider gateway, key management, budgets, routing, observability |

---

## Model gateway versus fusion runtime

A gateway such as LiteLLM focuses on provider access, keys, routing, budgets, load balancing, and observability. OpenFusion focuses on what happens **after one user request may involve several model calls**.

A useful production arrangement is:

```text
Application / agent / RAG service
        ↓
OpenFusion deliberation and synthesis
        ↓
LiteLLM or another AI gateway
        ↓
Local and cloud model providers
```

OpenFusion can also call providers directly without LiteLLM.

---

## Strategies

| Strategy | Behavior | Typical calls |
|---|---|---:|
| `fallback` | Try providers in order and return the first success. | 1 to panel size |
| `parallel_synthesis` | Independent drafts, then a synthesizer writes a new answer. | drafts + 1 |
| `self_moa` | Sample one provider repeatedly, then select or synthesize. | samples + 1 |
| `self_moa_seq` | Batched Self-MoA with a running selected or fused answer. | bounded by budget |
| `pairwise_rank_fuse` | Rank panel candidates by pairwise or score judging, then fuse top answers. | drafts + rank calls + 1 |
| `semantic_vote` | Group concise exact or semantically equivalent answers before voting. | candidates + optional equivalence |
| `uncertainty_cascade` | Start cheap and escalate on failure, low confidence, disagreement, or invalid format. | bounded by steps |
| `best_of_n` | Generate alternatives and select one unchanged answer. | candidates + 1 |
| `majority_vote` | Normalize concise answers and choose the largest exact consensus group. | candidates |
| `weighted_vote` | As above, but sum provider weights. | candidates |
| `critique_revision` | Independent drafts → critic feedback → new revised answer. | drafts + 2 |
| `layered_refinement` | Independent layer → one or more refinement layers → synthesis. | multiple layers + 1 |
| `adaptive` | Choose a bounded workflow using heuristics or an optional model planner. | depends on plan |

Voting is most suitable for concise, multiple-choice, classification, or regex-extractable answers. Open-ended prose usually benefits more from synthesis, ranking, critique–revision, or layered refinement.

---

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

---

## The configuration file you must edit

The runtime configuration file is:

```text
openfusion.yaml
```

Create it with:

```bash
openfusion init --path openfusion.yaml
```

After running the command, edit **`openfusion.yaml` in the repository root**.

Do **not** edit `.env.example`. Use:

- `.env` for secrets such as `OPENAI_API_KEY` and `OPENFUSION_API_KEY`;
- `openfusion.yaml` for provider names, model names, roles, strategy defaults, and workflow settings.

---

## Local-only Ollama setup

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

Make the `model` value in `openfusion.yaml` exactly match a model name returned by `ollama list`:

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

  panel_roles:
    - name: factual_checker
      instruction: Focus on factual accuracy and cite uncertainty.
    - name: edge_case_reviewer
      instruction: Focus on edge cases, failure modes, and missing assumptions.
    - name: concise_summarizer
      instruction: Produce the clearest concise answer.

  judge_provider: local-ollama
  critic_provider: local-ollama
  reviser_provider: local-ollama
  planner_provider:
  self_moa_provider: local-ollama
  ranker_provider: local-ollama
  fuser_provider: local-ollama
  vote_equivalence_provider:

  cascade_providers: [local-ollama]

  max_parallel: 2
  max_total_calls: 8
  samples_per_provider: 1
  refinement_rounds: 1

  self_moa_samples: 3
  self_moa_batch_size: 4
  self_moa_seq_carry_max_chars: 12000

  rank_top_k: 3
  pairwise_rank_max_pairs: 12
  semantic_vote_max_pairs: 12

  cascade_consistency_samples: 1
  cascade_max_steps: 3

  temperature: 0.2
  judge_temperature: 0.1
  critique_temperature: 0.1
  self_moa_temperature: 0.7

  self_moa_mode: synthesize
  pairwise_rank_mode: pairwise
  semantic_vote_mode: rule_only
  cascade_confidence_threshold: 0.75
  cascade_escalate_on_disagreement: true

  max_tokens: 256
  require_at_least_successes: 1
  include_candidate_outputs: true
  include_workflow_outputs: true
  structured_synthesis: false

  judge_candidate_max_chars: 4000
  transcript_max_chars: 12000
  vote_answer_regex:
  adaptive_use_model_planner: false

server:
  host: 127.0.0.1
  port: 8000
  api_key_env: OPENFUSION_API_KEY
```

For large CPU-only models, `timeout_seconds: 300` and `max_tokens: 128` or `256` are sensible starting points. A panel workflow can call the same model more than once, so it is naturally slower than direct routing.

---

## Cloud plus local example

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

---

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

---

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

---

## Run orchestration strategies

CLI:

```bash
openfusion chat "Compare two RAG deployment designs." \
  --config openfusion.yaml \
  --strategy critique_revision \
  --show-trace
```

OpenAI-compatible API request body:

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
openfusion/panel-judge          # legacy alias
openfusion/self-moa
openfusion/self-moa-seq
openfusion/pairwise-rank-fuse
openfusion/semantic-vote
openfusion/uncertainty-cascade
openfusion/critique-revision
openfusion/layered-refinement
openfusion/best-of-n
openfusion/majority-vote
openfusion/weighted-vote
openfusion/fallback
provider/{provider-name}/{configured-model}
```

---

## Adaptive planning

Adaptive mode is deliberately constrained. It can choose only enabled providers and built-in strategies, and every request has a hard call budget. By default it uses transparent heuristics and spends no planning model call:

```bash
openfusion plan "Review three RAG architectures and recommend one." \
  --config openfusion.yaml
```

To use a model-generated JSON plan, configure `planner_provider`, set `adaptive_use_model_planner: true`, or pass `--model-planner`. This consumes one call before workflow execution. Invalid plans fall back to heuristics.

---

## Self-MoA

`openfusion/self-moa` samples one provider multiple times, then either selects the best unchanged sample or synthesizes a new answer. Configure:

- `fusion.self_moa_provider`
- `fusion.self_moa_samples`
- `fusion.self_moa_temperature`
- `fusion.self_moa_mode`

If the provider is unset, OpenFusion uses `judge_provider`, then the first panel provider.

`openfusion/self-moa-seq` batches the samples and carries forward a running best or fused answer for long candidate sets. Tune `fusion.self_moa_batch_size` and `fusion.self_moa_seq_carry_max_chars` for large jobs. Both strategies obey `max_total_calls` and expose public trace metadata without hidden chain-of-thought.

Request override example:

```json
{
  "model": "openfusion/self-moa",
  "messages": [
    {"role": "user", "content": "Solve this carefully."}
  ],
  "fusion_self_moa_provider": "local-ollama",
  "fusion_self_moa_samples": 4,
  "fusion_self_moa_mode": "synthesize",
  "fusion_max_total_calls": 5
}
```

---

## Role-diverse panels and structured synthesis

Set `fusion.panel_roles` to give panel calls complementary public instructions such as factual checking, edge-case review, or concise summarization. Roles cycle when there are more panel calls than configured roles. Public traces include the role name only; role instructions are prompts, not hidden reasoning.


Set `fusion.structured_synthesis: true` or request `"fusion_structured_synthesis": true` to ask synthesizers to return parseable public sections:

- `consensus_points`
- `contradictions`
- `unique_insights`
- `missing_information`
- `final_answer`

Parsed sections appear in `openfusion.workflow_outputs` when `include_workflow_outputs` is true. If parsing fails, OpenFusion returns the plain synthesized answer.

---

## Ranking and semantic voting

`openfusion/pairwise-rank-fuse` generates panel candidates, ranks them with `ranker_provider`, then fuses the top `rank_top_k` candidates with `fuser_provider` or `judge_provider`.

- `pairwise_rank_mode: pairwise` performs bounded candidate comparisons up to `pairwise_rank_max_pairs`.
- `pairwise_rank_mode: score` asks for parseable JSON scores in one ranker call.
- If ranking output cannot be parsed, OpenFusion preserves candidate order and still attempts synthesis.

`openfusion/semantic-vote` keeps `majority_vote` and `weighted_vote` unchanged while adding a separate voting strategy.

- `semantic_vote_mode: rule_only` uses normalized exact voting.
- `semantic_vote_mode: llm_equivalence` asks `vote_equivalence_provider` whether concise answers are semantically the same, capped by `semantic_vote_max_pairs` and `max_total_calls`.
- Without an equivalence provider, it falls back to rule-only grouping.

---

## Uncertainty cascade

`openfusion/uncertainty-cascade` starts with `fusion.cascade_providers[0]`, or the first panel provider when no cascade list is configured. Each provider returns a concise answer plus a public confidence score.

OpenFusion escalates to the next provider on:

- provider failure;
- confidence below `cascade_confidence_threshold`;
- sample disagreement when `cascade_consistency_samples` is greater than one;
- invalid response format.

The cascade is bounded by `cascade_max_steps` and `max_total_calls`. Public trace entries show the provider attempted, confidence, disagreement status, and escalation reason without hidden chain-of-thought.

---

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

---

## Evaluation

Do not assume that more agents always improve a task. Measure quality, latency, and cost on a dataset representative of your use case.

Example JSONL:

```jsonl
{"id":"math-1","prompt":"Return only the answer: 2 + 2","reference":"4"}
{"id":"mcq-1","prompt":"Final answer only. A) red B) blue","reference":"B","answer_regex":"(?:Final answer|Answer):\\s*([A-D])"}
```

Run a single strategy:

```bash
openfusion evaluate examples/eval_sample.jsonl \
  --config openfusion.yaml \
  --strategy weighted_vote \
  --output evaluation-report.json
```

Run an equal-budget comparison:

```bash
openfusion evaluate examples/eval_moa_sample.jsonl \
  --config openfusion.yaml \
  --compare-strategies fallback,self_moa,parallel_synthesis \
  --max-total-calls 6 \
  --output comparison-report.json
```

Reports include accuracy, win/tie/loss rates versus the fallback baseline, call counts, latency percentiles, token totals, cost placeholder, accuracy per call, accuracy per 1k tokens, and strategy failures.

### Recommended baselines for papers

For any paper, blog post, or benchmark claim, compare against:

- single best model;
- direct provider route or `fallback`;
- `best_of_n`;
- `self_moa`;
- mixed MoA / `layered_refinement`;
- `pairwise_rank_fuse`;
- `semantic_vote` for short-answer tasks;
- `uncertainty_cascade` for cost-sensitive tasks.

Use equal `max_total_calls` budgets where possible, and report latency, token use, call counts, and failures alongside quality.

The default grader is exact match. Optional `--grader llm_pairwise` and `--grader llm_rubric --grader-provider PROVIDER` are useful for qualitative inspection, but LLM judge results are not ground truth. Treat them as noisy model outputs and validate important claims with task-specific graders or human review.

---

## Response metadata

Every response includes an `openfusion` object containing:

- the executed strategy;
- a bounded orchestration plan;
- candidate status and optional candidate text;
- a public execution trace with stages, providers, latency, and errors;
- usage summed across model calls;
- optional public critique, vote summary, ranking result, cascade decision, or structured synthesis sections.

It does not request or expose hidden chain-of-thought.

---

## Security and cost notes

- Keep secrets in `.env` or the process environment, never YAML.
- Leave the default host at `127.0.0.1` for local use.
- Set a strong `OPENFUSION_API_KEY` before binding to `0.0.0.0`.
- Set `include_candidate_outputs: false` when intermediate model text is sensitive.
- Set `include_workflow_outputs: false` to suppress critique, vote summaries, rankings, cascade decisions, and structured synthesis sections.
- Use `max_total_calls` to cap per-request model calls.
- Model-generated planning cannot invent executable tools or arbitrary code paths.

See [docs/SECURITY.md](docs/SECURITY.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/LITELLM.md](docs/LITELLM.md).

---

## Research positioning

OpenFusion v0.4.0 is inspired by self-consistency, LLM-Blender, Mixture-of-Agents, Self-MoA, multi-agent debate, OpenRouter Fusion, routing/cascade research, and Sakana-style orchestration research. It implements practical inference workflows, not proprietary training methods or weight merging.

OpenFusion is useful for studying questions such as:

- When does Self-MoA outperform mixed-model MoA?
- When does ranking plus fusion beat simple synthesis?
- When is semantic voting better than exact voting?
- When does uncertainty cascading save cost without hurting quality?
- How much latency and token usage does each strategy add?

See [docs/RESEARCH.md](docs/RESEARCH.md), [docs/MOA.md](docs/MOA.md), and [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Migration notes

- `panel_judge` still works but is normalized to `parallel_synthesis` in response metadata.
- Earlier valid configs should continue to load because new config fields have defaults.
- `/health` includes version and strategy names.
- `/v1/models` lists the expanded v0.4 strategy model IDs.
- Provider timeout errors are explicit.
- v0.4 adds Self-MoA, sequential Self-MoA, pairwise ranking, semantic voting, uncertainty cascade, role-diverse prompts, structured synthesis, and equal-budget evaluation.

See [docs/MIGRATION_V2.md](docs/MIGRATION_V2.md) for earlier migration details.

---

## Development and Codex CLI

The repository includes `AGENTS.md`, which Codex CLI discovers automatically when it starts from the repository root. `CODEX_INSTRUCTIONS.md` provides a longer maintenance checklist.

Interactive Codex session:

```bash
cd openfusion
codex
```

One-shot review:

```bash
codex "Review this OpenFusion repository, run the required checks in AGENTS.md, and fix only verified issues."
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

On Windows, if pytest cannot write to the default temp directory, use a local temp directory:

```powershell
pytest -q --basetemp .pytest-tmp
```

---

## License

MIT
