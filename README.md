# OpenFusion

![OpenFusion — open-source multi-model orchestration and fusion runtime](assets/openfusion-banner.svg)

OpenFusion v0.6.1 is an open-source, OpenAI-compatible runtime for multi-model orchestration, model fusion, and local benchmark experiments.

It supports local and cloud models through OpenAI-compatible APIs, including Ollama, vLLM, LM Studio, LiteLLM, OpenRouter-compatible endpoints, and other `/v1/chat/completions` servers.

OpenFusion helps users test whether fusion actually improves results by comparing
strategies against fallback when configured and every direct model, while reporting accuracy uncertainty,
latency, calls, tokens, optional complete-cost estimates, and improvement or regression.

## Benchmark evidence

The [complete evidence overview](docs/research/benchmark-evidence.md) includes positive test-guided results and the
later negative diversity controls. Start with a strong single model and fresh same-model
sampling; use a mixed workflow when a held-out comparison justifies it.

## Documentation

The [prospective MoA deployment study](docs/research/prospective-moa-deployment.md)
tests one preselected workflow on 500 previously unused programming tasks, with a
registered comparison against Qwen repair and explicit accuracy and resource criteria.

Full documentation is available at:

<https://johncheungmk.github.io/openfusion/>

The documentation includes:

- installation and quickstart guides;
- Ollama setup;
- OpenFusion Lab;
- baseline comparison;
- strategy guides;
- configuration reference;
- deployment notes;
- troubleshooting.

## Highlights

- OpenAI-compatible `/v1/chat/completions` API server
- Local and cloud provider support
- [Kev decision-model selection](docs/strategies/decision-select.md) through the System One API
- Order-balanced LLM judging with explicit abstention and position-consistency telemetry
- Wilson confidence intervals, end-to-end latency, and optional token-price cost accounting
- Lab panel-complementarity reports with oracle and all-model co-failure diagnostics
- Hard administrator call ceilings with bounded request scheduling
- Self-MoA and Self-MoA-Seq
- Semantic voting
- Parallel synthesis
- Pairwise rank fusion
- Uncertainty cascade
- Critique-revision
- Layered refinement
- OpenFusion Lab result-card evaluation
- Baseline and best-single-model comparison
- PowerShell and Linux/macOS cookbook documentation

## Quick start

### Windows PowerShell

```powershell
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

openfusion --help
openfusion lab --help
```

### Linux / macOS

```bash
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

openfusion --help
openfusion lab --help
```

## First local benchmark

The following example uses two local Ollama models and OpenFusion Lab.

```bash
ollama pull llama3.2:3b
ollama pull qwen3:latest

openfusion lab validate examples/lab_local_small.yaml
openfusion lab run examples/lab_local_small.yaml --out results-local-small.json
openfusion lab recommend results-local-small.json
```

OpenFusion Lab will report:

- single-model baselines;
- best-observed single-model baseline on the evaluated split;
- fusion strategy comparison;
- accuracy delta versus fallback;
- accuracy delta versus best single model;
- latency, call, and token ratios;
- objective-specific recommendations with preserved latency/call preferences.

For the full cookbook, see:

<https://johncheungmk.github.io/openfusion/lab/local-benchmark-cookbook/>

## Why baseline comparison matters

Fusion is not automatically better. A useful fusion strategy should be compared against:

1. each original base model;
2. the best single-model baseline;
3. fallback routing;
4. equal-budget alternatives such as Self-MoA.

OpenFusion Lab is designed to make negative results visible. If fusion does not improve accuracy, it should say so clearly.

## Strategy examples

| Strategy | Typical use |
|---|---|
| `fallback` | Operational baseline and failure recovery |
| `self_moa` | Sample one strong model multiple times, then select or synthesize |
| `self_moa_seq` | Sequential/batched Self-MoA for many samples |
| `semantic_vote` | Short-answer or MCQ-style tasks |
| `parallel_synthesis` | Open-ended synthesis tasks |
| `pairwise_rank_fuse` | Candidate ranking followed by fusion |
| `uncertainty_cascade` | Latency/cost-sensitive escalation |
| `critique_revision` | Draft, critique, and revision workflows |
| `layered_refinement` | MoA-style refinement layers |
| `adaptive` | Constrained heuristic/model-generated workflow planning |

Full strategy documentation is available at:

<https://johncheungmk.github.io/openfusion/concepts/strategies/>

## Positioning

OpenFusion is Mixture-of-Agents-inspired, but it is **not** the original Together AI Mixture-of-Agents implementation.

OpenFusion is **not** Sakana Fugu or a trained learned orchestrator.

OpenFusion is **not** a replacement for LiteLLM. LiteLLM is better suited for provider management, keys, budgets, and routing. OpenFusion focuses on explicit fusion workflows, evaluation, and local experiments.

OpenFusion can work with LiteLLM by treating a LiteLLM proxy as an OpenAI-compatible provider.

## Repository structure

```text
src/openfusion/      Python package
tests/               Offline tests
docs/                MkDocs documentation source
examples/            Example configs and datasets
assets/              Banner and static assets
.github/workflows/   CI and documentation deployment
```

## Development checks

```bash
python -m compileall -q src tests
ruff check src tests
pytest -q --basetemp .pytest-tmp
python -m build
mkdocs build --strict
```

## Documentation site

The documentation site is built with MkDocs Material and published through GitHub Pages.

Local preview:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Then open:

<http://127.0.0.1:8000>

## License

MIT

## Citation

Research users can cite the current software release using
[`CITATION.cff`](CITATION.cff). Benchmark claims should also publish the dataset/config
hashes, exact providers and model identifiers, grader, budget, uncertainty, and negative
results where applicable.
