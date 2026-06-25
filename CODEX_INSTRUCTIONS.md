# Codex maintenance instructions for OpenFusion v0.2

## Project identity

OpenFusion is an open-source, OpenAI-compatible multi-model orchestration and fusion runtime. It is not weight-level model merging, a general enterprise gateway replacement, or a claim to reproduce proprietary reinforcement-learning orchestration.

Keep the existing package layout:

- `src/openfusion/` — runtime code;
- `tests/` — offline pytest coverage using fake providers;
- `docs/` — architecture, research, migration, security, and evaluation;
- `examples/` — clients and sample data;
- `.github/workflows/ci.yml` — CI.

## Required invariants

- Never add API keys, tokens, private endpoints, or `.env`.
- Tests must not call external APIs.
- Preserve OpenAI-compatible `/v1/chat/completions` and `/v1/models` shapes.
- Preserve direct routes: `provider/{provider-name}/{configured-model}`.
- Preserve the legacy `panel_judge` alias.
- Every multi-call workflow must obey `max_total_calls`.
- Do not expose hidden chain-of-thought; traces contain operational metadata only.
- Model-generated plans must remain constrained to enabled providers and built-in strategies.

## Development commands — Windows PowerShell

```powershell
cd C:\Users\User\openfusion
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ruff check src tests
pytest -q
```

## Development commands — Linux/macOS

```bash
cd ~/openfusion
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check src tests
pytest -q
```

## Before finishing a Codex task

1. Read the affected implementation and tests before editing.
2. Make the smallest coherent change.
3. Add or update offline tests.
4. Run compilation, Ruff, and pytest.
5. Review `git diff --check` and `git status --short`.
6. Summarize behavior changes, compatibility risks, and exact commands run.
