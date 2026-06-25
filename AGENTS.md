# OpenFusion repository instructions

## Project identity

OpenFusion is an open-source, OpenAI-compatible multi-model orchestration and fusion runtime. It is not weight-level model merging, a replacement for every enterprise AI gateway, or a claim to reproduce proprietary reinforcement-learning orchestrators.

## Required invariants

- Keep runtime code under `src/openfusion/` and offline tests under `tests/`.
- Never add API keys, tokens, private endpoints, `.env`, or `openfusion.yaml`.
- Tests must not call external APIs.
- Preserve `/v1/chat/completions`, `/v1/models`, direct `provider/{provider-name}/{model}` routes, and the legacy `panel_judge` alias.
- Every multi-call workflow must obey `max_total_calls`.
- Do not request or expose hidden chain-of-thought. Public traces contain operational metadata only.
- Constrain model-generated plans to enabled providers and built-in strategies. Never execute planner-generated code, shell commands, URLs, or arbitrary tools.
- Add or update tests for every behavior change.

## Validation

Run before completing a change:

```bash
python -m compileall -q src tests
ruff check src tests
pytest -q
python -m build
git diff --check
```

Summarize behavior changes, compatibility risks, and the exact commands run.
