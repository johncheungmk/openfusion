# Codex maintenance instructions for OpenFusion

Use these instructions when working in this repository with Codex CLI.

## Project scope

OpenFusion is an API-level model-fusion gateway for OpenAI-compatible chat-completion
providers. Do not rewrite the project from scratch. Keep changes focused on the existing
package structure:

- `src/openfusion/` for package code.
- `tests/` for pytest coverage.
- `docs/` for architecture and security notes.
- `examples/` for runnable client examples.
- `.github/workflows/ci.yml` for CI.

OpenFusion is not a weight-level model-merging tool.

## Development commands

```powershell
cd C:\Users\User\openfusion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check src tests
pytest -q
```

## Local run

```powershell
Copy-Item .env.example .env
Copy-Item config.example.yaml openfusion.yaml
openfusion providers --config openfusion.yaml
openfusion chat "Explain model fusion in one paragraph" --config openfusion.yaml
openfusion serve --config openfusion.yaml --port 8000
```

The example config starts with Ollama enabled. Enable additional providers only after
setting their environment variables.

## Quality expectations

- Do not add API keys, tokens, private endpoints, or secrets to the repository.
- Keep provider API keys environment-based via `api_key_env`.
- Tests must not call external model APIs.
- Run `ruff check src tests` and `pytest -q` before finishing changes.
- Preserve the OpenAI-compatible `/v1/chat/completions` response shape.
