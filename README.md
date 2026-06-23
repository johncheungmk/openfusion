# OpenFusion

<p align="center">
  <img src="assets/openfusion-banner.svg" alt="OpenFusion banner: open-source API-level model fusion gateway" width="100%">
</p>

OpenFusion is an open-source, OpenAI-compatible **model-fusion gateway** for local and cloud LLMs.

It lets you combine multiple models such as Ollama, LM Studio, vLLM, OpenAI, OpenRouter, or any OpenAI-compatible API. The first release focuses on **API-level fusion**: run several models in parallel, compare their answers, and synthesize a stronger final answer using a judge model.

> OpenFusion is not weight-level model merging. It is a practical gateway for multi-model deliberation, direct provider routing, fallback, and judge synthesis.

## Features

- OpenAI-compatible `/v1/chat/completions` API.
- Works with local and cloud OpenAI-compatible APIs.
- Direct provider routing using model IDs such as `provider/local-ollama/llama3.2:3b`.
- `panel_judge` strategy: parallel model panel + judge synthesis.
- `fallback` strategy: try providers in order until one succeeds.
- CLI for local testing.
- FastAPI server for integration with Dify, custom apps, agents, and SDK clients.
- Dockerfile and docker-compose support.
- GitHub Actions test workflow.

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
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
openfusion init --path openfusion.yaml
```

OpenFusion uses `openfusion.yaml` as the runtime configuration file. The `openfusion init --path openfusion.yaml` command creates this file from the packaged example config.

After creating `openfusion.yaml`, edit **that file** to match your local or cloud models. API keys belong in `.env` or your shell environment, not in `openfusion.yaml`.

By default the server binds to `127.0.0.1`. If `OPENFUSION_API_KEY` is unset, OpenFusion logs a warning and accepts unauthenticated local requests. Set a strong `OPENFUSION_API_KEY` before binding to `0.0.0.0` or exposing the service.

## Local-only Ollama setup

The default local setup uses Ollama through its OpenAI-compatible endpoint:

```text
http://localhost:11434/v1
```

First, make sure Ollama is installed and running.

Check installed models:

### Windows PowerShell

```powershell
ollama list
```

### Linux / macOS Bash

```bash
ollama list
```

If you do not have a small model for testing, pull one:

### Windows PowerShell

```powershell
ollama pull llama3.2:3b
```

### Linux / macOS Bash

```bash
ollama pull llama3.2:3b
```

Then edit:

```text
openfusion.yaml
```

Set the `local-ollama` provider's `model` field to exactly one of the model names shown by `ollama list`.

Example `openfusion.yaml` for local-only Ollama:

```yaml
providers:
  - name: local-ollama
    type: openai_compatible
    enabled: true
    base_url: http://localhost:11434/v1
    api_key_env: OLLAMA_API_KEY
    model: llama3.2:3b
    timeout_seconds: 300
    weight: 1.0

  - name: local-lmstudio
    type: openai_compatible
    enabled: false
    base_url: http://localhost:1234/v1
    api_key_env: LMSTUDIO_API_KEY
    model: local-model
    timeout_seconds: 120
    weight: 1.0

  - name: cloud-openai
    type: openai_compatible
    enabled: false
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    model: gpt-4.1-mini
    timeout_seconds: 90
    weight: 1.0

  - name: cloud-openrouter
    type: openai_compatible
    enabled: false
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    model: openai/gpt-4.1-mini
    timeout_seconds: 90
    weight: 1.0
    headers:
      HTTP-Referer: https://github.com/johncheungmk/openfusion
      X-Title: OpenFusion

fusion:
  default_strategy: panel_judge
  panel:
    - local-ollama
  judge_provider: local-ollama
  max_parallel: 1
  temperature: 0.2
  max_tokens: 256
  require_at_least_successes: 1
  include_candidate_outputs: true
  judge_candidate_max_chars: 4000

server:
  host: 127.0.0.1
  port: 8000
  api_key_env: OPENFUSION_API_KEY
```

Important: if `openfusion.yaml` says `model: qwen2.5:7b-instruct` but `ollama list` does not show `qwen2.5:7b-instruct`, OpenFusion will return a provider error such as `model not found`.

For local CPU models, use a smaller output limit at first:

```yaml
fusion:
  max_tokens: 128
```

or:

```yaml
fusion:
  max_tokens: 256
```

Large local models running on CPU can be slow. If a request times out, increase the provider timeout:

```yaml
providers:
  - name: local-ollama
    timeout_seconds: 300
```

## Run OpenFusion

Show configured providers:

### Windows PowerShell

```powershell
openfusion providers --config openfusion.yaml
```

### Linux / macOS Bash

```bash
openfusion providers --config openfusion.yaml
```

Try the CLI:

### Windows PowerShell

```powershell
openfusion chat "Explain model fusion in one paragraph" --config openfusion.yaml
```

### Linux / macOS Bash

```bash
openfusion chat "Explain model fusion in one paragraph" --config openfusion.yaml
```

Start the server:

### Windows PowerShell

```powershell
openfusion serve --config openfusion.yaml --port 8000
```

### Linux / macOS Bash

```bash
openfusion serve --config openfusion.yaml --port 8000
```

Check health:

### Windows PowerShell

```powershell
curl.exe http://localhost:8000/health
```

### Linux / macOS Bash

```bash
curl http://localhost:8000/health
```

Expected result:

```json
{"ok": true, "providers": ["local-ollama"]}
```

## Test the OpenAI-compatible API

### Recommended Windows PowerShell test

PowerShell quoting can break JSON when using `curl.exe -d`. The safest Windows test is `Invoke-RestMethod`:

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
      content = "Give a short RAG deployment plan in 5 bullets."
    }
  )
  max_tokens = 256
} | ConvertTo-Json -Depth 10

$response = Invoke-RestMethod `
  -Uri "http://localhost:8000/v1/chat/completions" `
  -Method Post `
  -Headers $headers `
  -Body $body

$response.choices[0].message.content
```

To test panel-judge fusion:

```powershell
$body = @{
  model = "openfusion/panel-judge"
  messages = @(
    @{
      role = "user"
      content = "Give a RAG deployment plan in exactly 3 short bullets."
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

### Linux / macOS Bash test

Direct provider route:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer replace-with-a-long-random-token" \
  -d '{
    "model": "provider/local-ollama/llama3.2:3b",
    "messages": [
      {
        "role": "user",
        "content": "Give a short RAG deployment plan in 5 bullets."
      }
    ],
    "max_tokens": 256
  }'
```

Panel-judge fusion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer replace-with-a-long-random-token" \
  -d '{
    "model": "openfusion/panel-judge",
    "messages": [
      {
        "role": "user",
        "content": "Give a RAG deployment plan in exactly 3 short bullets."
      }
    ],
    "max_tokens": 128
  }'
```

## Example cloud + local config

Edit `openfusion.yaml`, enable a cloud provider, and set the API key in `.env` or your shell environment.

Example `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
OPENFUSION_API_KEY=replace-with-a-long-random-token
```

Example `openfusion.yaml`:

```yaml
providers:
  - name: local-ollama
    type: openai_compatible
    enabled: true
    base_url: http://localhost:11434/v1
    api_key_env: OLLAMA_API_KEY
    model: llama3.2:3b
    timeout_seconds: 300
    weight: 1.0

  - name: cloud-openai
    type: openai_compatible
    enabled: true
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    model: gpt-4.1-mini
    timeout_seconds: 90
    weight: 1.0

fusion:
  default_strategy: panel_judge
  panel:
    - local-ollama
    - cloud-openai
  judge_provider: cloud-openai
  max_parallel: 2
  temperature: 0.2
  max_tokens: 512
  require_at_least_successes: 1
  include_candidate_outputs: true
  judge_candidate_max_chars: 4000

server:
  host: 127.0.0.1
  port: 8000
  api_key_env: OPENFUSION_API_KEY
```

## Strategies

### `panel_judge`

1. Send the prompt to multiple models in parallel.
2. Collect candidate answers.
3. Send candidates to a judge model.
4. Return a final answer plus candidate metadata.

Provider `weight` values are included in the judge prompt as advisory hints in the MVP; they do not yet control sampling or voting. Candidate answers are truncated to `fusion.judge_candidate_max_chars` characters before judge synthesis to keep prompts bounded.

### `fallback`

1. Try provider 1.
2. If it fails, try provider 2.
3. Continue until one succeeds.

### Direct provider routing

To bypass fusion and route to one configured provider, use a model ID from `/v1/models`, for example:

```text
provider/local-ollama/llama3.2:3b
```

## Use with Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="replace-with-a-long-random-token",
)

completion = client.chat.completions.create(
    model="openfusion/panel-judge",
    messages=[{"role": "user", "content": "Compare vLLM and Ollama."}],
    extra_body={"fusion_strategy": "panel_judge"},
)
print(completion.choices[0].message.content)
```

To bypass fusion and route to one configured provider:

```python
completion = client.chat.completions.create(
    model="provider/local-ollama/llama3.2:3b",
    messages=[{"role": "user", "content": "Give a short RAG deployment plan."}],
)
```

## Troubleshooting

### `model not found`

Run:

```bash
ollama list
```

Then edit `openfusion.yaml` so the provider `model` exactly matches one of the installed model names.

### Slow response or timeout with local Ollama

Large models on CPU can be slow, especially with `panel_judge` because it may call a model once for candidate generation and once again for judge synthesis. Use a smaller model, lower `max_tokens`, or increase `timeout_seconds`.

Recommended CPU test settings:

```yaml
fusion:
  max_tokens: 128

providers:
  - name: local-ollama
    timeout_seconds: 300
```

### Windows PowerShell JSON errors with curl

If `curl.exe` returns JSON parsing errors, use the `Invoke-RestMethod` examples above or send JSON from a file with `--data-binary @body.json`.

## Roadmap

- Real token streaming from upstream providers.
- Cost-aware routing.
- Latency-aware routing.
- RAG-aware model fusion.
- Prompt classification for private vs public data routing.
- Evaluation dashboard.
- Optional integration with weight-level model-merging tools.

## License

MIT
