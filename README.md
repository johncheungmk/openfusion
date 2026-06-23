# OpenFusion

<p align="center">
  <img src="assets/openfusion-banner.svg" alt="OpenFusion banner: open-source API-level model fusion gateway" width="100%">
</p>

OpenFusion is an open-source, OpenAI-compatible **model-fusion gateway** for local and cloud LLMs.

It lets you combine multiple models such as Ollama, LM Studio, vLLM, OpenAI, OpenRouter, or any OpenAI-compatible API. The first release focuses on **API-level fusion**: run several models in parallel, compare their answers, and synthesize a stronger final answer using a judge model.

> OpenFusion is not weight-level model merging. It is a practical gateway for multi-model deliberation, routing, fallback, and judge synthesis.

## Features

- OpenAI-compatible `/v1/chat/completions` API.
- Works with local and cloud OpenAI-compatible APIs.
- `panel_judge` strategy: parallel model panel + judge synthesis.
- `fallback` strategy: try providers in order until one succeeds.
- CLI for local testing.
- FastAPI server for integration with Dify, custom apps, agents, and SDK clients.
- Dockerfile and docker-compose support.
- GitHub Actions test workflow.

## Quick start

Windows PowerShell:

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

Bash/Linux/macOS:

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

The generated config starts with a single Ollama provider. Start Ollama and pull the
configured model, or edit `openfusion.yaml` to enable a different provider. API keys
belong in `.env` or your shell environment, not in the YAML file.

By default the server binds to `127.0.0.1`. If `OPENFUSION_API_KEY` is unset,
OpenFusion logs a warning and accepts unauthenticated local requests. Set a strong
`OPENFUSION_API_KEY` before binding to `0.0.0.0` or exposing the service.

Then run:

```bash
openfusion providers --config openfusion.yaml
openfusion chat "Explain model fusion in one paragraph" --config openfusion.yaml
openfusion serve --config openfusion.yaml --port 8000
```

Test the OpenAI-compatible API:

```powershell
curl.exe http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer replace-with-a-long-random-token" `
  -d "{\"model\":\"openfusion/panel-judge\",\"messages\":[{\"role\":\"user\",\"content\":\"Give a short RAG deployment plan.\"}]}"
```

## Example local-only config

```yaml
providers:
  - name: local-ollama
    type: openai_compatible
    enabled: true
    base_url: http://localhost:11434/v1
    api_key_env: OLLAMA_API_KEY
    model: qwen2.5:7b-instruct

  - name: local-lmstudio
    type: openai_compatible
    enabled: true
    base_url: http://localhost:1234/v1
    api_key_env: LMSTUDIO_API_KEY
    model: local-model

fusion:
  default_strategy: panel_judge
  panel: [local-ollama, local-lmstudio]
  judge_provider: local-ollama
  max_parallel: 2
  temperature: 0.2
  max_tokens: 1200
  judge_candidate_max_chars: 4000

server:
  host: 127.0.0.1
  port: 8000
  api_key_env: OPENFUSION_API_KEY
```

## Example cloud + local config

```yaml
providers:
  - name: local-ollama
    type: openai_compatible
    enabled: true
    base_url: http://localhost:11434/v1
    api_key_env: OLLAMA_API_KEY
    model: qwen2.5:7b-instruct

  - name: cloud-openai
    type: openai_compatible
    enabled: true
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    model: gpt-4.1-mini

fusion:
  default_strategy: panel_judge
  panel: [local-ollama, cloud-openai]
  judge_provider: cloud-openai
```

## Strategies

### `panel_judge`

1. Send the prompt to multiple models in parallel.
2. Collect candidate answers.
3. Send candidates to a judge model.
4. Return a final answer plus candidate metadata.

Provider `weight` values are included in the judge prompt as advisory hints in the
MVP; they do not yet control sampling or voting. Candidate answers are truncated to
`fusion.judge_candidate_max_chars` characters before judge synthesis to keep prompts
bounded.

### `fallback`

1. Try provider 1.
2. If it fails, try provider 2.
3. Continue until one succeeds.

## Use with Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="replace-with-a-long-random-token")

completion = client.chat.completions.create(
    model="openfusion/panel-judge",
    messages=[{"role": "user", "content": "Compare vLLM and Ollama."}],
    extra_body={"fusion_strategy": "panel_judge"},
)
print(completion.choices[0].message.content)
```

To bypass fusion and route to one configured provider, use a model ID from
`/v1/models`, for example `provider/local-ollama/qwen2.5:7b-instruct`.

## Roadmap

- Streaming support.
- Cost-aware routing.
- Latency-aware routing.
- RAG-aware model fusion.
- Prompt classification for private vs public data routing.
- Evaluation dashboard.
- Optional integration with weight-level model-merging tools.

## License

MIT
