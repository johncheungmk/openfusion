# `openfusion.yaml`

`openfusion.yaml` is the runtime configuration file used by the OpenFusion API server and CLI.

Create it with:

```powershell
openfusion init --path openfusion.yaml
```

A minimal local Ollama configuration:

```yaml
providers:
  - name: ollama-llama32-3b
    type: openai_compatible
    enabled: true
    base_url: http://127.0.0.1:11434/v1
    api_key_env:
    model: llama3.2:3b
    timeout_seconds: 180
    weight: 1.0

fusion:
  default_strategy: fallback
  panel:
    - ollama-llama32-3b
  judge_provider: ollama-llama32-3b
  max_total_calls: 4
  max_tokens: 128

server:
  host: 127.0.0.1
  port: 8000
  api_key_env: OPENFUSION_API_KEY
```

Use `openfusion.yaml` when you want to serve OpenFusion as an OpenAI-compatible API.
