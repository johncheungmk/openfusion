# Providers

A provider is any model endpoint that OpenFusion can call.

OpenFusion works best with OpenAI-compatible APIs:

- Ollama
- vLLM
- LM Studio
- LiteLLM
- OpenRouter-compatible endpoints
- cloud APIs exposed through `/v1/chat/completions`

Provider example:

```yaml
- name: local-ollama
  type: openai_compatible
  enabled: true
  base_url: http://127.0.0.1:11434/v1
  api_key_env:
  model: llama3.2:3b
  timeout_seconds: 180
  weight: 1.0
```

For local Ollama, `api_key_env` can be empty.
