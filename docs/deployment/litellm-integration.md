# LiteLLM Integration

OpenFusion and LiteLLM solve different problems.

- LiteLLM is a model gateway for provider management, keys, budgets, and routing.
- OpenFusion is a fusion and experiment runtime.

A common architecture is:

```text
Application → OpenFusion → LiteLLM → model providers
```

Configure LiteLLM as an OpenAI-compatible provider in OpenFusion:

```yaml
providers:
  - name: litellm-gateway
    type: openai_compatible
    enabled: true
    base_url: http://127.0.0.1:4000/v1
    api_key_env: LITELLM_API_KEY
    model: my-model
```
