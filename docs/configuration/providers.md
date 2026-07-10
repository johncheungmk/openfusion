# Providers

OpenFusion providers are OpenAI-compatible model endpoints, such as Ollama, vLLM, LM Studio, LiteLLM, OpenRouter-compatible endpoints, and cloud APIs.

Each provider has a nonempty name that cannot contain `/`, a base URL, model name,
timeout, weight, and optional API-key environment variable. Fusion roles and panels may
reference only enabled providers.

Base URLs cannot contain URL user information, query strings, or fragments. Put secrets
in `api_key_env` or configured headers, never in a URL.

Optional v0.6 pricing fields enable cost-complete traces and reports:

```yaml
input_cost_per_million_tokens_usd: 0.15
output_cost_per_million_tokens_usd: 0.60
```

Set both fields for every provider used by a workflow. If any executed call is unpriced,
the aggregate cost is `null`; OpenFusion never presents a partial sum as complete. Use
zero for local API billing only when that matches your accounting assumptions. Hardware,
electricity, cached tokens, reasoning tokens, images, and other provider-specific charges
are not inferred.
