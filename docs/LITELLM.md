# Using OpenFusion with LiteLLM

LiteLLM and OpenFusion address different layers:

- LiteLLM: provider access, virtual keys, budgets, load balancing, fallback, and observability.
- OpenFusion: independent model attempts, voting, critique, refinement, and synthesis.

Point one or more OpenFusion providers to the LiteLLM proxy:

```yaml
providers:
  - name: litellm-fast
    type: openai_compatible
    enabled: true
    base_url: http://localhost:4000/v1
    api_key_env: LITELLM_API_KEY
    model: fast-model-alias
    timeout_seconds: 120
    weight: 1.0

  - name: litellm-strong
    type: openai_compatible
    enabled: true
    base_url: http://localhost:4000/v1
    api_key_env: LITELLM_API_KEY
    model: strong-model-alias
    timeout_seconds: 180
    weight: 1.5

fusion:
  default_strategy: critique_revision
  panel: [litellm-fast, litellm-strong]
  critic_provider: litellm-strong
  reviser_provider: litellm-strong
  judge_provider: litellm-strong
  max_total_calls: 8
```

The model aliases must exist in LiteLLM. OpenFusion's `max_total_calls` limits calls at the orchestration layer; LiteLLM should enforce financial budgets, rate limits, and tenant policies at the gateway layer.
