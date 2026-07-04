# Using OpenFusion with LiteLLM

LiteLLM and OpenFusion address different layers:

- LiteLLM: provider access, virtual keys, budgets, load balancing, fallback, and observability.
- OpenFusion: independent model attempts, voting, critique, refinement, and synthesis.

OpenFusion is not a replacement for LiteLLM. LiteLLM is the better place to
centralize provider credentials, tenant policy, rate limits, budgets, retries, and
gateway observability. OpenFusion is the layer that can sit above a gateway to run
transparent, bounded, MoA-inspired workflows with local or cloud providers.

| Capability | OpenFusion | LiteLLM |
|---|---|---|
| Open-source implementation | Yes | Yes |
| Self-hostable | Yes | Yes |
| Local model support | Yes, through OpenAI-compatible providers | Yes, through configured providers |
| OpenAI-compatible gateway | Basic app endpoint | Primary focus |
| Learned orchestrator | No | No |
| Configurable workflow strategies | MoA/voting/ranking/cascade workflows | Routing and gateway policies |
| Transparent traces | Workflow traces | Gateway logs/observability |
| Built-in evaluation | Yes | No MoA evaluation harness |
| Provider/key management focus | Basic | Strong |
| Intended role | Orchestration runtime | Provider gateway and operations layer |

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

Do not claim quality gains from adding OpenFusion above LiteLLM without evaluation.
Use equal-budget baselines such as a single best model, direct/fallback routing,
`best_of_n`, `self_moa`, `layered_refinement`, `pairwise_rank_fuse`,
`semantic_vote` for short answers, and `uncertainty_cascade` for cost-sensitive
tasks.
