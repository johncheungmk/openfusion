# MoA and Self-MoA

OpenFusion supports several inference-time workflows inspired by mixture-of-agents
patterns. These are API-level orchestration strategies, not weight-level model merging
or claims to reproduce proprietary orchestrators.

## Strategies

- `parallel_synthesis`: asks a panel of providers for independent drafts, then asks a
  judge provider to synthesize a final answer.
- `layered_refinement`: runs one or more refinement layers over previous candidate
  outputs, then synthesizes the final layer.
- `self_moa`: samples one provider multiple times, then either selects the best sample
  unchanged or synthesizes a new final answer.
- `self_moa_seq`: batches Self-MoA samples and carries forward a running best or fused
  answer for larger sample counts or long candidates.

## Self-MoA configuration

```yaml
fusion:
  self_moa_provider: local-ollama
  self_moa_samples: 4
  self_moa_temperature: 0.7
  self_moa_mode: synthesize  # select or synthesize
  self_moa_batch_size: 4
  self_moa_seq_carry_max_chars: 12000
  max_total_calls: 8
```

If `self_moa_provider` is unset, OpenFusion uses `judge_provider`, then the first
panel provider. Both Self-MoA strategies obey `max_total_calls`. When the budget is
exhausted before a selector or synthesizer call, OpenFusion falls back to the best
usable candidate by deterministic local scoring.

Public traces show provider, requested sample count, mode, and call count. They do
not request or expose hidden chain-of-thought.
