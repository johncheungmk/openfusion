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
  panel_roles:
    - name: factual_checker
      instruction: Focus on factual accuracy and cite uncertainty.
    - name: edge_case_reviewer
      instruction: Focus on edge cases, failure modes, and missing assumptions.
  structured_synthesis: true
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

## Role-diverse panels

`panel_roles` gives independent panel calls different public instructions. If more
panel calls are made than roles are configured, OpenFusion cycles through the role
list. This applies to `parallel_synthesis`, `critique_revision` draft generation,
and `layered_refinement` panel layers. Public traces include the role name, not the
role's private reasoning.

## Structured synthesis

When `structured_synthesis` is enabled, synthesis calls are prompted to return
parseable public sections:

- `consensus_points`
- `contradictions`
- `unique_insights`
- `missing_information`
- `final_answer`

OpenFusion parses `final_answer` as the user-facing answer and stores the public
sections in `workflow_outputs` when `include_workflow_outputs` is enabled. If the
model returns invalid structure, OpenFusion degrades to the plain synthesized text.
