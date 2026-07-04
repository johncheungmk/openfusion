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
- `pairwise_rank_fuse`: generates panel candidates, ranks them with a ranker provider,
  then fuses the top candidates.
- `semantic_vote`: groups concise answers by exact normalized text or bounded LLM
  equivalence checks before voting.
- `uncertainty_cascade`: starts with cheaper providers and escalates only when
  confidence, consistency, format, or provider health requires it.

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
  ranker_provider: local-ollama
  fuser_provider: local-ollama
  rank_top_k: 3
  pairwise_rank_max_pairs: 12
  pairwise_rank_mode: pairwise
  vote_equivalence_provider: local-ollama
  semantic_vote_max_pairs: 12
  semantic_vote_mode: rule_only
  cascade_providers: [local-ollama]
  cascade_confidence_threshold: 0.75
  cascade_consistency_samples: 1
  cascade_escalate_on_disagreement: true
  cascade_max_steps: 3
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

## Pairwise rank fuse

`pairwise_rank_fuse` is useful when candidate answers are long enough that direct
voting is brittle. In `pairwise` mode, OpenFusion compares candidate pairs up to
`pairwise_rank_max_pairs` and records public win counts. In `score` mode, it asks
the ranker for parseable JSON scores in one call. If parsing fails, the original
candidate order is used and synthesis still runs over the top candidates.

## Semantic voting

`semantic_vote` is intended for concise answers that may use different wording, such
as `4`, `four`, and `the answer is 4`. `rule_only` matches the existing normalized
exact voting behavior. `llm_equivalence` asks `vote_equivalence_provider` whether
answers are equivalent, but stops at `semantic_vote_max_pairs` and the remaining
`max_total_calls` budget. If no equivalence provider is configured, it falls back
to rule-only grouping.

## Uncertainty cascade

`uncertainty_cascade` is intended for cost-saving routing. It starts with the first
configured cascade provider and accepts an answer when the provider returns a
parseable public confidence score at or above `cascade_confidence_threshold`.
With `cascade_consistency_samples: 2`, it can also sample the same provider twice
and escalate when normalized answers disagree.

Escalation reasons are public operational metadata: provider failure, low confidence,
sample disagreement, invalid response format, exhausted provider steps, or exhausted
call budget. The trace does not include hidden chain-of-thought.

## Evaluation

Compare MoA-style workflows at equal budgets. For example:

```bash
openfusion evaluate examples/eval_moa_sample.jsonl \
  --config openfusion.yaml \
  --compare-strategies fallback,self_moa,parallel_synthesis,pairwise_rank_fuse \
  --max-total-calls 6 \
  --output moa-eval.json
```

Optional LLM graders can help inspect open-ended outputs, but they are not ground
truth and should not replace deterministic or human evaluation for published claims.
