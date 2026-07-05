# Mixture-of-Agents

Mixture-of-Agents is a family of inference-time methods where multiple model agents generate, refine, critique, vote on, or synthesize responses.

OpenFusion supports MoA-inspired workflows such as:

- `self_moa`
- `self_moa_seq`
- `parallel_synthesis`
- `semantic_vote`
- `pairwise_rank_fuse`
- `layered_refinement`
- `uncertainty_cascade`

## OpenFusion and MoA

OpenFusion is inspired by MoA-style systems, but it is not the original Together AI MoA implementation.

OpenFusion is a transparent, configurable, OpenAI-compatible runtime for experiments. It is designed to help users measure whether fusion helps on their own models, hardware, dataset, and call budget.

## Recommended baselines

Always compare fusion against:

1. each individual model;
2. the best single model;
3. fallback;
4. Self-MoA;
5. mixed-model fusion;
6. equal-budget alternatives.

A fusion strategy is only useful if the quality improvement justifies the additional latency, calls, and token usage.
