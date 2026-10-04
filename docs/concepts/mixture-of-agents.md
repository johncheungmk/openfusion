# Mixture-of-Agents

Mixture-of-Agents is a family of inference-time methods where multiple model agents generate, refine, critique, vote on, or synthesize responses.

OpenFusion supports MoA-inspired workflows, including `self_moa`, `parallel_synthesis`, `semantic_vote`, `pairwise_rank_fuse`, `layered_refinement`, and `uncertainty_cascade`.

OpenFusion is not the original Together AI MoA implementation and does not claim the same benchmark results.

## When to use it

Our [benchmark evidence](../research/benchmark-evidence.md) supports particular test-guided
workflows and motivates selective escalation. It does not establish that mixing similarly
sized models generally beats repeated use of the strongest model. Compare both on held-out
tasks, include synthesis and failed calls, and distinguish measured benefits from possible
use cases such as open-ended writing or specialist collaboration.
