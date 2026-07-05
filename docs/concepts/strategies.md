# Strategies

OpenFusion strategies are bounded workflows for deciding how many model calls to make and how to combine their outputs.

| Strategy | Best for | Notes |
|---|---|---|
| `fallback` | Reliability baseline | Tries providers in order. |
| `self_moa` | Same-model test-time compute | Samples one provider multiple times, then selects or synthesizes. |
| `self_moa_seq` | Many same-model samples | Sequential batched Self-MoA with carry-forward aggregation. |
| `semantic_vote` | Short answers and MCQ | Groups semantically equivalent concise answers before voting. |
| `parallel_synthesis` | Open-ended synthesis | Uses independent drafts then synthesizes a final answer. |
| `pairwise_rank_fuse` | Candidate ranking + synthesis | Ranks candidates before fusing top answers. |
| `uncertainty_cascade` | Latency/cost-sensitive tasks | Starts cheap and escalates on low confidence, disagreement, or failure. |
| `critique_revision` | Draft review and improvement | Drafts, critiques, then revises. |
| `layered_refinement` | MoA-style refinement | Later layers read earlier outputs. |
| `adaptive` | Heuristic orchestration | Chooses a bounded workflow from configured strategies. |

## Choosing a strategy

- For MCQ/exact-answer tasks, start with `fallback`, `semantic_vote`, and `self_moa`.
- For open-ended planning or review tasks, test `parallel_synthesis`, `pairwise_rank_fuse`, and `critique_revision`.
- For production latency/cost constraints, test `uncertainty_cascade`.
