# Strategies

OpenFusion includes:

| Strategy | Purpose |
|---|---|
| `fallback` | Try providers in order until one succeeds. |
| `self_moa` | Sample one provider multiple times, then select or synthesize. |
| `semantic_vote` | Group concise equivalent answers before voting. |
| `parallel_synthesis` | Generate independent drafts and synthesize. |
| `pairwise_rank_fuse` | Rank candidates, then fuse top answers. |
| `uncertainty_cascade` | Start cheap and escalate on low confidence or disagreement. |

No strategy is universally best. Use OpenFusion Lab to measure trade-offs.
