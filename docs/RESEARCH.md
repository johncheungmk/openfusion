# Research positioning

OpenFusion is an engineering implementation inspired by several inference-time scaling directions. The links below are references, not claims that OpenFusion reproduces every reported result.

## Parallel synthesis

OpenRouter Fusion runs a panel in parallel, compares consensus and contradictions, and uses structured analysis to write a stronger final response:

- https://openrouter.ai/docs/guides/features/plugins/fusion
- https://openrouter.ai/blog/announcements/fusion-beats-frontier/

OpenFusion's `parallel_synthesis` is the closest built-in analogue, without OpenRouter's proprietary panel selection or integrated web tools. OpenFusion v0.4 can optionally ask the synthesizer for public structured sections covering consensus, contradictions, unique insights, missing information, and the final answer.

## Self-consistency and voting

Self-consistency samples multiple reasoning paths and chooses a consistent answer:

- https://openreview.net/forum?id=1PL1NIMMrw

OpenFusion implements textual `majority_vote` and `weighted_vote`. These are transparent baselines and work best for concise or regex-extractable outputs. OpenFusion v0.4 also adds `self_moa`, which samples one provider multiple times and then selects the best sample or synthesizes a new final answer.

## Ranking and generative fusion

LLM-Blender separates candidate ranking from generative fusion:

- https://aclanthology.org/2023.acl-long.792/

OpenFusion exposes both ideas as `best_of_n` and `parallel_synthesis`.

## Mixture of Agents

Mixture-of-Agents presents previous-layer outputs to later agents for iterative improvement:

- https://arxiv.org/abs/2406.04692

OpenFusion's `layered_refinement` implements a configurable, bounded version of this pattern.

OpenFusion's `self_moa_seq` is a sequential, single-provider variant for larger sample counts or long candidates. It batches candidates and carries forward a running selected or fused answer instead of presenting every candidate to one final prompt.

Role-diverse panel prompts are a practical engineering control for assigning complementary public perspectives to otherwise similar panel calls. They should be evaluated empirically; a role prompt is not a guarantee that a model will perform that function well.

Research also warns that mixing lower-quality models can reduce performance, so provider diversity should be evaluated rather than assumed beneficial:

- https://arxiv.org/abs/2502.00674

## Debate, critique, and revision

Multi-agent debate and round-table approaches explore iterative criticism and consensus:

- https://arxiv.org/abs/2305.19118
- https://arxiv.org/abs/2309.13007

OpenFusion implements a controlled `critique_revision` workflow rather than unrestricted conversational debate.

## Sakana orchestration

Sakana Fugu and the Conductor research dynamically choose models and workflow structures:

- https://sakana.ai/fugu/
- https://sakana.ai/learning-to-orchestrate/
- https://arxiv.org/abs/2606.21228

OpenFusion's `adaptive` strategy is intentionally more modest. It uses readable heuristics or a constrained JSON planner. It is not a reinforcement-learned orchestration foundation model.

Sakana's AB-MCTS research explores multi-model tree search:

- https://sakana.ai/ab-mcts/

Search trees, external verification, and tool execution remain future OpenFusion work.

## Weight-level model merging

Sakana's evolutionary model merging combines model parameters or layers offline. That is a different problem from OpenFusion's API-level inference workflows:

- https://sakana.ai/evolutionary-model-merge/

## Evaluation principle

Multi-agent methods spend additional inference compute and do not improve every task. Compare accuracy, latency, token use, and financial cost at equal or explicitly reported budgets. OpenFusion includes a small exact-match harness to encourage reproducible local comparisons, but serious benchmarks require domain-specific graders.
