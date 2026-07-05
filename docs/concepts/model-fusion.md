# What is Model Fusion?

Model fusion means using more than one model call to produce, select, or validate a final answer.

In OpenFusion, fusion happens at inference time. It does not merge neural network weights.

## Examples

- Ask two models independently and synthesize a final answer.
- Sample one model several times and choose the best result.
- Ask models to vote on a short answer.
- Run a cheap model first and escalate only when confidence is low.
- Rank candidate answers and fuse the top candidates.

## Why evaluate fusion?

Fusion can improve some tasks, but it can also reduce quality or increase latency.

OpenFusion therefore reports:

- accuracy or win rate;
- latency;
- number of model calls;
- token usage;
- improvement or regression versus fallback;
- improvement or regression versus the best single model.

Do not assume that more agents or more model calls automatically improve results.
