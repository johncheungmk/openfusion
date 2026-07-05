# Uncertainty Cascade

`uncertainty_cascade` starts with cheaper/faster providers and escalates on failure, low confidence, or disagreement. It is for latency/cost-sensitive tasks.

## Evaluation advice

Always compare this strategy against the best single-model baseline, fallback, latency, calls, and token usage.
