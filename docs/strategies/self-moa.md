# Self-MoA

`self_moa` samples one provider multiple times, then selects or synthesizes. It is useful for testing whether repeated sampling from the best model beats mixed-model fusion.

## Evaluation advice

Always compare this strategy against the best single-model baseline, fallback, latency, calls, and token usage.
