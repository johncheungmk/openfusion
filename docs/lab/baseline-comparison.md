# Baseline Comparison

OpenFusion Lab compares fusion strategies against:

- each individual provider;
- the fallback baseline when `fallback` is included in `strategies`;
- the best-observed single-model baseline on the evaluated split.

A fusion strategy is only a descriptive accuracy improvement on a Lab run if it beats the
best-observed single-model baseline. For a test-set claim, choose that single model on an
independent validation split. A strategy may still be useful if it is faster or cheaper
while losing little accuracy.

Interpret point estimates with their Wilson 95% intervals and keep outcomes paired by
example. `accuracy_per_call` and `accuracy_per_1k_tokens` mean correct answers divided by
resource use, so they remain comparable when dataset size changes. Latency is end-to-end
wall time; summed provider work is not substituted for user-visible latency in parallel
strategies.
