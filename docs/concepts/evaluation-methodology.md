# Evaluation Methodology

OpenFusion is designed to make fusion claims testable.

## Required comparisons

A responsible fusion experiment should compare:

1. model A alone;
2. model B alone;
3. best single-model baseline;
4. fallback;
5. Self-MoA using the best model;
6. mixed-model fusion;
7. uncertainty cascade when latency/cost matters.

## Required metrics

Report:

- accuracy or win rate;
- delta versus fallback in percentage points;
- delta versus best single model in percentage points;
- relative accuracy improvement;
- average and p95 latency;
- model calls per example;
- tokens per example;
- accuracy per call;
- accuracy per 1k tokens.

## Negative results

If fusion does not improve over the best single model, say so. Negative results are useful because they help identify when the extra calls and latency are not justified.
