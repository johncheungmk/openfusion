# Baseline Comparison

OpenFusion Lab compares fusion strategies against two important baselines.

## Fallback baseline

The fallback baseline is the default operational strategy that tries providers in order. It is useful because it represents a simple production setup.

## Best single-model baseline

The best single-model baseline is the highest-performing individual provider on the dataset.

This is often the most important baseline. Fusion should not be considered successful unless it improves over the best single model or provides a useful latency/cost trade-off.

## Metrics

OpenFusion Lab reports:

- accuracy delta versus fallback;
- accuracy delta versus best single model;
- relative accuracy improvement;
- latency ratio;
- call ratio;
- token ratio;
- balanced score difference.

## Example interpretation

If a strategy has:

```text
accuracy_delta_vs_best_single_pp = -10.0
latency_ratio_vs_best_single = 0.5
```

then it is 10 percentage points less accurate but twice as fast as the best single model.

That may be useful for latency-sensitive use cases, but it is not an accuracy improvement.
