# First Fusion Test

For a first test, use OpenFusion Lab rather than a long free-form prompt. The Lab command records accuracy, latency, calls, tokens, and comparisons against baselines.

```powershell
openfusion lab validate examples/lab_local_small.yaml
openfusion lab run examples/lab_local_small.yaml --out results-local-small.json
openfusion lab recommend results-local-small.json
```

The output includes:

- single-model baselines;
- strategy comparison;
- delta versus fallback;
- delta versus best single model;
- recommendations for accuracy, latency, efficiency, and balanced use.

!!! note
    A negative result is still useful. If fusion does not beat the best single model, OpenFusion Lab should say so.
