# OpenFusion Lab

OpenFusion Lab is the experiment runner for comparing models and fusion strategies.

It answers practical questions:

- Which model works best alone?
- Does fusion improve over the best base model?
- Which strategy is fastest?
- Which strategy gives the best accuracy per call?
- Which strategy is best balanced for quality and latency?

## Workflow

```text
Choose models
    ↓
Start inference engines
    ↓
Create lab.yaml
    ↓
Run benchmark
    ↓
Compare baselines and fusion strategies
    ↓
Read recommendation
    ↓
Export result card
```

## Main commands

```bash
openfusion lab validate lab.yaml
openfusion lab generate-config lab.yaml --out openfusion.lab.generated.yaml
openfusion lab run lab.yaml --out results.json
openfusion lab recommend results.json
openfusion lab export results.json --out result-card.json
openfusion lab engine-plan lab.yaml
```
