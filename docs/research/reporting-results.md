# Reporting Results

A credible OpenFusion report should include:

- dataset name and size;
- prompt template;
- hardware;
- models and quantization;
- engine versions;
- strategy settings;
- max tokens;
- max total calls;
- accuracy or win rate;
- latency p50 and p95;
- calls per example;
- token usage;
- delta versus fallback;
- delta versus best single model.

Never report only the best fusion result without showing the baseline.
