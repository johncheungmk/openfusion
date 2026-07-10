# lab.yaml

`lab.yaml` defines a local experiment: engines, models, dataset, strategies, and recommendation preferences.

Use it with:

```bash
openfusion lab validate lab.yaml
openfusion lab run lab.yaml --out results.json
openfusion lab recommend results.json
```

Each model may define `input_cost_per_million_tokens_usd` and
`output_cost_per_million_tokens_usd`. New runs produce Lab result-card schema v2 with
Wilson accuracy intervals, p99 wall latency, corrected efficiency metrics, cost coverage,
and panel-complementarity diagnostics. Existing v1 result cards remain readable.

`recommendation.objective` selects the primary recommendation. `max_latency_ms` is a hard
eligibility cap for that choice, while `prefer_lower_calls` controls the call penalty in
the documented balanced-score heuristic. Schema v2 stores these settings so `openfusion
lab recommend` reproduces the original choice. Unknown strategy options are rejected
instead of being silently ignored.

Shareable result cards omit engine base URLs, API-key environment-variable names, and
provider headers. The configuration hash preserves change detection without publishing a
private endpoint.
