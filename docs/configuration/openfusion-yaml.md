# openfusion.yaml

`openfusion.yaml` configures providers, fusion behavior, and the API server.

Create one with:

```bash
openfusion init --path openfusion.yaml
```

Edit provider model names so they exactly match your local or cloud model endpoints.

`fusion.max_total_calls` is the hard administrator ceiling for every multi-call request.
Clients may request a lower `fusion_max_total_calls`, but they cannot raise the configured
ceiling. The runtime bounds candidate scheduling and pair comparisons before creating
work, so a large request override cannot allocate an unbounded task list.

See [Providers](providers.md) for optional token-pricing fields and
[Evaluation methodology](../concepts/evaluation-methodology.md) for how estimated cost is
reported.
