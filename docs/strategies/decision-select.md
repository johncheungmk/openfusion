# Decision selection with Kev

`decision_select` generates candidate answers with the configured panel, then asks a
local decision model to select one. It returns the selected answer unchanged.
The selector uses the System One `/v1/systemone` API. Kev is the tested implementation;
other compatible services need their own compatibility and quality checks.

## Configure

Install and serve [Kev](https://github.com/jaredpalmer/kev) in its own environment.
Keep the endpoint on localhost or behind an authenticated private service.

```bash
python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

Add a decision endpoint alongside your existing text providers:

```yaml
providers:
  - name: model-a
    base_url: http://localhost:11434/v1
    model: qwen3:4b
  - name: model-b
    base_url: http://localhost:11434/v1
    model: gemma3:4b

decision_model:
  base_url: http://localhost:8009/v1
  model: kev-latest
  # api_key_env: KEV_API_KEY
  timeout_seconds: 60
  min_probability: 0

fusion:
  default_strategy: decision_select
  panel: [model-a, model-b]
  max_total_calls: 3
  max_tokens: 768
```

Send a normal chat request using model `openfusion/decision-select`. Existing direct
provider routes and other fusion strategies continue to work. A decision model is
configured separately because it cannot generate chat responses.

## Behavior and limits

- Every proposal and every selector request counts toward `max_total_calls`.
- With a budget of at least three, one call is reserved for selection. A budget of
  one or two returns the first usable candidate without a selector call.
- At most 255 proposals are generated for this strategy, matching the Choice API limit.
- Failed or empty proposals are excluded. If all fail, the request fails.
- A timeout, malformed probability distribution, invalid choice, or probability below
  `min_probability` returns the first usable candidate in panel order.
- Selection uses the conversation, including system/developer messages, bounded by
  `transcript_max_chars`. Each proposal is bounded by `judge_candidate_max_chars`.
  Tune these limits for long tasks; truncation can remove evidence needed for selection.
- Traces contain call status, latency, and token counts. They do not expose hidden reasoning,
  endpoint credentials, or upstream exception text.
- Kev's output-token count describes the serialized decision, **not generated prose**.
  Decision-service monetary cost is unknown, so total estimated cost remains unknown.

`min_probability` applies to the selected option's probability, not the API's separately
defined `confidence` field. Neither is a measured accuracy rate on your workload.
Validate thresholds on separate development data before relying on them.

## When to use it

A decision model can only choose among the answers it receives. It cannot recover a
correct answer if every proposal is wrong. Compare it with the strongest individual
model and a conventional LLM selector using identical proposals.

For code, run trusted checks in an isolated execution environment before semantic
selection. The runtime strategy itself does **not** execute candidate code or accept
model-generated tools. Visible-test filtering in the experiment is an external benchmark
policy, distinct from the generic `decision_select` runtime.

See the [Spark experiment](../research/kev-selection-spark.md) for measured results and
the exact scope of the evidence.
