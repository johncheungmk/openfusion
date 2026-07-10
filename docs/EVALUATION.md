# Evaluation guide

OpenFusion does not assume that more calls produce a better answer. Evaluate each
strategy on representative tasks.

Do not claim benchmark gains without evaluation. Report the dataset, grader,
strategy settings, call budget, token use, latency, failures, and whether web or
retrieval tools were available.

## Dataset format

One JSON object per line:

```json
{"id":"math-1","prompt":"Return only the answer: 2 + 2","reference":"4"}
{"id":"mcq-1","prompt":"Answer: A, B, C, or D","reference":["B","b"],"answer_regex":"Answer:\\s*([A-D])"}
```

Fields:

- `id`: unique case ID;
- `prompt`: user message;
- `reference`: string or list of accepted strings;
- `system`: optional system message;
- `answer_regex`: optional extraction regex; group 1 is used when present;
- `rubric`: optional task-specific instruction for LLM graders;
- `metadata`: optional object preserved by the loader.

## Run

```bash
openfusion evaluate examples/eval_sample.jsonl \
  --config openfusion.yaml \
  --strategy fallback \
  --output fallback.json

openfusion evaluate examples/eval_sample.jsonl \
  --config openfusion.yaml \
  --strategy weighted_vote \
  --output weighted-vote.json

openfusion evaluate examples/eval_sample.jsonl \
  --config examples/semantic_vote.yaml \
  --strategy semantic_vote \
  --output semantic-vote.json

openfusion evaluate examples/eval_sample.jsonl \
  --config examples/moa_pairwise_rank_fuse.yaml \
  --strategy pairwise_rank_fuse \
  --output pairwise-rank-fuse.json

openfusion evaluate examples/eval_sample.jsonl \
  --config examples/uncertainty_cascade.yaml \
  --strategy uncertainty_cascade \
  --output uncertainty-cascade.json

openfusion evaluate examples/eval_moa_sample.jsonl \
  --config openfusion.yaml \
  --compare-strategies fallback,self_moa,parallel_synthesis \
  --max-total-calls 6 \
  --output equal-budget-comparison.json
```

`--compare-strategies` enforces the same `--max-total-calls` for every listed
strategy. The report always includes a fallback baseline, and includes `self_moa`
when a Self-MoA provider is configured.

## Graders

The default grader is `exact_match`. `regex` applies each case's `answer_regex`
before matching. Optional LLM graders are available:

```bash
openfusion evaluate examples/eval_moa_sample.jsonl \
  --config openfusion.yaml \
  --compare-strategies fallback,parallel_synthesis \
  --grader llm_pairwise_swap \
  --grader-provider local-ollama

openfusion evaluate examples/eval_moa_sample.jsonl \
  --config openfusion.yaml \
  --strategy parallel_synthesis \
  --grader llm_rubric \
  --grader-provider local-ollama
```

`llm_pairwise_swap` judges A/B and B/A orderings when two calls remain in the per-case
budget and the first verdict is valid. It accepts a preference only when both valid
verdicts agree after answer-identity remapping. Insufficient budget or malformed verdicts
are `abstain`; changed preferences are `inconsistent`. Grader calls, tokens, latency,
estimated cost, normalized verdicts, order consistency, provider/model, and self-judge
risk are reported separately from generation work. Single-pass `llm_pairwise` remains
available for exploratory runs.

LLM judge scores are not ground truth. They are model outputs and can be biased,
inconsistent, or contaminated by prompt wording. Use them for triage, then verify
important claims with deterministic task graders, executable tests, citation checks,
or blinded human review.

## Metrics

Reports include:

- `total_examples`, `accuracy`, Wilson 95% accuracy bounds, and per-case results;
- conditional win, tie, and loss rates for LLM pairwise graders in compare mode;
- total and average model calls per example;
- total, average, p50, p95, and p99 end-to-end latency in milliseconds;
- prompt, completion, and total token counts;
- complete estimated cost when every executed call reports usage and both per-provider
  token prices are configured;
- correct answers per call and per 1,000 tokens;
- grader calls, tokens, latency, estimated cost, abstentions, inconsistencies, position
  consistency, and self-judge counts;
- `strategy_failures` and failed provider-call totals.

Set `input_cost_per_million_tokens_usd` and
`output_cost_per_million_tokens_usd` on every used provider for cost-complete totals. A
missing price makes the aggregate cost `null` rather than silently reporting a partial
total. The configured `fusion.max_total_calls` is a hard ceiling; a request or CLI option
may select a lower budget but cannot raise it.

Compare at least:

- accuracy or task-specific quality;
- total model calls;
- total tokens;
- wall-clock latency;
- financial cost;
- failure rate.

## Recommended baselines for papers

For papers, benchmark posts, and release claims, include:

- single best model;
- direct provider route or `fallback`;
- `best_of_n`;
- `self_moa`;
- mixed MoA / `layered_refinement`;
- `pairwise_rank_fuse`;
- `semantic_vote` for short-answer tasks;
- `uncertainty_cascade` for cost-sensitive tasks.

Prefer equal `max_total_calls` comparisons when possible. When budgets differ,
state the exact call budget and token budget for every strategy.

The built-in score is normalized exact match. Use a domain-specific test executor, citation checker, retrieval-grounding grader, or blinded human evaluation for open-ended tasks.

Use `semantic_vote` for concise answers that may be equivalent despite different
wording. Use `pairwise_rank_fuse` for open-ended answers where ranked top-candidate
synthesis is more appropriate than exact voting.
Use `uncertainty_cascade` when cost and latency matter and a cheaper provider can
often answer confidently enough without escalation.

## Recommended Testing Methodology

1. Single-model baselines: evaluate model A alone, model B alone, and the best single model.
2. Same-model test-time compute: evaluate `self_moa` with the best model, and `self_moa_seq` when many samples are used.
3. Mixed-model fusion: use `semantic_vote` for short exact-answer tasks, `parallel_synthesis` for open-ended tasks, and `pairwise_rank_fuse` for candidate ranking plus synthesis.
4. Cascade: use `uncertainty_cascade` for cost- or latency-sensitive use.
5. Equal-budget comparison: compare N calls of the best single model against N calls of mixed fusion.
6. Metrics: report accuracy or win rate with uncertainty, latency p50/p95/p99, total
   calls, tokens, cost coverage, correct answers per call/per 1k tokens, delta versus
   fallback, and delta versus the best single model.
7. Report negative results: if fusion does not help, say so.

OpenRouter Fusion compared solo models, self-fusion, mixed panels, and budget panels. OpenFusion users should reproduce that structure with their own models and tasks, not claim the same scores.

## Contamination

OpenFusion evaluation does not implement web search. If a future deployment enables
web tools or retrieval during evaluation, document excluded domains and blocked
domains before running benchmarks. Evaluation sets should not be fetched, searched,
or leaked to tools that can reveal labels or memorized benchmark pages.
