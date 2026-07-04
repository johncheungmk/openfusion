# Evaluation guide

OpenFusion does not assume that more calls produce a better answer. Evaluate each strategy on representative tasks.

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
  --grader llm_pairwise \
  --grader-provider local-ollama

openfusion evaluate examples/eval_moa_sample.jsonl \
  --config openfusion.yaml \
  --strategy parallel_synthesis \
  --grader llm_rubric \
  --grader-provider local-ollama
```

LLM judge scores are not ground truth. They are model outputs and can be biased,
inconsistent, or contaminated by prompt wording. Use them for triage, then verify
important claims with deterministic task graders, executable tests, citation checks,
or blinded human review.

## Metrics

Reports include:

- `total_examples`, `accuracy`, and per-case results;
- win, tie, and loss rates versus the fallback baseline in compare mode;
- total and average model calls per example;
- total, average, p50, and p95 latency in milliseconds;
- prompt, completion, and total token counts;
- `estimated_cost` when cost configuration is added by deployments;
- `accuracy_per_call` and `accuracy_per_1k_tokens`;
- `strategy_failures`.

Compare at least:

- accuracy or task-specific quality;
- total model calls;
- total tokens;
- wall-clock latency;
- financial cost;
- failure rate.

The built-in score is normalized exact match. Use a domain-specific test executor, citation checker, retrieval-grounding grader, or blinded human evaluation for open-ended tasks.

Use `semantic_vote` for concise answers that may be equivalent despite different
wording. Use `pairwise_rank_fuse` for open-ended answers where ranked top-candidate
synthesis is more appropriate than exact voting.
Use `uncertainty_cascade` when cost and latency matter and a cheaper provider can
often answer confidently enough without escalation.

## Contamination

OpenFusion evaluation does not implement web search. If a future deployment enables
web tools or retrieval during evaluation, document excluded domains and blocked
domains before running benchmarks. Evaluation sets should not be fetched, searched,
or leaked to tools that can reveal labels or memorized benchmark pages.
