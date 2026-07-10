# Evaluation methodology

OpenFusion evaluates a workflow as a system: generators, judges, retries, failures, and
resource use all count. A final answer is not "better" merely because more models produced
it.

## Start with a falsifiable question

A useful experiment has a task, a primary metric, a budget, and a baseline:

> On this frozen short-answer dataset, does `semantic_vote` improve exact-match accuracy
> over the best direct provider when both are limited to four model calls?

Avoid questions such as "does fusion work?" They combine unrelated tasks and leave the
budget undefined.

## Current v0.6.0 evaluation path

OpenFusion currently supports:

1. JSONL cases with references, optional regexes, and optional rubrics.
2. Direct-provider, best-single-model, and workflow comparisons in OpenFusion Lab.
3. Exact, regex, LLM pairwise, order-balanced LLM pairwise, and LLM rubric graders.
4. Calls, end-to-end latency, prompt/completion token totals, optional estimated cost,
   and failures.
5. A hard per-request `max_total_calls` workflow budget.
6. Wilson 95% accuracy intervals and explicit small-sample warnings.
7. Dataset and configuration hashes in Lab result cards.
8. Direct-provider panel-complementarity diagnostics in Lab.

These facilities make local comparisons reproducible, but they do not remove benchmark
contamination, judge bias, sampling variance, or provider drift.

## Baseline ladder

Run the least complicated credible alternatives first:

1. Each provider directly.
2. Best single provider, selected without using the test labels.
3. `fallback`.
4. Same-model test-time compute: `best_of_n` or `self_moa`.
5. The mixed-provider workflow under study.
6. Equal-call, equal-token, or equal-cost alternatives.

If a complex workflow cannot beat the best direct provider at an acceptable budget, that
is a useful negative result.

## Grader hierarchy

Use the most objective applicable grader:

1. Execution, schema, symbolic, exact, or regex checks.
2. Task-expert human review with blinded answer identity.
3. Calibrated and audited LLM judging.
4. Unaudited single-pass LLM judging only for exploratory work.

An LLM judge approximates a preference; it does not establish factual correctness.
[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)
documents position, verbosity, self-enhancement, and reasoning biases, and
[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html)
shows that evaluators can favor their own generations.

## Order-balanced judges and abstention

Use `--grader llm_pairwise_swap --grader-provider PROVIDER` to send the same pair in up
to two position orders. Both calls run when two calls remain in the per-case budget and
the first verdict is valid; a malformed first verdict abstains immediately because a
second verdict cannot make the pair consistent. Responses are remapped to stable
candidate identities.
A win, tie, or loss is accepted only when both valid judgments agree after remapping.
Contradictory verdicts become `inconsistent`; malformed or missing verdicts become
`abstain`. The legacy `llm_pairwise` mode remains available for lower-cost exploratory
checks, but malformed output now abstains instead of silently becoming a tie.

Each result record includes:

- up to two normalized order-specific verdicts;
- order consistency;
- judge provider and provider-reported model;
- whether the judge produced either candidate;
- an abstention or inconsistency reason.

This protocol is motivated by the 15-judge, 150,000-evaluation
[position-bias study](https://aclanthology.org/2025.ijcnlp-long.18/) published at
IJCNLP-AACL 2025. It mitigates position bias but cannot eliminate self-preference, style,
language, or rubric bias. Important evaluations should also sample human-reviewed cases
and, where affordable, compare independent judge families.

## Wilson intervals

Point estimates hide sample size. Evaluation and Lab summaries attach two-sided 95%
Wilson score intervals to accuracy. Lab complementarity reports also include them for
oracle accuracy and all-model co-failure.

Every reported proportion should retain:

```text
successes / eligible_examples, point_estimate, confidence_interval
```

Abstentions must remain visible. Report both the abstention rate and the conditional win
rate among decided pairs; do not silently count every judge failure as a tie or delete it.

For two strategies evaluated on the same examples, a paired comparison is still needed.
Use McNemar's test for binary correctness or a paired bootstrap for more general scores.
Wilson intervals on two separate rates are not a substitute for a paired test.

## Cost accounting

Calls and tokens are useful but do not equal cost. Each provider can define
`input_cost_per_million_tokens_usd` and `output_cost_per_million_tokens_usd`. OpenFusion
then reports per-call and aggregate `estimated_cost_usd` values. An aggregate is `null`
unless every executed call in that scope has both prices configured, so a partial price
table cannot masquerade as a complete total.

Cost accounting follows three rules:

1. Users supply the pricing snapshot; OpenFusion does not hard-code a supposedly current
   cloud price table.
2. Input and output token classes remain separate.
3. Results say `estimated_cost` unless reconciled with a provider invoice.

Local models should report hardware, quantization, wall time, and token counts. A dollar
estimate for local inference is optional because electricity and amortization assumptions
vary.

The existing `max_total_calls` remains a hard safety bound even when token, latency, or
dollar budgets are added later.

## Panel complementarity

A mixed panel is promising only when its members succeed on different examples and a
selector can exploit those differences. Lab result-card schema v2 derives the following
from direct-provider baseline outcomes:

| Metric | Meaning | Important caveat |
|---|---|---|
| Best-observed-single accuracy | Highest individual score on this Lab run | Descriptive only; select on independent validation data for a test claim |
| Selection-oracle accuracy | At least one direct provider was correct | Ceiling only for policies that select one member answer |
| Co-failure rate | Every provider was wrong | Applies directly only to the evaluated task |
| Pairwise disagreement | Providers differ in correctness | Disagreement alone does not identify the correct one |
| Marginal contribution | Oracle loss when a provider is removed | Does not measure style or safety diversity |

The report includes Wilson intervals for selection-oracle accuracy and all-model co-failure.
The selection oracle is a diagnostic reference, not a ceiling for generative synthesis,
which may construct an answer not present among the direct outputs.
It is a panel-curation aid, not evidence that the runtime can reach oracle performance.

## Evaluate uncertainty cascades as selective systems

`uncertainty_cascade` currently parses a model's verbal confidence and can combine it with
multiple-sample consistency. The score is not automatically a probability.

[ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6733cf15e10e2cd1d59af033c3bb8507-Abstract-Conference.html)
finds that verbalized confidence is often overconfident and task dependent. Evaluate a
cascade with:

- expected calibration error or a reliability diagram;
- Brier score where a binary correctness target exists;
- failure-prediction AUROC;
- risk versus coverage at each threshold;
- escalation rate, cost, latency, and final task accuracy.

Do not tune a confidence threshold on the test split. For black-box providers, combine
held-out calibration with consistency and objective validation rather than trusting one
self-reported number.

## Compare a frontier, not one score

For each strategy report:

- primary task quality;
- calls and input/output tokens;
- estimated financial cost when configured;
- average, p50, p95, and preferably p99 latency;
- provider and workflow failure counts;
- grader abstentions and agreement;
- result intervals.

A strategy is Pareto-dominated when another measured configuration is at least as good on
every relevant axis and strictly better on one. Prefer a frontier table or plot. If a
single balanced score is used, publish its formula, normalization, and weights.

Lab's `balanced_score` is a run-local heuristic, not a portable benchmark metric:

```text
accuracy
- 0.05 * max(0, latency_ratio_vs_fallback - 1)
- call_weight * max(0, calls_ratio_vs_fallback - 1)
- failures / examples
```

`call_weight` is `0.03` when `prefer_lower_calls: true` and `0.01` otherwise. Ratios use
the fallback result when present. The configured `objective` chooses the primary
recommendation, and `max_latency_ms` removes slower strategies from that choice; the card
still reports all four objective leaders. The formula currently has no token or dollar
cost term, so use the Pareto table for cost-sensitive decisions.

## Reproducibility and leakage controls

Before a run:

- freeze or hash the dataset;
- separate panel/threshold selection data from test data;
- record prompt and rubric versions;
- pin explicit model identifiers where the provider allows it;
- record the evaluation date and local hardware;
- decide how malformed outputs and timeouts will be scored.

After a run:

- retain every failed example;
- publish aggregate and per-example artifacts without secrets or private prompts;
- disclose all tested strategies when making a selected-best claim;
- scope conclusions to the evaluated data, models, grader, and budget.

Frequently refreshed, objectively graded designs such as
[LiveBench](https://arxiv.org/abs/2406.19314) (ICLR 2025) reduce contamination
risk, but no public benchmark proves performance on a private deployment distribution.

## Interpretation template

A defensible conclusion looks like this:

> On dataset revision X, strategy A scored 74/100 (Wilson 95% CI reported in the result
> card) versus 68/100 for the paired best-single baseline, using at most six calls per
> example. It used 2.1x tokens and had 1.7x p95 latency. The result applies to the named
> model versions and grader; it is not a general claim that A is better on other tasks.

See [Benchmarking OpenFusion](../research/benchmarks.md) for the full experiment and claim
checklists.
