# Reporting Results

An OpenFusion result is evidence about one named dataset, provider panel, configuration, grader,
and budget. It is not evidence that a strategy is universally better. Publish enough information
for another user to reconstruct the comparison and retain negative or failed cases.

## Required run context

Record at least:

- OpenFusion version, result-card schema version, evaluation date, and dataset revision/hash;
- configuration hash, exact configured model identifiers, and provider-reported model identifiers
  when available;
- strategy settings, panel order, maximum calls, per-call token cap, and concurrency;
- grader mode, rubric or normalization rule, judge provider/model, and whether the judge generated
  either compared answer;
- hardware, serving engine, quantization, and software versions for local inference;
- input/output price rates and their snapshot date when reporting estimated cost.

Do not publish API keys, headers, private endpoints, raw private prompts, or an actual
`openfusion.yaml` file. A hash can identify a private artifact without disclosing it.

## Quality and uncertainty

For objective tasks, report `correct / total`, accuracy, and the two-sided Wilson 95% interval.
Keep per-example outcomes paired across strategies. A difference between two marginal intervals is
not a paired significance test; use a method such as McNemar's test or a paired bootstrap when a
claim depends on the difference.

For pairwise LLM grading, report win/tie/loss rates conditional on decided examples together with:

- the number and rate of decided examples;
- abstention and inconsistency counts/rates;
- position-consistency telemetry for swapped-order judging;
- self-judge counts and a human-reviewed or objective audit sample.

Never recode malformed or contradictory judge output as a tie. Treat judge scores as measurements,
not ground truth.

## Resource and reliability metrics

Report the resource axes separately:

- generation calls and total calls including grader calls;
- prompt, completion, and total tokens;
- end-to-end average, p50, p95, and p99 latency;
- workflow failures and failed provider-call counts;
- estimated cost, priced-example coverage, and grader cost separately;
- correct answers per call and per 1,000 tokens.

Cost is complete only when every executed call has usable token accounting and both configured
rates. Otherwise report `null` for the aggregate rather than a partial sum. Built-in estimates do
not price cached, reasoning, image, or other provider-specific units.

## Baselines and panel diagnostics

Include every direct provider, fallback, same-model sampling, and the fusion strategy under study at
the same stated budget. `best-observed-single` is the highest direct score on the reported split; it
is descriptive and must not be presented as a provider selected independently on validation data.
For a deployable best-single baseline, select the provider on a separate validation split and freeze
that choice before evaluating the test split.

Selection-oracle accuracy and all-model co-failure describe complementarity in the direct-provider
correctness matrix. The selection oracle is a ceiling only for policies restricted to selecting one
member answer. It is a diagnostic reference, not a ceiling, for generative synthesis.

## Built-in balanced score

The Lab balanced score is an interpretation heuristic, not a standardized research metric. Relative
to its comparison baseline, it is:

```text
accuracy
- 0.05 * max(0, latency_ratio - 1)
- call_weight * max(0, call_ratio - 1)
- failure_rate
```

`call_weight` is `0.03` when `prefer_lower_calls: true` and `0.01` otherwise. The score does not
include tokens or financial cost. Publish the formula, baseline, candidate set, and settings whenever
the score is shown, and do not compare scores computed against different baselines.

## Claim template

A bounded conclusion should name both the benefit and its cost:

> On dataset revision X, strategy A scored 74/100 with its Wilson 95% interval, versus
> 68/100 for the preselected single-provider baseline on the same examples. It used at most six
> calls per example, 2.1x the tokens, and 1.7x the p95 latency. The conclusion applies only to the
> named model versions, grader, price snapshot, and evaluation date.

Report regressions and null results with the same detail. They identify tasks where a direct model is
the better deployment choice.
