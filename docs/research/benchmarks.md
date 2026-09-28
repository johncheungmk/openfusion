# Benchmarking OpenFusion

An OpenFusion benchmark asks whether a workflow improves a defined task under a defined
budget. It does not ask whether "fusion" is better in the abstract.

At minimum, preserve the exact dataset snapshot, provider and model identifiers, prompts,
strategy configuration, random seed where supported, grader, and call/token limits. Hosted
models can change behind a stable alias, so a date and provider-reported model identifier
are part of the result.

## Current support

OpenFusion v0.6.0 provides local JSONL evaluation, direct-provider and fallback baselines,
exact/regex grading, optional LLM pairwise or rubric grading, calls, end-to-end latency,
token totals, failures, result-card hashes, and a hard `max_total_calls` administrator
ceiling. The v0.6 reliability additions are:

- order-balanced pairwise judging with an explicit inconsistent/abstain result;
- Wilson confidence intervals for accuracy, oracle accuracy, and co-failure;
- provider-configured input/output token prices and a complete estimated run-level cost
  only when every executed call reports usage and has both prices configured;
- a panel complementarity report with oracle accuracy, co-failure, pairwise disagreement,
  and marginal provider contribution.

Lab emits these additions in `openfusion-lab-result-v2`; v1 result cards remain readable.

## Choose a benchmark with a checkable target

Prefer an objective grader whenever the task permits one:

| Task | Preferred grader | Secondary analysis |
|---|---|---|
| Multiple choice / classification | Exact normalized label | Confusion matrix by class |
| Short-answer QA | Exact or task-specific regex | Manual audit of normalization errors |
| Code generation | Sandboxed hidden tests | Compile rate and failure taxonomy |
| Structured extraction | Schema validation plus field comparison | Per-field precision and recall |
| Mathematics | Exact final answer or symbolic equivalence | Reasoning-format diagnostics |
| Open-ended assistance | Blinded human preference or audited LLM judge | Rubric dimensions and judge agreement |

[LiveBench](https://arxiv.org/abs/2406.19314) (ICLR 2025) is a useful design
reference because it refreshes questions and uses objective ground truth to reduce both
contamination and judge bias. [LiveCodeBench](https://arxiv.org/abs/2403.07974)
(ICLR 2025) applies a related approach to code using newly collected contest problems and
execution-grounded scenarios. OpenFusion tests must remain offline; users should import a
versioned local snapshot rather than make benchmark downloads part of the test suite.

For routing research, [RouterBench](https://arxiv.org/abs/2403.12031) (preprint,
2024-03-18) and [LLMRouterBench](https://arxiv.org/abs/2601.07206) (preprint,
2026-01-12) provide outcome matrices across model pools. Their static prices, models, and
latencies are examples, not current production truth.

For judge audits, [RewardBench 2](https://openreview.net/forum?id=fb0G86Dewb)
(ICLR 2026 Poster; preprint first posted 2025-06-02) supplies difficult preference pairs and relates reward-model accuracy to
downstream best-of-N performance. It should complement, not replace, task-specific human
or objective validation.

## Required baselines

For each dataset, run as many of these as the task supports:

1. Every panel provider directly.
2. The best single provider selected on validation data, not the test split.
3. `fallback` with the production provider order.
4. Same-model sampling: `best_of_n` or `self_moa`.
5. The mixed-provider strategy under study.
6. A cheaper strategy at the same quality target.
7. A stronger strategy at the same call, token, latency, or dollar budget.

The best single-model baseline is essential. A panel can beat its first provider while
still losing to another member that should simply have been used directly.

## Define "equal budget" before running

Call equality alone is often misleading: one call may use a much larger model or generate
far more tokens. Report all applicable constraints:

- maximum model calls per request;
- input, output, cached, and reasoning-token totals when the provider exposes them;
- maximum completion tokens for every stage;
- elapsed latency and concurrency;
- provider price snapshot and estimated or invoiced cost;
- hardware and quantization for local models.

If prices are configured, estimate a provider call as:

```text
cost = input_tokens  * input_price_per_token
     + output_tokens * output_price_per_token
```

Store input and output rates separately. Do not hard-code a global price table: API prices,
cached-token discounts, and model aliases change independently of OpenFusion releases.
Label cost as an estimate unless it is reconciled with a provider invoice. Cached,
reasoning, image, and other separately billed token classes are not yet priced by the
built-in estimator.

## Pair prompts and outcomes

All compared strategies should receive the same ordered example set. Preserve one row per
example and strategy so statistical tests use paired outcomes. Do not compare percentages
from different random subsets.

For stochastic workflows, either:

- fix a supported seed and report that the provider honored it; or
- run multiple independent replicates and report between-run variation.

A seed is not a cross-provider reproducibility guarantee.

## Reliable pairwise judging

One fixed A/B presentation confounds answer identity with prompt position. Select
`llm_pairwise_swap` to evaluate both orders when the per-case budget permits:

```text
pass 1: judge(A, B)
pass 2: judge(B, A)
```

After remapping the second verdict to the original identities:

- matching directional verdicts produce a win or loss;
- two ties produce a tie;
- a malformed response produces `abstain`;
- any valid disagreement after identity remapping produces `inconsistent`, not a silent tie.

When two calls remain and the first verdict is valid, OpenFusion records both normalized
verdicts, judge provider/model, and whether the judge generated either candidate. An
insufficient budget abstains before judging; a malformed first verdict abstains without
spending the second call because consistency can no longer be established. This responds
directly to position-bias evidence from the
[IJCNLP-AACL 2025 study](https://aclanthology.org/2025.ijcnlp-long.18/) and
self-preference evidence from
[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html).
Order balancing reduces one known bias; it does not make an LLM verdict ground truth.

For multilingual data, report language-level agreement. Findings of
[EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.587/) found low
cross-language judge consistency, especially for lower-resource languages.

## Confidence intervals and paired tests

Every accuracy or pairwise rate should include its numerator, denominator, and interval.
OpenFusion v0.6 reports a two-sided 95% Wilson score interval for evaluation/Lab accuracy
and the Lab oracle/co-failure metrics.
Wilson intervals behave better than the simple normal interval for small samples and rates
near zero or one.

Do not treat overlapping or non-overlapping marginal intervals as a paired significance
test. When comparing two strategies on the same examples, also report a paired method such
as McNemar's test for binary correctness or a paired bootstrap for a more general metric.
Pre-register the primary metric when a benchmark will support a public claim.

Small samples should produce a warning, not a confident recommendation. A point estimate
of 80% from 10 examples is not equivalent evidence to 80% from 1,000 examples.

## Panel complementarity

The Lab v2 complementarity report operates on the direct-provider correctness matrix.
For each example and provider, it records whether the direct answer was correct, then
derives:

- **best-observed-single accuracy**: the strongest individual provider on the evaluated
  Lab split;
- **selection-oracle accuracy**: fraction where at least one direct panel member is correct;
- **co-failure rate**: fraction where every panel member is wrong;
- **pairwise disagreement**: how often two providers have different correctness outcomes;
- **marginal oracle contribution**: oracle-accuracy loss when one provider is removed.

Selection-oracle accuracy is a ceiling only for policies restricted to selecting one
direct member answer; it is a diagnostic reference, not a ceiling, for a generative fuser
that can construct a new answer. A real selector does not know the correct member in
advance. Report a binomial interval for co-failure and selection-oracle accuracy. A
provider with negligible marginal contribution may still add useful style or safety
behavior, but accuracy data alone does not demonstrate it.

Because Lab computes `best_single_accuracy` on the same split shown in the report, treat it
as descriptive. For an inferential test-set comparison, select the single provider on a
separate validation split and carry that fixed choice into the test run.

## Quality, cost, and latency form a frontier

Avoid collapsing every result into one undocumented "balanced" number. Plot or tabulate
the Pareto frontier over:

- task quality;
- estimated dollar cost;
- calls and tokens;
- p50, p95, and preferably p99 latency;
- failure rate.

The preferred configuration depends on the deployment constraint. A strategy dominated on
all measured axes should not be recommended. A weighted summary is acceptable only when
its normalization, weights, and candidate set are included in the result card.

## Minimum reproducibility manifest

Publish the following with any benchmark claim:

```yaml
dataset_name: example-benchmark
dataset_revision: immutable-id-or-date
dataset_hash: sha256:...
openfusion_version: 0.6.0
config_hash: sha256:...
provider_model_ids: []
run_started_at: 2026-07-10T00:00:00Z
strategies: []
max_total_calls: 6
max_tokens: 256
grader: exact_match
price_snapshot_at: null
replicates: 1
```

Never include API keys, private endpoints, `.env`, or a real `openfusion.yaml` in an
artifact.

## Claim checklist

Before writing "strategy X improves quality," verify that:

- the test data was not used to choose the provider panel or thresholds;
- every baseline received the same examples;
- budgets and failures were included rather than silently filtered;
- objective grading was used where possible;
- judge order, identity, rubric, and abstentions were disclosed otherwise;
- intervals and paired comparisons support the claimed difference;
- the result names the exact models and evaluation date;
- the claim is limited to the evaluated task and budget.

## Published case study

A complete worked example — five local Ollama models across two DGX Spark machines, a
negative fusion result, calibration recovery, and a confidence-gated escalation policy
(+15.7 pp, exact McNemar p = 0.001 on a held-out split) — is documented in the
[DGX Spark case study](case-study-spark-escalation.md). It is the reference pattern for
reporting your own Lab experiments.
