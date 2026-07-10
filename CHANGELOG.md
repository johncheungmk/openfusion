# Changelog

## 0.6.0

- Added the order-balanced `llm_pairwise_swap` grader with explicit abstention and
  inconsistency outcomes, judge telemetry, and self-judge disclosure.
- Stopped treating malformed pairwise-judge output as a tie or counting ties as correct.
- Added Wilson 95% accuracy intervals and small-sample warnings to evaluation reports.
- Added OpenFusion Lab result-card schema v2 with Wilson intervals, p99 latency, cost
  telemetry, and backward-compatible v1 loading.
- Added panel-complementarity diagnostics: oracle accuracy, all-model co-failure,
  pairwise correctness disagreement, and marginal oracle contribution.
- Added optional per-provider input/output pricing and complete-cost reporting for calls,
  workflows, graders, evaluations, and Lab runs.
- Preserved provider-reported model identifiers in traces and judge provenance when the
  upstream API supplies them.
- Added explicit workflow failure status and failed-call totals to evaluation and Lab
  reports instead of treating handled all-provider failures as normal answers.
- Normalized exceptions from custom provider adapters into failed, budget-counted model
  calls and prevented failed sentinel text from being graded as correct.
- Corrected efficiency metrics to report correct answers per call and per 1,000 tokens.
- Corrected evaluation and Lab latency to use end-to-end wall time rather than summed
  provider latency, which overstated parallel-workflow latency.
- Added UTF-8 BOM-safe loading to the standalone evaluation JSONL path.
- Enforced configured `max_total_calls` as a hard administrator ceiling and bounded task
  creation before scheduling, including lazy pair generation.
- Rejected unsupported `n > 1` requests instead of silently paying for and discarding
  extra upstream choices.
- Tightened enabled-provider and provider-name validation.
- Rejected credential-bearing, query, or fragment base URLs; omitted private engine URLs
  from shareable Lab cards; and rejected unknown Lab strategy options.
- Preserved Lab recommendation settings in v2 cards and made the configured objective and
  latency cap control the primary recommendation.
- Excluded zero-success and all-failure runs from the primary Lab recommendation, while
  retaining their descriptive objective metrics.
- Stopped relabeling the first configured Lab strategy as a fallback baseline when no
  `fallback` run was requested.
- Strengthened candidate-as-untrusted instructions across selector, critic, reviser,
  refinement, ranker, and equivalence prompts; workflow-output suppression now also hides
  judge analysis and plan rationale.
- Replaced the misplaced no-op workflow with real compile, lint, offline test, package,
  whitespace, and strict documentation checks.
- Updated the research and benchmarking guidance with primary sources through 2026-07-10.

## 0.5.2

- Added direct per-provider baseline reporting in OpenFusion Lab.
- Added best single-model baseline comparison.
- Added percentage-point and relative improvement metrics.
- Improved Lab recommendation wording by objective.
- Fixed Windows UTF-8 BOM JSONL loading.
- Added OpenFusion Lab cookbook with PowerShell and Linux/macOS examples.
- Added baseline-vs-fusion testing methodology documentation.

## 0.5.1

- Fixed Python package metadata version mismatch in pyproject.toml.
- No runtime behavior changes.
## 0.5.0

- Added OpenFusion Lab for local model-fusion experiments.
- Added lab.yaml schema and validation.
- Added generated OpenFusion config from lab definitions.
- Added local result-card JSON format.
- Added strategy comparison reports and recommendations.
- Added Hugging Face model search helper.
- Added engine launch guidance for Ollama, vLLM, and TGI.
- Added MiniBench sample dataset.

## 0.4.0

- Added first-class `self_moa` and `self_moa_seq` strategies.
- Added model IDs `openfusion/self-moa` and `openfusion/self-moa-seq`.
- Added Self-MoA config and request overrides for provider, sample count, mode, batching, carry size, and sampling temperature.
- Added role-diverse panel prompts via `fusion.panel_roles`.
- Added optional structured synthesis sections via `fusion.structured_synthesis` and `fusion_structured_synthesis`.
- Added `pairwise_rank_fuse` and `semantic_vote` strategies with model IDs.
- Added `uncertainty_cascade` for confidence and consistency based escalation.
- Added equal-budget evaluation comparisons, optional LLM graders, and expanded metrics.
- Preserved `panel_judge` as a backward-compatible alias for `parallel_synthesis`.

## 0.2.1

- Redacted configured header secret values from provider errors.
- Prevented `extra_body` from overriding fixed provider fields such as `model`, `messages`, `temperature`, and `stream`.
- Forwarded common OpenAI Chat Completions tool and log probability fields.

## 0.2.0

- Added best-of-N selection, majority vote, weighted vote, critique–revision, layered refinement, and adaptive planning.
- Renamed the canonical panel workflow to `parallel_synthesis`; `panel_judge` remains an alias.
- Added hard per-request call budgets and bounded public workflow traces.
- Added separate critic, reviser, and planner provider roles.
- Added optional constrained model planning with heuristic fallback.
- Added JSONL exact-match evaluation CLI.
- Added explicit provider timeout and request error messages.
- Expanded OpenAI-compatible strategy model IDs and request extensions.
- Updated documentation for Windows PowerShell and Linux/macOS.

## 0.1.0

- Initial API-level panel synthesis, fallback routing, direct provider routing, CLI, FastAPI server, tests, and Docker support.
