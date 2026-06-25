# Codex CLI prompt: upgrade an existing OpenFusion v0.1 checkout to v0.2

Paste the following prompt into Codex CLI from the repository root.

```text
Upgrade the existing OpenFusion repository to version 0.2.0. Do not rewrite unrelated files and do not add secrets.

Product direction:
OpenFusion is an open-source, OpenAI-compatible multi-model orchestration and fusion runtime. It should go beyond one fixed panel+judge workflow while remaining transparent, testable, cost-bounded, and honest about not reproducing proprietary trained systems such as Sakana Fugu.

Preserve:
- existing provider configuration and OpenAI-compatible API;
- direct model route provider/{provider_name}/{model};
- fallback behavior;
- panel_judge as a backward-compatible alias;
- fake SSE streaming;
- environment-variable credentials;
- offline tests only.

Implement these canonical strategies:
1. fallback — try providers in order;
2. parallel_synthesis — independent drafts plus a synthesizer that writes a new answer;
3. best_of_n — independent alternatives plus strict-JSON evaluator selection, returning the selected candidate unchanged;
4. majority_vote — normalized exact or regex-extracted voting;
5. weighted_vote — voting weighted by ProviderConfig.weight;
6. critique_revision — independent drafts, a critic role, then a reviser role;
7. layered_refinement — initial independent layer, configurable refinement layers, then synthesis;
8. adaptive — transparent heuristic planning, with an optional constrained model planner.

Adaptive planner safety:
- allow only the built-in strategies above except recursive adaptive;
- allow only enabled provider names;
- validate strict JSON;
- clamp samples to 1..3 and refinement rounds to 0..2;
- fall back to heuristics on invalid output;
- do not execute arbitrary tools, code, URLs, or shell commands;
- expose only a one-sentence operational rationale, not chain-of-thought.

Cost and execution controls:
- add a hard per-request max_total_calls budget;
- every provider call must reserve one budget unit;
- record skipped calls when the budget is exhausted;
- add samples_per_provider, refinement_rounds, judge_temperature, critique_temperature, critic_provider, reviser_provider, planner_provider, include_workflow_outputs, transcript_max_chars, vote_answer_regex, and adaptive_use_model_planner to config with backward-compatible defaults.

Metadata:
- add an orchestration plan and public execution trace to FusionResult;
- trace only stage, provider/model, status, latency, and bounded error/note;
- add critic_provider, reviser_provider, and workflow_outputs;
- preserve candidate redaction and add workflow-output redaction;
- never request or reveal hidden chain-of-thought.

API:
- expose model IDs openfusion/adaptive, openfusion/parallel-synthesis, openfusion/panel-judge, openfusion/critique-revision, openfusion/layered-refinement, openfusion/best-of-n, openfusion/majority-vote, openfusion/weighted-vote, and openfusion/fallback;
- add request extensions fusion_critic, fusion_reviser, fusion_planner, fusion_samples_per_provider, fusion_refinement_rounds, fusion_max_total_calls, and fusion_vote_regex;
- unknown model IDs must return HTTP 400;
- /health should report version, providers, and strategies;
- add /v1/strategies.

Provider hardening:
- catch httpx.TimeoutException separately and return a clear timeout message;
- catch RequestError separately;
- continue redacting credentials from HTTP errors;
- reuse shared HTTP clients and close them on shutdown.

CLI:
- add `openfusion strategies`;
- add `openfusion plan`;
- extend `openfusion chat` with critic, reviser, planner, samples, rounds, max-total-calls, and trace options;
- add `openfusion evaluate` for JSONL normalized exact-match evaluation.

Documentation:
- clearly identify openfusion.yaml as the file users edit after `openfusion init`;
- include both PowerShell and Bash examples;
- explain the difference from a gateway such as LiteLLM;
- document limitations and research positioning;
- add CHANGELOG.md, CONTRIBUTING.md, docs/RESEARCH.md, docs/MIGRATION_V2.md, and docs/EVALUATION.md;
- version package and FastAPI app as 0.2.0.

Tests must cover:
- all new strategies;
- adaptive heuristic and model plans;
- plan validation and call-budget enforcement;
- timeout error messages;
- new API model IDs and unknown-model handling;
- redaction of workflow outputs;
- evaluator normalization and JSONL loading.

After implementation run:
python -m pip install -e ".[dev]"
python -m compileall -q src tests
ruff check src tests
pytest -q
python -m build
openfusion strategies
git diff --check

Fix every failure. Do not make benchmark-performance claims without measured results. Summarize files changed and commands run.
```
