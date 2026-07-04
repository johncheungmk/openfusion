# Changelog

## 0.4.0

- Added first-class `self_moa` and `self_moa_seq` strategies.
- Added model IDs `openfusion/self-moa` and `openfusion/self-moa-seq`.
- Added Self-MoA config and request overrides for provider, sample count, mode, batching, carry size, and sampling temperature.
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
