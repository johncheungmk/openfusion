# Changelog

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
