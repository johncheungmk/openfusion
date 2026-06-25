# Migrating from OpenFusion v0.1 to v0.2

## Compatible behavior

Existing provider blocks remain valid. Existing `panel_judge`, `parallel_judge`, and `fusion` strategy names remain accepted as aliases for `parallel_synthesis`.

Direct provider model IDs remain:

```text
provider/{provider-name}/{configured-model}
```

## Changed metadata

A request using `openfusion/panel-judge` now reports the canonical strategy name:

```json
{"strategy": "parallel_synthesis"}
```

Responses now include `plan`, `trace`, role fields, and optional `workflow_outputs` inside the `openfusion` metadata object.

## New optional config fields

```yaml
fusion:
  critic_provider:
  reviser_provider:
  planner_provider:
  max_total_calls: 12
  samples_per_provider: 1
  refinement_rounds: 1
  judge_temperature: 0.1
  critique_temperature: 0.1
  include_workflow_outputs: true
  transcript_max_chars: 12000
  vote_answer_regex:
  adaptive_use_model_planner: false
```

Defaults allow old valid YAML files to load without these fields.

## Recommended upgrade procedure

1. Back up `.env` and `openfusion.yaml`; both are intentionally ignored by Git.
2. Pull or replace the tracked repository files.
3. Reinstall editable dependencies.
4. Compare your config with `config.example.yaml`.
5. Run `ruff check src tests` and `pytest -q`.
6. Test a direct provider model ID before a multi-call strategy.
