# OpenFusion v0.2 architecture

OpenFusion is an inference-time multi-model orchestration runtime. It does not merge model weights and it does not train a proprietary orchestration policy.

## Request path

```text
OpenAI-compatible request
          |
          v
 model/strategy resolver
          |
          v
 constrained orchestration plan
          |
          +--> independent provider calls
          +--> voting / selection
          +--> critique and revision
          +--> refinement layers
          +--> synthesis
          |
          v
 OpenAI-compatible response + public trace
```

## Components

### Provider adapter

`OpenAICompatibleProvider` sends non-streaming requests to `POST /chat/completions` under a configured `/v1` base URL. Provider credentials are read from environment variables. HTTP clients are pooled by timeout and closed during application shutdown.

### Call budget

Every orchestrated request receives a `CallBudget`. Each provider call must reserve one unit before execution. When the budget is exhausted, the step is recorded as skipped rather than silently creating more cost.

The budget limits model calls, not tokens. Provider-level token, rate, and financial limits should still be enforced by the upstream gateway.

### Independent generation

When a workflow requests more than one sample, OpenFusion inserts a neutral sampling instruction and calls providers independently. Agents do not see peer answers during the initial stage. This preserves diversity before aggregation.

### Aggregation methods

- `parallel_synthesis` asks a synthesis model to create a new answer from independent candidates.
- `best_of_n` asks an evaluator for a strict JSON winner and returns the selected candidate unchanged.
- `majority_vote` and `weighted_vote` group normalized answers. They are intended for concise or regex-extractable answers, not long prose.
- `critique_revision` separates the critic and reviser roles.
- `layered_refinement` exposes one layer's outputs to the next layer before final synthesis.

### Adaptive planning

Adaptive mode has two planner options:

1. **Heuristic planner** — local, deterministic, and free of model calls.
2. **Model planner** — an optional provider returns a constrained JSON plan.

A model plan can select only built-in strategies and enabled provider names. It cannot create Python code, arbitrary tools, endpoints, or recursive adaptive plans. Invalid plans fall back to heuristics. The plan is reduced automatically when its estimated calls exceed the remaining budget.

### Public trace

The response trace records stages, provider/model names, success status, latency, and bounded error messages. It does not request or expose hidden chain-of-thought. Candidate and workflow text can be suppressed independently.

## Strategy call shapes

```text
fallback
  provider A -> success, or provider B -> ...

parallel_synthesis
  draft A --\
  draft B ----> synthesizer -> final
  draft C --/

best_of_n
  candidate A --\
  candidate B ----> evaluator -> return selected candidate
  candidate C --/

critique_revision
  drafts -> critic -> reviser -> final

layered_refinement
  independent layer -> refinement layer(s) -> synthesizer -> final

adaptive
  heuristic/model planner -> one of the bounded workflows above
```

## Deliberate limitations

- Fake streaming emits the completed result as SSE; provider tokens are not streamed through each workflow stage.
- OpenFusion does not execute arbitrary model-requested tools in v0.2.
- Voting uses normalized textual agreement, not semantic clustering.
- The built-in evaluator is exact match and is not a general quality judge.
- Adaptive heuristics are transparent rules, not learned reinforcement-learning orchestration.
