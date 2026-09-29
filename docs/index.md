# OpenFusion

OpenFusion v0.6.1 is an open-source, OpenAI-compatible runtime for multi-model
orchestration, model fusion, and local benchmark experiments.

It supports local and cloud models through OpenAI-compatible APIs, including Ollama, vLLM, LM Studio, LiteLLM, OpenRouter-compatible endpoints, and other `/v1/chat/completions` servers.

OpenFusion is designed for users who want to test whether model fusion actually improves
quality under a hard model-call ceiling and per-call token limits, while measuring the
resulting latency, token, and cost trade-offs.

## What OpenFusion does

Read the [prospective MoA deployment study](research/prospective-moa-deployment.md)
for a fixed workflow tested on 500 previously unused programming tasks, its registered
comparison against Qwen repair, and the resulting deployment recommendation.

- Runs single-model and multi-model strategies.
- Supports local Kev decision models for candidate selection through the System One API.
- Supports Self-MoA, semantic voting, parallel synthesis, pairwise rank fusion, and uncertainty cascades.
- Compares fusion strategies against fallback when configured and every direct model; its built-in
  "best single" is the best observed on that run's evaluated split.
- Reports accuracy with Wilson intervals, end-to-end latency, calls, tokens, optional
  complete-cost estimates, and improvement/regression.
- Audits mixed panels with selection-oracle accuracy, all-model co-failure, disagreement, and
  marginal-contribution diagnostics.
- Offers order-balanced LLM pairwise grading with explicit abstention and inconsistency.
- Enforces a hard administrator call ceiling before scheduling provider work.
- Helps users avoid unsupported claims that fusion always improves results.

## What OpenFusion is not

OpenFusion is not the original Together AI Mixture-of-Agents implementation.

OpenFusion is not Sakana Fugu or a trained reinforcement-learning orchestrator.

OpenFusion is not a replacement for LiteLLM. LiteLLM is better suited for provider management, routing, virtual keys, budgets, and enterprise gateway features. OpenFusion focuses on model-fusion experiments and evaluation.

## Recommended first steps

1. Install OpenFusion.
2. Start Ollama.
3. Pull two small models.
4. Run a local benchmark with OpenFusion Lab.
5. Compare strategies against each base model. Select a deployable best-single baseline
   on separate validation data when making a test-set claim.
6. Interpret accuracy and its interval together with wall latency, calls, token use,
   configured cost coverage, and panel complementarity.
