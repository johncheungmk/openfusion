# OpenFusion

OpenFusion is an open-source, OpenAI-compatible runtime for multi-model orchestration, model fusion, and local benchmark experiments.

It supports local and cloud models through OpenAI-compatible APIs, including Ollama, vLLM, LM Studio, LiteLLM, OpenRouter-compatible endpoints, and other `/v1/chat/completions` servers.

OpenFusion is designed for users who want to test whether model fusion actually improves quality under explicit latency, token, and model-call budgets.

## What OpenFusion does

- Runs single-model and multi-model strategies.
- Supports Self-MoA, semantic voting, parallel synthesis, pairwise rank fusion, and uncertainty cascades.
- Compares fusion strategies against fallback and the best single-model baseline.
- Reports accuracy, latency, calls, tokens, and improvement/regression.
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
5. Compare fusion strategies against each base model.
6. Interpret accuracy together with latency, calls, and token usage.
