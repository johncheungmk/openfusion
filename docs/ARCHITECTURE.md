# OpenFusion architecture

OpenFusion is API-level model fusion. It does not merge neural-network weights. Instead, it runs multiple models through compatible chat-completion APIs and fuses their outputs.

## Components

1. **Providers**
   - Any service exposing `POST /v1/chat/completions`.
   - Examples: Ollama, LM Studio, vLLM, llama.cpp server, OpenAI, OpenRouter, Azure OpenAI behind a compatible proxy.

2. **Fusion engine**
   - `panel_judge`: sends the same prompt to a panel of models in parallel. A judge model compares and synthesizes.
   - `fallback`: tries models in order and returns the first successful answer.
   - Provider `weight` values are advisory in the MVP and are included in the judge prompt as hints.
   - Candidate answers are truncated before judge synthesis using `fusion.judge_candidate_max_chars`.

3. **OpenAI-compatible server**
   - Provides `/v1/chat/completions` and `/v1/models`.
   - Applications can point existing OpenAI SDK clients to OpenFusion.
   - Model IDs of the form `provider/{name}/{model}` route directly to a single enabled provider.

4. **CLI**
   - `openfusion chat` for quick experiments.
   - `openfusion serve` for API serving.

## Why API-level fusion first?

Weight-level model merging is powerful but depends on model architecture compatibility, GPU memory, storage, licensing, and evaluation pipelines. API-level fusion is easier for an institution or developer to test because it can combine local and cloud models without retraining.

## Future roadmap

- Streaming responses.
- Cost-aware routing.
- Evaluation harness with golden answers.
- RAG-aware fusion where different models use different retrievers.
- Policy routing for private-data vs public-data prompts.
- Optional integration with weight-merging tools such as mergekit for advanced users.
