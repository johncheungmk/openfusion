# OpenFusion Lab

OpenFusion Lab is a local experiment runner for comparing single-model and fusion strategies under the same dataset, model set, hardware, and call budget.

It does not launch model servers. Start engines manually, validate the lab file, run the experiment, then inspect the result card.

## Workflow

1. Choose OpenAI-compatible engines and models in `lab.yaml`.
2. Start each engine manually.
3. Generate a runtime config:

```bash
openfusion lab generate-config examples/lab_llama_gptoss.yaml --out openfusion.lab.generated.yaml
```

4. Validate the lab:

```bash
openfusion lab validate examples/lab_llama_gptoss.yaml
```

5. Run direct per-provider baselines and strategies:

```bash
openfusion lab run examples/lab_llama_gptoss.yaml --out results.json
```

6. Read recommendations:

```bash
openfusion lab recommend results.json
```

7. Export a normalized result card:

```bash
openfusion lab export results.json --out result-card.json
```

New result cards use `schema_version: openfusion-lab-result-v2`; v1 cards still load with
backward-compatible defaults. Cards omit engine URLs, API-key environment-variable names,
provider headers, raw prompts, and raw references by default. They include direct
baselines, best-observed-single and (when configured) fallback baselines,
strategy comparisons, Wilson accuracy intervals, end-to-end p50/p95/p99 latency,
cost-completeness metrics, objective-specific recommendations, and a panel-complementarity
report. The complementarity report exposes best-observed-single/selection-oracle accuracy, all-model
co-failure, pairwise correctness disagreement, and marginal oracle contribution. Oracle
accuracy is diagnostic and is not a deployable selector score.

The configured recommendation objective and latency cap are preserved in v2 cards. The
latency cap filters the primary recommendation; it does not hide the unfiltered leaders
reported for accuracy, latency, efficiency, and the documented balanced heuristic.

## Recommended Testing Methodology

1. Single-model baselines: run every model directly. Lab reports the best observed on this
   split; select a deployable best single model on separate validation data.
2. Same-model test-time compute: run `self_moa` with the best model, and use `self_moa_seq` if many samples are used.
3. Mixed-model fusion: use `semantic_vote` for short exact-answer tasks, `parallel_synthesis` for open-ended tasks, and `pairwise_rank_fuse` for candidate ranking plus synthesis.
4. Cascade: use `uncertainty_cascade` for cost- or latency-sensitive runs.
5. Equal-budget comparison: compare N calls of the best single model against N calls of mixed fusion.
6. Metrics: report accuracy or win rate with uncertainty, latency p50/p95/p99, total
   calls, tokens, configured cost coverage, correct answers per call/per 1k tokens, delta
   versus fallback, and delta versus the best single model.
7. Report negative results: if fusion does not help, say so.

OpenRouter Fusion compared solo models, self-fusion, mixed panels, and budget panels. OpenFusion users should reproduce the structure for their local model set and dataset, not claim the same scores.

## Dataset Format

Chat-style examples:

```json
{"id":"mmlu-mini-001","messages":[{"role":"user","content":"Answer only A, B, C, or D..."}],"answer":"C","answer_regex":"\\b([ABCD])\\b"}
```

Prompt-style examples:

```json
{"id":"math-001","prompt":"Return only the answer: 2 + 2","reference":"4"}
```

The evaluator supports exact match and regex extraction through `answer_mode: exact_or_regex`.

## Model Discovery

`openfusion lab search-models` queries the Hugging Face model API only when called. It does not require a token, download models, or use `trust_remote_code`.

```bash
openfusion lab search-models --query llama --limit 5 --sort downloads
```

## Engine Guidance

Use:

```bash
openfusion lab engine-plan examples/lab_llama_gptoss.yaml
```

The command prints manual launch guidance for Ollama, vLLM, and TGI. OpenFusion Lab does
not auto-launch engines.
