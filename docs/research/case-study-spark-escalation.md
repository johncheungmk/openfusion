# Case study: panel fusion and gated escalation on two DGX Sparks

This page reports a complete OpenFusion Lab experiment run on real hardware, including a
negative result, a diagnosis, and the escalation policy that turned it into a measurable
gain. It follows the reporting policy in [Reporting Results](reporting-results.md):
dataset and configuration are published, uncertainty is reported, and the negative results
are shown alongside the improvement.

All numbers below are measured, not illustrative. Re-run instructions are at the bottom.

## Setup

- **Hardware.** Two NVIDIA GB10 DGX Spark units (unified memory), connected over a
  ConnectX-7 RoCE link. A live `sglang` tensor-parallel service was already resident on
  both nodes and was **not** disturbed; the experiment ran entirely in the remaining
  memory and over the LAN.
- **Panel (five local models, four vendors, 1B–4B parameters) via Ollama:**

  | Provider   | Model                | Spark | Params |
  |------------|----------------------|-------|--------|
  | qwen3-4b   | qwen3:4b-instruct    | 1     | 4B     |
  | llama32-3b | llama3.2:3b          | 1     | 3B     |
  | gemma3-4b  | gemma3:4b            | 1     | 4B     |
  | phi4-mini  | phi4-mini:3.8b       | 2     | 3.8B   |
  | llama32-1b | llama3.2:1b          | 2     | 1B     |

  A sixth, much larger flagship model served on `sglang` (OpenAI-compatible) is used only
  as an escalation target, never as a routine panel member.

- **Benchmark (OFBench).** 94 items generated deterministically (seed `20260928`) with
  programmatic gold answers across 11 subjects: arithmetic, algebra, percent, logic,
  combinatorics, probability, sequence, geometry, units, cs/number-base, and knowledge.
  The set is split into 24 dev / 70 test with a fixed stride; the split is used once for
  calibration and once for held-out evaluation respectively. Multiple-choice answers are
  graded with `answer_mode: exact_or_regex` and a letter regex.

- **Budget.** `max_total_calls: 10`, `max_tokens: 32`, `temperature: 0`, `seed: 42`.

!!! note "Hardware gotcha that mattered"
    Several panel models advertise 128K–256K context windows. Loaded at their default
    context next to the resident `sglang` service, they exhausted unified memory and hung
    for over 60 s. We pinned each model's default context to 8192 with a `Modelfile`
    (`PARAMETER num_ctx 8192`) — no `sudo`, no service restart, fully reversible. The same
    lesson applies to any memory-constrained local deployment; see
    [Troubleshooting: slow models](../troubleshooting/slow-cpu-models.md).

## Result 1 — the naive panel fusion REGRESSED

Single-model baselines and equal-weight strategies on the **dev** split (`dev_v1.json`):

| Method                       | Accuracy | Avg calls | Δ vs best single |
|------------------------------|:--------:|:---------:|:----------------:|
| qwen3-4b (best single)       |  79.2%   |   1.00    |       —          |
| fallback                     |  79.2%   |   1.00    |    +0.0 pp       |
| parallel_synthesis           |  79.2%   |   6.00    |    +0.0 pp       |
| uncertainty_cascade          |  79.2%   |   1.00    |    +0.0 pp       |
| **majority_vote**            |  62.5%   |   5.00    |  **−16.7 pp**    |
| **weighted_vote (equal)**    |  62.5%   |   5.00    |  **−16.7 pp**    |
| **semantic_vote**            |  62.5%   |   5.00    |  **−16.7 pp**    |

The OpenFusion Lab recommendation correctly reported
`Fusion is not recommended for accuracy on this dataset`.

The panel-complementarity diagnostic located the headroom precisely: **oracle accuracy was
87.5%** over the same 24 items, and marginal oracle contribution was concentrated in
`qwen3-4b` (+16.7 pp) and `llama32-1b` (+8.3 pp) while `llama32-3b`, `gemma3-4b` and
`phi4-mini` added **zero** marginal correct answers. The panel was complementary but the
equal-weight **aggregator** was the problem, not the panel.

## Result 2 — the regression is statistically real, and calibration only recovers

We confirmed the regression held out, and tested cheaper aggregators, with 5-fold
cross-validation (94 items) and exact McNemar tests against the best single model:

| Policy                         | CV accuracy | McNemar vs best single |
|--------------------------------|:-----------:|:----------------------:|
| Best single (anchor)           |   76.6%     |          —             |
| Equal-weight panel vote        |   67.0%     |  +2 −11, **p = 0.023** |
| Reliability-weighted vote      |   75.5%     |  +0 −1, p = 1.00       |
| Subject-expertise vote (dev-fit)|  76.6%     |  +0 −0, p = 1.00       |
| Self-MoA self-consistency (5×) |   76.6%     |  +0 −0, p = 1.00       |
| Oracle ceiling                 |   89.4%     |  +12 −0, p = 0.0005    |

Three findings, each matching a distinct 2026 result:

1. **Equal-weight mixing genuinely hurts.** The −13-point regression on the test split is
   significant (exact McNemar p = 0.049; pooled CV p = 0.023). This reproduces the
   "Self-MoA" and "anchor corruption" findings on-device — see
   [Related Work](../research/related-work.md).
2. **Static reliability/expertise weighting fully closes the regression but cannot beat the
   anchor** (p = 1.00). The +12.8-point oracle gap is blocked by the *selection*
   bottleneck, not by panel quality.
3. **Repeated sampling was wasted budget here.** Five samples at temperature 0.7 changed
   the anchor's majority answer on **zero** items, because small-model errors on
   programmatic items are systematic, not random. Self-consistency helps when errors are
   high-variance; it does nothing when they are stable.

## Result 3 — the improvement: confidence-gated escalation

The productive move was not to fuse the small models harder, but to escalate selectively to
the flagship model only when the small-model consensus is weak.

Policy **G(τ)** — fit on the dev split only, evaluated once on held-out test:

1. Query three local models (`qwen3-4b`, `llama32-3b`, `phi4-mini`): 3 local calls.
2. Compute consensus = (sum of calibrated reliability weights on the winning answer) /
   (sum over all votes).
3. If consensus < τ, escalate with one flagship call and take its answer; otherwise keep
   the local consensus answer.

Dev fitting selected γ = 0.5, τ = 0.70. Held-out **test** results (n = 70):

| Policy                     | Test accuracy        | Flagship calls / item |
|----------------------------|:--------------------:|:---------------------:|
| Best single small model    |  75.7%               |  0.00                 |
| Equal 5-panel vote         |  62.9%               |  0.00                 |
| **Gated escalation**       |  **91.4%**           |  **0.40**             |
| Flagship always on         |  98.6%               |  1.00                 |

Key comparisons (exact McNemar on the test split):

- Gated escalation vs best single small model: **+11 −0, p = 0.0010** — a significant
  accuracy gain.
- Gated escalation vs flagship-always: +1 −6, p = 0.125 — **not significantly worse**, at
  40% of the flagship calls.

Gate behaviour on the test items: 11 escalations fixed an anchor error, 17 spent a
flagship call on an item the anchor already had right, and 6 low-consensus items were
still missed. The gain comes from correctly detecting that the small panel is out of its
depth and handing off, rather than from averaging small-model opinions.

## What this demonstrates about OpenFusion

- Lab surfaced the negative result and the diagnosis (marginal-oracle contribution) instead
  of hiding it, which is what pointed the experiment toward escalation instead of fusion.
- The whole study — five models across two Sparks, a live service left untouched, 120 s
  baseline collection, ~50 s of extra calls for the improvement — is exactly the local,
  falsifiable loop OpenFusion is built for.
- The ceiling is honest: the flagship model is near-perfect on this suite, so the
  interesting question is *when to call it*, and OpenFusion's cascade + calibration story
  answers that.

## Reproduce

Config and result cards live under `examples/` and `results/` in the repository:

```bash
openfusion lab run examples/ofbench_spark_lab.yaml --out ofbench_results.json
openfusion lab recommend ofbench_results.json
```

The gated-escalation evaluation is a short offline script over a single model-call log
(`ofb_answers.jsonl`, `ofb_flagship.jsonl`); it is deterministic given those calls and adds
no model calls of its own. See the reproducibility notes in the paper for the exact
prompts, splits, and hashes.
