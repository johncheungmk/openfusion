# Kev selection and verified MoA on NVIDIA Spark

Measured on 2026-09-29. Code, decisions, and fresh-task results are published together.

The [method and model follow-up](moa-method-model-ablation.md) compares model diversity
with repeated sampling at the same proposal budget on a fresh HumanEval subset.

## Findings

**Verifier-assisted MoA improved coding results on the fresh sample: 51/60, compared with
43/60 for the strongest individual component and 45/60 for repeated Qwen repair.**
The mixture had eight wins and no losses against Qwen, and six wins and no losses
against the repair control. This is evidence for this workflow and sampled workload.

**Visible-test filtering followed by Kev matched the rule selector's accuracy:
51/60 versus 51/60.** Its contribution must be judged separately from the
benefit of the mixture itself. The generic Kev integration is available as
[`decision_select`](../strategies/decision-select.md); it is optional.

## Fresh confirmation: identical candidates, different selectors

Sixty previously unused MBPP task IDs were frozen with seed `920260929` before
generating any answers. None overlap the earlier 100 tasks or eight development tasks.
All selector policies were written before their test outcomes were inspected.
No threshold or prompt was tuned on these results.

| Selection policy | Correct | Extra selector calls per task | Selector seconds per task |
|---|---:|---:|---:|
| Qwen alone | 43/60 | 0.00 | 0.000 |
| Gemma alone | 36/60 | 0.00 | 0.000 |
| Phi alone | 40/60 | 0.00 | 0.000 |
| First candidate passing visible test | 51/60 | 0.00 | 0.000 |
| Qwen LLM selector | 45/60 | 1.00 | 0.335 |
| Kev selector | 45/60 | 1.00 | 0.672 |
| Visible-test filter + Kev | 51/60 | 0.88 | 0.563 |

- Visible-test filter + Kev versus rule selection: 0 wins / 0 losses; exact paired McNemar p=1.
- Kev alone versus Qwen alone: 4 wins / 2 losses; exact paired McNemar p=0.6875.
- Kev alone versus the conventional Qwen selector: 1 wins / 1 losses; exact paired McNemar p=1.
- At least one of the three initial proposals was correct on **51/60** tasks. This is a hindsight selection ceiling, not a deployable policy.

The visible-test selector already reached that 51/60 ceiling on this sample. A more
sophisticated selector cannot exceed it without generating or repairing candidates.
The five attempted MoA repairs did not add another success on these fresh tasks.

For selection policies, three proposals are already present. Extra selector calls above
exclude those three generation calls. Direct-model rows require just one generation.
Seconds measure selector requests, not end-to-end generation. The rule selector's check
time is not included. Kev ran on the second Spark, the Qwen judge on the first, with
the pre-existing SGLang service resident. These are operational observations, not an
isolated hardware speed comparison. Kev output tokens describe serialized decisions,
not generated text; token prices were not configured.

## Fresh confirmation: checker plus bounded repair

The unchanged earlier workflow obtains Qwen, Gemma, and Phi proposals. It chooses the
first proposal passing the visible test in that order, or makes one Qwen repair if all
fail. The control starts with Qwen and allows up to three repairs using the same test.
Both have a four-call ceiling. These results include repair; the selector table above does not.

| Workflow | Correct | Mean logical calls | Mean output tokens |
|---|---:|---:|---:|
| `qwen` | 43/60 | 1.00 | 63 |
| `gemma` | 36/60 | 1.00 | 95 |
| `phi` | 40/60 | 1.00 | 46 |
| `verified_moa` | 51/60 | 3.08 | 212 |
| `qwen_verified_repair` | 45/60 | 1.58 | 121 |
| `adaptive_verified_moa` | 51/60 | 1.52 | 102 |

- Verified MoA versus Qwen: 8 wins / 0 losses; exact paired McNemar p=0.0078125.
- Verified MoA versus Phi: 11 wins / 0 losses; exact paired McNemar p=0.000976562.
- Verified MoA versus repeated Qwen repair: 6 wins / 0 losses; exact paired McNemar p=0.03125.

The adaptive variant stops immediately if Qwen passes the visible test. It produces the
same answers as verified MoA. Its logical calls and tokens are reconstructed from shared
recorded requests; it was not separately timed as a deployed pipeline.

## Earlier 100 tasks: retrospective selector comparison

These tasks were already examined in the previous MoA study. They are useful for a
controlled comparison on frozen proposals, but **are not fresh confirmation for Kev**.
Their original splits were 40 pilot tasks and 60 subsequent confirmation tasks.

| Selection policy | Correct | Extra selector calls per task | Selector seconds per task |
|---|---:|---:|---:|
| Qwen alone | 72/100 | 0.00 | 0.000 |
| Gemma alone | 67/100 | 0.00 | 0.000 |
| Phi alone | 72/100 | 0.00 | 0.000 |
| First candidate passing visible test | 80/100 | 0.00 | 0.000 |
| Qwen LLM selector | 78/100 | 1.00 | 0.332 |
| Kev selector | 79/100 | 1.00 | 0.642 |
| Visible-test filter + Kev | 81/100 | 0.89 | 0.557 |

Filtered Kev versus rules on these earlier tasks: 1 wins / 0 losses; exact paired McNemar p=1.
The single extra success is too little evidence for a general accuracy claim.

### Where Kev helped once

On `mbpp-confirm-432`, every proposal passed the visible test for a trapezium's median.
Qwen used `(base1 + base2) // 2`, which incorrectly rounds down fractional results.
Gemma and Phi used `(base1 + base2) / 2`. Kev selected Phi, which passed the withheld
fractional case. The conventional Qwen judge also selected a correct candidate.
This illustrates the useful role: choosing between answers that pass an incomplete
check. It does not establish that Kev does this better than an ordinary LLM judge.

The earlier verifier-plus-repair result was **82/100** versus **72/100** for the best
individual component and **78/100** for Qwen repair. Its four-point advantage over repair
was uncertain (four wins, no losses; p=0.125). The earlier 60-task confirmation tied Phi.
The new 60-task result above is separate; pooling does not erase these differences.

The earlier mathematics experiment also remains relevant: critique/revision scored
96/100 versus Qwen's 95/100 on sampled GSM8K, while self-synthesis scored 95/100.
There is no strong mathematics improvement claim from those results.

## Protocol and scope

- Dataset: [Google Research sanitized MBPP](https://github.com/google-research/google-research/tree/master/mbpp).
  Only tasks supported by the restricted Python evaluator were eligible; imports and
  other unsupported constructs were excluded. This is **not** the full MBPP leaderboard.
- Each prompt includes the function signature and one published test. Remaining published
  tests are withheld from generation, selection, and repair; success requires all tests.
  Reference code is used only for evaluator compatibility and signature extraction.
- Qwen3-4B, Gemma3-4B, and Phi4-mini run through Ollama, with temperature 0, seed 42,
  context 8192 and a 768-token output limit. Local aliases identify those models in artifacts.
- Kev: `jaredpalmer/kev-4b`, base `Qwen/Qwen3.5-4B-Base`, bf16 CUDA, shipped calibration
  temperature 2.406050; no fine-tuning or threshold tuning. Prefix caching
  and CUDA graphs were disabled. Python 3.12, PyTorch 2.7.1+cu128, Transformers 5.17.0.
- Kev code revision: `fdfdfd2c7a98225715a9cc8f4d9e1ef29ebbe79c`. Checkpoint revision: `139fdd94f1b6a6ad80cc15e08fcb99cac885a101`.
  Base revision: `1001bb4d826a52d1f399e183466143f4da7b741b`. Jev was not used.
- Rule selection chooses the first visible-test passer; if none pass, it returns Qwen.
  Plain Kev compares all three proposals. Filtered Kev compares visible-test passers,
  falling back to all three if none pass and skipping inference if only one passes.
  The conventional Qwen judge compares all three and returns a candidate number.
  Invalid selector responses fall back to the first eligible candidate and are counted.
- Selector failures across all 168 evaluated tasks (including development):
  Kev 0, filtered Kev
  0, conventional judge
  0.
- Candidate ordering is fixed (Qwen, Gemma, Phi). Order bias was not averaged away.
  Published benchmarks may be present in model pretraining; withholding tests from
  these requests does not establish absence of pretraining contamination.
- P-values are exact paired McNemar tests, unadjusted for multiple comparisons.
  Small samples, this restricted task population, and shared hardware limit generalization.
- The runtime `decision_select` strategy does not execute code. Test filtering and repair
  belong to the external benchmark workflow. No arbitrary tools are exposed to a model.

## Reproduce and inspect

- [Per-task data, proposals, decisions, repairs, tests and frozen protocols](https://github.com/johncheungmk/openfusion/blob/main/examples/results/kev-spark-2026-09-29.json)
- [Replay script](https://github.com/johncheungmk/openfusion/blob/main/examples/kev_replay.py)
- [Configuration and fallback behavior](../strategies/decision-select.md)
- [Kev source and model documentation](https://github.com/jaredpalmer/kev)

```bash
python examples/kev_replay.py examples/results/kev-spark-2026-09-29.json --split fresh
python examples/kev_replay.py examples/results/kev-spark-2026-09-29.json --split fresh --policy kev --base-url http://localhost:8009/v1
```

The default command recomputes accuracy from recorded selections and candidate labels.
The live command sends only the question and candidate text; labels and withheld tests
never enter the selector request. The script does not execute candidate code.

Data SHA-256: `b6fec879c0aeb8ae9e7003366150f4c499ec66fb1974cf76882998b1dca4aeac`.
