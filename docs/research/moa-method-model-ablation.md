# Does MoA improvement come from the method or the models?

## Controlled study on 2026-09-29

The earlier study found a gain for verified MoA over one Qwen answer and repeated Qwen
repair. It did not isolate diverse models from independent samples of one model.
This follow-up compares those alternatives with the same **three-proposal budget** and
the same visible-test selector, then changes the selector and starting model separately.

## Fresh HumanEval result

The primary comparison is **mixed models with Qwen first: 37/40**, versus
**three Qwen answers: 36/40** (1 wins, 0 losses; exact McNemar p=1).
With Phi as the starting model, the corresponding scores are **34/40 versus 30/40**
(5 wins, 1 losses; exact McNemar p=0.21875).

The deterministic mixed panel scored **36/40**, compared with Qwen's **36/40** and
Phi's **27/40**. The existing SGLang endpoint scored **30/40** in one call per task.
These comparisons answer different questions: the latter is not compute-matched to
three small models.

The larger endpoint spent some of its 768 completion tokens on reasoning. It hit that
limit on 10/40 tasks. A separately declared policy retries **only** those requests
with a 4096-token allowance and scores **35/40**. This retry policy was added
after the larger-model run began, in response to token accounting. Its trigger never
consults test correctness. Its larger token budget is a different experimental condition.
Three requests still hit the 4096-token limit; two completed answers fail the evaluator.
The 35/40 result therefore describes this bounded serving policy, and does not establish
that MoA beats a fully configured larger model.

| Policy | Correct | Proposal calls | Mean output tokens |
|---|---:|---:|---:|
| Qwen alone | 36/40 | 1 | 74 |
| Gemma alone | 29/40 | 1 | 207 |
| Phi alone | 27/40 | 1 | 59 |
| Three Qwen answers + one check | 36/40 | 3 | 219 |
| Qwen first, mixed alternatives + one check | 37/40 | 3 | 334 |
| Three Phi answers + one check | 30/40 | 3 | 188 |
| Phi first, mixed alternatives + one check | 34/40 | 3 | 332 |
| Three different models, temperature 0 + one check | 36/40 | 3 | 340 |
| Qwen-first mixed panel + exact-code vote | 36/40 | 3 | 334 |
| Qwen-first mixed panel + two checks | 38/40 | 3 | 334 |
| Existing SGLang endpoint (`qwen3.8-flash-next`) | 30/40 | 1 | 528 |
| SGLang + retry on token-limit stop | 35/40 | 1.25 | 1095 |

### Interpretation

On this fresh sample, changing two Qwen samples into samples from other models adds only
one success with the same one-test selector. The primary comparison does **not** establish
a statistically reliable advantage. The deterministic mixed panel ties Qwen alone,
unlike the improvement seen on the earlier MBPP sample. The benefit depends on the tasks,
candidate models, sampling and selection together.

The Qwen-only pool contains a correct answer on 36/40 tasks; the sampled mixed pool
contains one on 38/40. The one-test selector recovers 37 of those. There are therefore
two distinct constraints: obtaining a useful alternative and recognizing it.

### Two concrete examples

- **HumanEval-142:** all three Qwen answers misinterpret the index-based arithmetic.
  One sampled Phi answer passes every test, and the mixed selector chooses it. Phi's
  deterministic answer and its other sampled answer also fail. This is a gain from a
  particular complementary sample, not proof that Phi is consistently better at the task.
- **HumanEval-65:** Qwen's digit-rotation answer passes the first visible test but mishandles
  a special case. A sampled Gemma answer passes every test. Revealing the second test to
  the selector lets it choose that answer. Accuracy on the tests that remain hidden in
  both conditions does not increase, so this is not independent evidence of better
  generalization from the extra check.

### What changes in each comparison?

1. **Model diversity:** keep the first answer, prompt, three-call allowance and selector
   fixed; replace two samples from that model with samples from the other model families.
2. **Starting model:** repeat the comparison with Phi first. This is a separate controlled
   comparison, not a post-hoc choice of the better starting model for each task.
3. **Selection method:** use the identical mixed candidates with visible-test selection or
   an exact-code vote. Voting groups matching Python ASTs and breaks ties by candidate
   order; it is not semantic equivalence voting or an LLM judge.
4. **More checking information:** let the selector see a second published assertion.
   The generators still see only the first assertion. All later assertions remain hidden.
5. **Stronger single endpoint:** send the same prompt to the already deployed SGLang model.
   Different model size and serving costs prevent a compute-equivalence claim.

Checker versus code vote: 1 wins, 0 losses; exact McNemar p=1.
Two checks versus one: 1 wins, 0 losses; exact McNemar p=1.
On the tests remaining hidden under **both** conditions, one-check selection passed
38/40;
two-check selection passed
38/40.
The main table requires passing *all* tests, including visible ones.

## Was a correct answer available?

This table uses the evaluation answers only after inference to ask whether **any**
proposal passed every test. It is a hindsight ceiling, not a selectable production policy.
If the pool contains no correct proposal, changing selectors cannot recover one.

| Candidate pool | Fresh HumanEval | Earlier MBPP |
|---|---:|---:|
| `qwen_three` | 36/40 | 45/60 |
| `mixed_qwen_first` | 38/40 | 51/60 |
| `phi_three` | 32/40 | 44/60 |
| `mixed_phi_first` | 37/40 | 51/60 |
| `mixed_deterministic` | 36/40 | 51/60 |

## Previously examined MBPP tasks

These are the 60 MBPP tasks used for the previous fresh study. New sampled answers were
generated, but the tasks and deterministic answers had already been examined. Treat this
as a retrospective comparison; the fresh HumanEval sample above is the primary test.

| Policy | Correct | Proposal calls | Mean output tokens |
|---|---:|---:|---:|
| Qwen alone | 43/60 | 1 | 63 |
| Gemma alone | 36/60 | 1 | 95 |
| Phi alone | 40/60 | 1 | 46 |
| Three Qwen answers + one check | 45/60 | 3 | 190 |
| Qwen first, mixed alternatives + one check | 51/60 | 3 | 201 |
| Three Phi answers + one check | 44/60 | 3 | 139 |
| Phi first, mixed alternatives + one check | 50/60 | 3 | 207 |
| Three different models, temperature 0 + one check | 51/60 | 3 | 204 |
| Qwen-first mixed panel + exact-code vote | 43/60 | 3 | 201 |
| Qwen-first mixed panel + two checks | 51/60 | 3 | 201 |

- Mixed versus repeated Qwen: 6 wins, 0 losses; exact McNemar p=0.03125.
- Mixed versus repeated Phi: 8 wins, 2 losses; exact McNemar p=0.109375.

## Does the first model matter?

Using exactly the same three deterministic proposals and one visible test:

| Model order (one visible check) | Fresh HumanEval | Earlier MBPP |
|---|---:|---:|
| qwen → gemma → phi | 36/40 | 51/60 |
| qwen → phi → gemma | 36/40 | 51/60 |
| gemma → qwen → phi | 33/40 | 50/60 |
| gemma → phi → qwen | 32/40 | 50/60 |
| phi → qwen → gemma | 34/40 | 50/60 |
| phi → gemma → qwen | 32/40 | 50/60 |

Ordering matters when more than one answer passes an incomplete check but some fail
unseen cases. These scores describe all six predeclared orders. They are not a fitted
router or evidence that choosing the best order after seeing test outcomes is deployable.

## Early stopping: useful follow-up for efficiency

The same policies can stop at the first passing proposal, requesting alternatives only
when needed. Reconstructing that sequence from these recorded requests gives:

| Early-stop policy | Fresh score / mean calls | Earlier score / mean calls |
|---|---:|---:|
| `qwen_three_adaptive_check1` | 36/40 / 1.150 | 45/60 / 1.417 |
| `mixed_qwen_first_adaptive_check1` | 37/40 / 1.125 | 51/60 / 1.350 |
| `phi_three_adaptive_check1` | 30/40 / 1.350 | 44/60 / 1.533 |
| `mixed_phi_first_adaptive_check1` | 34/40 / 1.275 | 50/60 / 1.400 |

These call counts are **reconstructed**, not separately executed end-to-end measurements.
They motivate a live latency and resource test; they do not establish a dollar-cost saving.
The main accuracy comparisons above retain the full three-call allowance for every task.

## Protocol, costs and limitations

- Fresh data: 40 tasks sampled with seed `1020260929` from 121 eligible tasks in
  [OpenAI HumanEval](https://github.com/openai/human-eval). None had been used in this project.
  Eligibility requires at least three assertions, a `check` function containing only
  assertions, and compatibility with the restricted Python evaluator. Every assertion
  in each eligible task is retained. Typing imports and function annotations are removed.
- One published assertion is appended to each prompt. All remaining assertions are withheld
  except the deliberately revealed second check in the two-check selector condition.
  This is a modified subset experiment, **not the standard HumanEval leaderboard**.
- Reference implementations are used only to establish evaluator compatibility. No reference
  implementation or withheld expected answer is sent to a model.
- Small models: Qwen3-4B, Gemma3-4B and Phi4-mini through Ollama. The anchor uses
  temperature 0, seed 42. Alternatives use temperature 0.7, seeds 31415 and 27182.
  In the mixed Qwen-first pool, the alternatives are Phi and Gemma at seed 31415.
  In the mixed Phi-first pool, the alternatives are Qwen and Gemma at seed 31415.
- Every model receives the same coding instruction, context limit 8192 and output
  limit 768. All three candidate calls are counted, even if the first one passes.
  No repairs, aggregation call, or learned selector is included in these three-call arms.
- Candidate tests run in the same restricted Python evaluator with a three-second process
  timeout. Unsupported constructs and timeouts count as failures. The restriction is part
  of this task definition, not a claim that all otherwise valid Python programs are wrong.
- Selection returns the first answer passing available checks; if none passes, it returns
  the first answer. Candidate ordering is fixed before evaluation.
- The larger-model baseline was planned before its inference and before inspecting the
  fresh accuracy summary. It uses temperature 0, a 768-token output cap, and no retries.
- The separately reported length-triggered policy permits at most one additional request,
  with a 4096-token cap, only after `finish_reason=length`. SGLang completion-token counts
  include its reported reasoning tokens. No hidden reasoning content is stored or published.
- Reported output tokens and call counts describe logical policies reconstructed from
  shared generations. They are not measured dollar costs or isolated end-to-end latency.
  Requests to separate Sparks can overlap, and the production SGLang model stays resident.
- The additional-check experiment assumes a trustworthy extra test is available. The cost
  and correctness of automatically generating that test have **not** been evaluated.
- One seed pair and one small task sample cannot establish a universal model-family effect.
  Published tasks may have appeared in model training. P-values are exact paired McNemar;
  secondary comparisons are exploratory and unadjusted for multiplicity.

## What is worth testing next?

- **Repeat across seeds and a larger untouched task set.** The primary fresh gain is only
  one task. Confirm whether alternative models reliably add coverage before relying on it.
- **Generate and validate extra tests.** A second trustworthy check can resolve an ambiguous
  selection, but these experiments used published tests. Test generation needs its own
  accuracy and cost control, including the possibility of rejecting correct code.
- **Early stopping and routing.** Start with a model selected on development data and invoke
  alternatives only when checks fail. Measure live latency and GPU memory at a fixed
  quality target rather than inferring cost from call counts.
- **Tasks without executable tests.** Grounded question answering or extraction would test
  whether semantic selectors such as Kev help when a code checker is unavailable. Those
  workloads have not been established by this coding study.

## Reproduction artifacts

- [Frozen protocol, all proposals, test outcomes and policy selections](https://github.com/johncheungmk/openfusion/blob/main/examples/results/moa-ablation-2026-09-29.json)
- [Offline analysis script](https://github.com/johncheungmk/openfusion/blob/main/examples/moa_ablation_analysis.py)
- [Previous MoA and Kev experiment](kev-selection-spark.md)

```bash
python examples/moa_ablation_analysis.py examples/results/moa-ablation-2026-09-29.json
python examples/moa_ablation_analysis.py examples/results/moa-ablation-2026-09-29.json --split retrospective
```

The script verifies selection from recorded test outcomes without executing generated code.
Data SHA-256: `f5b158a7a17f673d52f04c689b8c6aa3148f1657a917d07b7d205b949fdb630c`.
