# Can a fixed MoA workflow earn its deployment cost?

## In plain language

**Yes: this preselected workflow improved results on the registered workload.** Ask Qwen
first and check its program against the supplied example. If that check fails, ask Phi,
then Gemma, stopping at the first passing answer. If all three fail, let Qwen repair its
original answer once. The same rule was used for every task, with no access to hidden answers.

Across 500 previously unused project tasks and two runs per task, this approach passed
78.1% of all-test evaluations. Giving Qwen repeated repair opportunities passed 72.5%.
That is about **six extra successes per 100 attempts**, with **17.2% less average waiting
time and 29.2% fewer total tokens** than Qwen repair in this setup. A single Qwen attempt
passed 69.0% and was faster than either multi-call workflow.

**Recommendation: supported for similar short Python tasks with a known function interface
and a supplied executable check, using these model versions and this serving setup.** A user
can choose this rule before seeing a future answer. This evidence does not establish the
same improvement for another model combination, tasks without tests, general chat, or all
coding workloads. The model versions, verification rule, and task distribution matter.

The 40-task stronger-model check is encouraging but too small to establish superiority over
that endpoint. The confirmed comparison is against the preselected Qwen repair workflow.

## Results

**Registered conclusion: supported under the tested conditions.**

The fixed MoA workflow changed accuracy by **5.6 percentage
points** relative to Qwen repair. The 95% task-cluster bootstrap interval was
**[3.6, 7.7] points**; the whole-task sign-flip p-value was
**0.00001**. This is the single registered primary comparison.

The mean latency ratio was **0.828×** and the mean total-token
ratio was **0.708×**, both relative to Qwen repair.

| Workflow | All-tests-pass runs | Accuracy | Mean calls | Mean input + output tokens | Mean seconds | Median seconds | 95th percentile seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single Qwen | 690/1000 | 69.0% | 1.000 | 250.8 | 2.212 | 1.250 | 4.849 |
| Qwen repair | 725/1000 | 72.5% | 1.616 | 621.9 | 4.243 | 1.434 | 17.423 |
| Fixed MoA | 781/1000 | 78.1% | 1.538 | 440.6 | 3.511 | 1.478 | 12.346 |

The 1,000 runs per small-model arm cover 500 tasks with two seeds each. They are not 1,000 independent tasks.

### Did it meet the registered deployment criteria?

| Criterion | Outcome |
|---|---|
| Positive 95% lower bound | Met |
| Sign-flip p < .05 | Met |
| Observed gain at least 5 points | Met |
| Mean latency ratio at most 1.5 | Met |
| Mean token ratio at most 1.5 | Met |

The fixed workflow meets the registered rule for this workload and comparator. This supports choosing it beforehand for similar verified Python tasks, subject to the stated model and serving conditions. It does not establish superiority over every single-model strategy.

The latency-ratio 95% interval was [0.762, 0.905];
the token-ratio interval was [0.654, 0.770].
These intervals quantify task variation on this run, not variation across independent
deployment days or concurrent production workloads.

### Stronger endpoint: matched 40-task subset

This is a secondary descriptive comparison on the preselected subset, using only seed
17429 for every arm. It is not the 500-task primary comparison or an equal-compute claim.

| Workflow on the same 40 tasks | Correct | Mean seconds | Mean input + output tokens |
|---|---:|---:|---:|
| Single Qwen | 32/40 | 1.464 | 196.3 |
| Qwen repair | 32/40 | 2.524 | 394.1 |
| Fixed MoA | 35/40 | 2.236 | 293.4 |
| Existing stronger endpoint | 33/40 | 2.481 | 244.6 |

### Descriptive checks across seeds and task sources

These checks were not used to choose a subgroup or change the deployment recommendation.

| Generation seed | Single Qwen | Qwen repair | Fixed MoA |
|---|---:|---:|---:|
| 17429, 500 tasks | 345/500 | 366/500 | 389/500 |
| 58103, 500 tasks | 345/500 | 359/500 | 392/500 |

The paired task-average difference favored MoA on 44 tasks and repair on eight; the other
448 tasks tied. Descriptive summaries for sanitized wording and the original dataset splits
are included in `supplementary.json`. No new subgroup deployment rule follows from them.

The mixed workflow stopped at its initial Qwen answer in 783 of 1,000 runs, at Phi in 30,
and at Gemma in 53; 134 reached the final Qwen repair. Thus it requested other models only
when the initial visible check failed. Fresh attempts avoid carrying a previous failed
answer in the prompt, which can reduce token use compared with repeated repair. This is
an explanation consistent with the recorded execution, not a separate causal ablation of
model diversity, prompt length, or repair strategy.

### Completion and verification

All 3,040 generation records and 3,040 grading records passed the structural audit. The audit verifies model order, call budgets, early stopping, seeds, cohort membership, registration, and generation-to-grading checksums.

Generation completed at `2026-09-29T10:00:18Z`. Hidden grading began afterward and completed at `2026-09-29T10:09:43Z`.

| Arm | Requests stopped at token limit |
|---|---:|
| moa | 14 |
| qwen_repair | 22 |
| single | 13 |
| stronger | 0 |

No confirmation outcome selected the method, model order, stopping rule, or resource thresholds.

No inference restart or accuracy-dependent retry was required. Token-limit responses were
retained and scored under the frozen rule. All model endpoints were healthy after completion,
and all three small-model digests still matched registration. No evaluator container remained
after grading.

### Serving work and resource interpretation

| Small-model workflow | Mean reported prefill + generation seconds | Mean model-load seconds |
|---|---:|---:|
| single | 1.866 | 0.153 |
| qwen_repair | 3.669 | 0.248 |
| moa | 3.011 | 0.177 |

These are provider-reported durations, not measured GPU cycles or joules. Token counts
also use different model tokenizers. Qwen runs on one GB10 host; the mixed workflow additionally
uses Phi and Gemma on the other host. Its additional resident model footprint is therefore
part of the deployment choice even when average request latency is acceptable. No equal
hardware ownership cost or energy-efficiency claim follows from the registered token gate.

## What was fixed beforehand?

The [registration commit](https://github.com/johncheungmk/openfusion/commit/ca2c9df64d656616c8bd6bade979e21694248d99)
contains the executable controller, analysis, model manifests, task identities, and
protocol. Confirmation records identify that commit and the protocol checksum.

A [publication correction](https://github.com/johncheungmk/openfusion/commit/2a6432d93f9af30de48684ab40b148c587456b15)
restores the original CRLF bytes of the JSON artifacts: the first upload normalized line
endings. The JSON data and files used by the running experiment are identical, and the
declared checksums are unchanged. The corrected downloads were verified against all seven
declared data/code hashes before hidden grading.

The primary comparison is a fixed mixed-model workflow against a Qwen repair workflow.
Both use a maximum of four model calls, the same supplied visible acceptance test, and
the same 2,048-token output allowance per small-model call. Equal ceilings do not imply
equal actual compute costs; measured latency and tokens are part of the decision.

| Workflow | Fixed behavior |
|---|---|
| Single Qwen | One answer |
| Qwen repair | Qwen answer, then up to three repairs using visible-test feedback |
| MoA | Qwen answer, then independent Phi and Gemma attempts, then one Qwen repair of its initial answer |
| Existing stronger endpoint | One answer on a preselected 40-task subset, with reasoning disabled and an 8,192-token ceiling |

Each workflow stops as soon as the supplied test passes. A failed final repair remains
the fallback. No hidden test, correct reference answer, or learned judge selects an answer.

## Workload and exclusions

The cohort contains 500 tasks from [Google Research's public MBPP dataset](https://github.com/google-research/google-research/tree/master/mbpp), not previously used for generation in the known
project records. Two seeds are run for each small-model workflow. Statistical resampling
keeps both seeds of a task together; the analysis uses 500 task units, not 1,000 independent trials.

The audit excluded 168 previous MBPP task identities, exact matches to prior prompts and
reference syntax trees, and repeated candidate prompts. Eligibility requires at least
three regular tests and a reference
program passing the sandbox checks. Of the 801 eligible candidates, a fixed shuffle
selected 500 and left 301 unused. The selected cohort contains 155 tasks with hand-verified
wording and tests; the others use the original MBPP records.

The dataset combines original training, validation, prompting, and testing splits. It is
therefore a custom project-held-out experiment, not the official MBPP test score. Public
tasks may have appeared in model training. Exact duplicate filtering cannot rule out
semantic duplicates, and passing benchmark tests is not proof of complete correctness.

The deployment setting assumes an executable acceptance example and known function
interfaces. The cost of obtaining those inputs is not measured. No claim is made for
tasks without usable verification or for automatically generated tests.

Here MoA means generating candidates with different model families, selecting through the
supplied check, and using the declared final repair when every candidate fails that check.

## Registered decision rule

The primary outcome is the paired difference in passing all regular tests, averaged over
the two seeds within each task. A deployment recommendation requires all of:

- A positive lower limit of the 95% task-cluster bootstrap interval.
- A two-sided whole-task sign-flip p-value below 0.05.
- An observed accuracy improvement of at least five percentage points.
- Mean end-to-end latency no more than 1.5 times Qwen repair.
- Mean total token volume no more than 1.5 times Qwen repair.

These are explicit provisional engineering thresholds. They were not inferred from the
confirmation results or supplied as business requirements. The five-point point-estimate
threshold does not prove the true population gain exceeds five points.

The sample-size calculation assumes paired discordance of 0.15 and a true five-point gain.
Under a normal approximation, 500 tasks provide about 83% power even if both seeds are
perfectly correlated. This refers to detecting a positive accuracy difference, not to the
probability of meeting every accuracy and cost criterion together. Higher discordance
reduces power. Secondary model and subgroup
comparisons are exploratory and cannot establish a new deployment rule.

## Execution and measurement

The policies run independently in a fixed randomized order across two Dell Pro Max with
NVIDIA GB10 systems (product identifier FCM1253). These are the machines described as
"Sparks" in the project discussions; host inspection established their exact OEM identity.
The Qwen node uses NVIDIA driver 580.95.05 and Linux 6.11.0-1016-nvidia; the Phi/Gemma
node uses driver 580.142 and Linux 6.17.0-1014-nvidia. These software differences are part
of the tested deployment, not a controlled comparison of identical serving stacks.
Latency includes generation, loading, transport, and visible verification; hidden grading
is excluded. The existing SGLang service stays resident. Measurements describe this shared
laboratory setup and do not establish energy efficiency, dollar cost, or a service guarantee.
Arms and seeds repeat task prompts. Serving caches are not reset between jobs, so later
requests may benefit from earlier ones; randomized order reduces systematic ordering bias.
The study does not separately estimate latency with empty caches for every request.

Every code check uses a fresh Python 3.12 container with no network, host mount, GPU access,
or elevated container privileges. Its root is read-only, with a small temporary filesystem,
256 MiB memory, one CPU, and an eight-second wall limit. Standard-library Python is allowed,
including imports, classes, loops, and dictionary methods rejected by the earlier evaluator.

Development checks used eight previously exposed tasks. Reasoning enabled on the stronger
endpoint produced one 8,192-token truncation. Disabling reasoning eliminated truncation on
those development tasks before registration. This configuration is fixed for confirmation.

## Reproduction

The [completed result artifacts](https://github.com/johncheungmk/openfusion/tree/main/examples/results/prospective-2026-09-29)
include [raw responses and grades](https://github.com/johncheungmk/openfusion/raw/refs/heads/main/examples/results/prospective-2026-09-29/results.json.gz),
[analysis and task-level outcomes](https://github.com/johncheungmk/openfusion/blob/main/examples/results/prospective-2026-09-29/analysis.json),
the structural audit, descriptive summaries, development checks, environment observations,
and SHA-256 checksums. Private credentials and service addresses are omitted.

The [frozen artifacts](https://github.com/johncheungmk/openfusion/tree/2a6432d93f9af30de48684ab40b148c587456b15/examples/results/prospective-2026-09-29)
include separate public inputs and grading tests. Keep private endpoint configuration
outside the repository. The runner checks its own checksum and the public task checksum
against the registered protocol before confirmation inference.

Use the recorded model digests and templates in `models.json`. The controller uses only
the Python standard library and a Docker daemon that can run the pinned ARM64 Python image.
Download the frozen files without line-ending conversion; the runner intentionally rejects
files whose byte hashes differ. An endpoint configuration has this form, with the placeholder
hostnames replaced privately:

```json
{
  "qwen": {"url": "http://qwen-host:11434", "model": "of-qwen3-4b"},
  "phi": {"url": "http://phi-host:11434", "model": "of-phi4-mini"},
  "gemma": {"url": "http://gemma-host:11434", "model": "of-gemma3-4b"},
  "stronger": {
    "url": "http://stronger-host:30000/v1",
    "api": "openai",
    "model": "qwen3.8-flash-next",
    "chat_template_kwargs": {"enable_thinking": false}
  }
}
```

```bash
python examples/prospective_moa.py run \
  --tasks examples/results/prospective-2026-09-29/tasks-public.json \
  --protocol examples/results/prospective-2026-09-29/protocol.json \
  --registration-commit ca2c9df64d656616c8bd6bade979e21694248d99 \
  --endpoints /path/to/private-endpoints.json \
  --output /path/to/records --stronger
```

Run hidden grading only after all 3,040 generation records exist. The analysis and
independent structural audit operate on exported records without making model calls.

```bash
python examples/prospective_moa.py grade \
  --tasks examples/results/prospective-2026-09-29/tasks-grading.json \
  --records /path/to/records --output /path/to/grades
```

To reproduce the published numerical analysis without model access, decompress the
published `results.json.gz` to `results.json`, then run:

```bash
python examples/prospective_analysis.py results.json --output analysis.json
python examples/prospective_audit.py results.json --output audit.json
```

The compressed bundle contains public task inputs, grading tests, every model response,
selection and timing records, per-test grading, and completion timestamps. The analysis
includes paired task-level outcomes. Reading these released answers makes these tasks
development data for any subsequent policy revision; a new confirmation must use an
untouched cohort.

## Software and operational changes

The local project Python environment gained Paramiko 5.0.0 and its dependencies for SSH
transport. The evaluation host gained the pinned `python:3.12-slim` Docker image used by
the sandbox. No host packages or model weights were removed, and no existing serving service
was restarted or reconfigured. Existing model weights were reused. Small-model residency
uses the recorded 60-minute keep-alive expiry; the existing distributed SGLang service remains
running. All 165 offline software tests passed, along with lint, compilation, package build,
and strict documentation validation.

Related studies: [MoA method and model comparison](moa-method-model-ablation.md) and
[Kev selection on Spark](kev-selection-spark.md).
