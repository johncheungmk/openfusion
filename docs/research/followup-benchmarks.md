# Follow-up benchmarks: diversity, sampling and memory

Completed September-October 2026; published 4 October 2026. These are project-held-out
coding experiments, not official dataset leaderboard scores. Read the
[evidence overview](benchmark-evidence.md) for recommendations across all studies.

## Fresh-Qwen control

On 1,500 tasks with two seeds, mixed-model selection scored 60.43%, fresh-Qwen selection
61.00%, and single Qwen 56.37%. The unconditional mixed-minus-fresh difference was
-0.57 percentage points (95% CI [-1.47, 0.33], p=.2433).
The primary conditional analysis retained 753 task-seed pairs across 383 tasks:
16.58% versus 19.32%, difference -2.74 points (CI [-6.40, 0.78], p=.1588).
This does not prove equivalence or that fresh sampling explains every earlier gain.
Recovery left aborted-request resource usage unknown, so latency and token claims are
suppressed. Two terminal generation failures were retained.
[Analysis](data/2026-10-followups/fresh-qwen.json).

## Synthesis without execution feedback

The 149-task, two-seed confirmation scored 58.05% mixed versus 59.40% Qwen-only,
difference -1.34 points (95% CI [-6.71, 4.03], p=.7118). Direct Qwen scored 56.38%.
A synthesis policy was selected on a separate 58-task development set. No registered
improvement criterion passed. Two requests had unknown usage; totals shared across
strategies are not independent per-arm costs.
[Analysis](data/2026-10-followups/no-verifier.json).

## Wider policy search

After searching 32 configurations on 24 development tasks, the frozen winner was
independent proposals with stronger-model synthesis. On 120 held-out tasks with two seeds,
it scored 60% against 65% for Qwen-only proposals with the same synthesizer.
The difference was -5 points (95% CI [-9.58, -0.83], p=.0426).
Direct small Qwen scored 45.42%; direct stronger model scored 61.67%.
The mixture improved on the small direct baseline but failed the full registered rule.
The Qwen-only label identifies the proposal pool, not the synthesizer. Twenty-four
requests had unknown usage. [Analysis](data/2026-10-followups/wide-search.json).

## Similar-size 4B comparison

Seventy tasks, two seeds; Qwen and Gemma proposal pairs with each model as synthesizer.
No execution feedback was supplied.

| Synthesizer | Two Qwen proposals | Qwen + Gemma proposals | Two Gemma proposals |
|---|---:|---:|---:|
| Qwen | 40.71% | 40.71% | 34.29% |
| Gemma | 39.29% | 24.29% | 17.14% |

Direct Qwen scored 33.57%; direct Gemma 23.57%. Mixed versus repeated Qwen had zero
difference with Qwen synthesis (CI [-4.29, 4.29] points, Holm p=1), and -15 points with
Gemma synthesis (CI [-22.86, -7.14], Holm p=.00252). Neither synthesizer supported the
registered diversity claim. These results do not isolate model quality from diversity.

The run saved 1,680 answers from 1,684 attempts. There were 22 Gemma failures with unknown
usage, runtime-version mismatch across workers, and four extra attempts affecting two
task-seed pairs. Preserve those deviations when interpreting the scores. Worst-case
retry sensitivity bounds for mixed-minus-repeated Qwen were [-1.43, 1.43] points with
Qwen synthesis and [-15.71, -12.86] with Gemma synthesis; these are not confidence intervals.
[Analysis](data/2026-10-followups/diversity4b.json) and
[retry sensitivity](data/2026-10-followups/diversity4b-retry-sensitivity.json).

## 24 GB unified-memory proxy study

Both isolated Ollama servers used pinned version 0.34.4. Separate development selection
froze the policies before 160 held-out tasks were run with two seeds: 1,920 workflows,
5,760 calls. Each workflow ran sequentially on one Spark under a 600-second ceiling,
including loading and synthesis. Call caps were one for direct, three for two-proposal
synthesis and four for three-proposal synthesis. No inference retries were made.

| Workflow | Correct / 320 | Accuracy | Median wall seconds | Final generation failures |
|---|---:|---:|---:|---:|
| 27B Qwen direct | 202 | 63.125% | 69.3 | 0 |
| 27B Qwen two proposals + synthesis | 228 | 71.250% | 196.1 | 0 |
| Granite 8B + Gemma 12B, Gemma synthesis | 91 | 28.438% | 32.3 | 3 |
| Gemma 12B two proposals + synthesis | 106 | 33.125% | 37.8 | 12 |
| Qwen 8B + Granite 8B + Gemma 12B, Gemma synthesis | 105 | 32.813% | 46.7 | 2 |
| Gemma 12B three proposals + synthesis | 107 | 33.438% | 48.1 | 11 |

Registered contrasts below are selected heterogeneous pair minus control. Intervals are
95% task-cluster bootstrap intervals retaining both seeds; p-values use 100,000 task-level
sign flips with Holm correction across the three contrasts. Intervals are marginal.

| Control | Difference, percentage points | 95% CI | Holm p |
|---|---:|---|---:|
| 27B direct | -34.69 | [-42.19, -27.19] | .000030 |
| Repeated 27B | -42.81 | [-50.94, -35.00] | .000030 |
| Repeated Gemma | -4.69 | [-8.75, -0.94] | .035380 |

Every contrast favored the control at both seeds. Other comparisons, including repeated
27B against direct 27B, are descriptive. The 28 terminal workflow failures stayed in the
denominator. The 339 failed calls include failed proposals followed by scheduled synthesis.
There was no retry-until-success or replacement by the best intermediate answer.

### Memory and interpretation limits

- Sparks use unified memory. These measurements do not establish discrete 24 GB GPU latency
  or isolated peak VRAM. The cap was 24,000,000,000 bytes on recorded proxy measures.
- Maximum sampled CUDA device-global increase was 22,057,324,544 bytes; maximum Ollama
  reported resident allocation was 17,601,470,135 bytes. Neither is isolated peak VRAM;
  sampling every 50 ms can miss transients.
- System swap increased in 84 workflows, by up to 10,403,840 bytes. The frozen gate recorded
  swap but did not reject increases. Worker swap delta was zero, but worker counters exclude
  the model runner. Strict no-swap compliance was not established, and system swap cannot
  be attributed to the model from these measurements. This misses the requested no-swap
  requirement; the study must remain labeled a feasibility proxy.
- Equal time ceilings did not mean equal realized compute. Small policies left much of the
  allowance unused. No optimally budget-filling small-model comparison was performed.
- Models were Q4_K_M. The 27.3B artifact's local alias was `qwen3.8:latest`, with recorded
  architecture `qwen35`; this is not a verified vendor release name. It was the strongest
  tested fitting candidate, not a claim about every available model.
- Context was 16,384 and output allowance 2,048 tokens per call, temperature .7. Grading used
  a pinned, network-disabled Python container and at most eight tests per task. Passing
  bounded benchmark tests is not proof of full correctness.

[Analysis](data/2026-10-followups/vram24.json) and
[evidence consistency audit](data/2026-10-followups/vram24-audit.json).
An audit pass verifies records and frozen proxy checks; it does not certify strict VRAM
or no-swap compliance.

## Publication scope and provenance

The downloadable JSON files are aggregate analysis exports, with original-source and
published-byte hashes in the [manifest](data/2026-10-followups/manifest.json).
They preserve the recorded numerical results. These follow-up exports do not include all
raw responses, private deployment files or a complete public reproduction package.
Original local evidence remains preserved. Earlier public reproduction bundles remain
linked from their original study pages. Development outcomes are not fresh confirmation,
and reused early cohorts must not be counted twice.
