# What our benchmarks support

Updated 4 October 2026. This overview includes the published September studies and the
completed local follow-ups. Each result concerns its own tasks, models and workflow.
Scores from different rows must not be pooled or treated as a common leaderboard.

## Recommendation

Start with a strong single-model baseline, then compare fresh same-model sampling and
verification against a mixed-model workflow. Our strongest positive evidence concerns
short Python tasks with supplied executable checks and selective escalation. The later
controls do not establish a consistent advantage from mixing similarly sized models.

A test-guided mixed workflow, a stronger-model cascade, and multi-model synthesis are
three different interventions. A gain for one does not validate the others.

## Evidence across studies

| Study | Observed result | Supported interpretation |
|---|---|---|
| Early mathematics, 100 tasks | Critique/revision 96%; Qwen 95%; self-synthesis 95% | No convincing mathematics advantage. |
| Early coding, 100 tasks | Verified mixture 82%; best component 72%; Qwen repair 78% | The difference against repair was uncertain (p=.125); this combines pilot and subsequent tasks. |
| Fresh coding, 60 tasks | Verified mixture 51/60; Qwen 43/60; repair 45/60 | Positive evidence for this test-guided workflow; Kev after filtering tied rule selection. |
| Fresh HumanEval, 40 tasks | Mixed candidates 37/40; repeated Qwen 36/40 | Diversity advantage not established (p=1). The earlier 60-task ablation reused examined tasks. |
| Registered coding, 500 tasks, two seeds | Fixed mixture 78.1%; repair 72.5%; single 69.0% | Primary gain +5.6 points, 95% CI [3.6, 7.7]; lower measured latency and tokens than repair in that serving setup. |
| Fresh-Qwen control, 1,500 tasks, two seeds | Mixed 60.43%; fresh Qwen 61.00%; single 56.37% | No demonstrated diversity advantage; primary conditional comparison also inconclusive. |
| No execution feedback, 149 tasks, two seeds | Mixed synthesis 58.05%; Qwen-only 59.40% | No demonstrated diversity advantage. |
| 32-policy development search, 120 held-out tasks, two seeds | Selected mixture 60.00%; repeated-Qwen proposals 65.00%; stronger single 61.67% | Selected mixture failed the registered improvement rule. Both proposal arms used the same stronger synthesizer. |
| Similar-size 4B study, 70 tasks, two seeds | Qwen synthesis: mixed pair 40.71%; repeated Qwen 40.71%; direct Qwen 33.57% | No measured gain from diversity; serving and retry deviations limit interpretation. |
| 24 GB proxy study, 160 tasks, two seeds | Selected small pair 28.44%; repeated Gemma 33.13%; 27B direct 63.13%; repeated 27B 71.25% | Selected mixture underperformed its same-model control. Strict VRAM/no-swap compliance was not established. |

Sources: [early coding, mathematics and Kev](kev-selection-spark.md),
[HumanEval ablation](moa-method-model-ablation.md),
[registered 500-task study](prospective-moa-deployment.md), and
[later follow-ups with downloadable analysis](followup-benchmarks.md).
The same early cohorts appear in several reports; those are not independent replications.

## Where to use these workflows

### Test-guided short programming tasks

This is the strongest supported setting: a known function interface, meaningful supplied
checks, bounded alternative attempts and early stopping. The registered mixed workflow
beat repeated repair on its workload. It did not establish that model diversity caused
the gain. The later fresh-Qwen control did not show superiority over independent attempts
from the same model. A passing visible example is not proof of correctness on unseen cases.

### Selective stronger-model escalation

The [70-task escalation study](case-study-spark-escalation.md) achieved 91.4% accuracy
with stronger-model calls on 40% of tasks, versus 75.7% for the small baseline and 98.6%
for the stronger endpoint always. This is a promising accuracy/request-count trade-off.
It does not establish equal accuracy, dollar savings, or superiority over a simpler router.
Count the small-panel calls too. The gain includes access to the stronger model.

### General synthesis and specialist teams

Research writing, brainstorming, security review and specialist routing remain unvalidated
use cases in this benchmark series. Runtime support for a strategy is not evidence of
quality improvement. Evaluate the actual workload before recommending a mixed panel.

## Practical selection rule

1. Select the strongest suitable single model using development data.
2. Measure fresh repeated sampling with the same verification and stopping rules.
3. Add another model only when a held-out comparison shows a useful improvement.
4. Include loading, synthesis, failures and all model calls in the resource accounting.
5. Keep a hard call cap. Report uncertainty and any protocol deviations.

A hindsight pool containing a correct answer only shows potential. A deployable selector
must recognize that answer without hidden labels. Our evidence supports measuring that
complete workflow rather than assuming that different model names create useful diversity.
