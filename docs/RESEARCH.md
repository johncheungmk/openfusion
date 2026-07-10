# Research positioning

Research review cutoff: **2026-07-10**.

OpenFusion is an open-source, OpenAI-compatible runtime for explicit, bounded
multi-model inference workflows. It supports local and hosted OpenAI-compatible providers
and records public operational traces. It is not weight-level model merging, a learned
orchestration foundation model, or evidence that multi-agent inference always improves an
answer.

The project turns research ideas into inspectable engineering baselines:

- same-model sampling and selection;
- exact, weighted, and semantic voting;
- pairwise ranking followed by generative fusion;
- parallel synthesis and layered refinement;
- critique and revision;
- confidence- and consistency-aware cascading;
- heuristic or constrained-plan adaptation;
- local equal-budget evaluation.

Reported results from a paper or hosted product belong to that system's models, prompts,
data, judge, and budget. OpenFusion does not inherit those results by implementing a
similar workflow shape.

## Evidence policy

This documentation distinguishes among:

- **peer-reviewed papers**, identified by venue and year;
- **accepted 2026 papers**, whose final proceedings may still be recent;
- **preprints**, identified with their first-posted date;
- **product documentation and vendor experiments**, which describe a product rather than
  independent academic evidence;
- **planned OpenFusion work**, which is not described as current behavior.

The strongest recurring lesson is conditionality: routing, fusion, judging, and extra
test-time compute work only when their estimator, candidate pool, task, and budget make
them useful.

## Research map

| Direction | Primary source | Status and date | OpenFusion relationship | Main caveat |
|---|---|---|---|---|
| Self-consistency | [Wang et al.](https://openreview.net/forum?id=1PL1NIMMrw) | ICLR 2023 | `best_of_n`, voting, same-model sampling | Established on selected reasoning tasks |
| Rank then fuse | [LLM-Blender](https://aclanthology.org/2023.acl-long.792/) | ACL 2023 | `pairwise_rank_fuse`, synthesis | OpenFusion uses prompted, not trained, ranking |
| Small/large routing | [Hybrid LLM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b47d93c99fa22ac0b377578af0a1f63a-Abstract-Conference.html) | ICLR 2024 | Cascade and adaptive-routing motivation | Two-model results do not imply arbitrary-pool performance |
| Routing benchmark | [RouterBench](https://arxiv.org/abs/2403.12031) | Preprint, 2024-03-18 | Offline router evaluation reference | Historical models, costs, and outcomes |
| Layered agents | [Mixture-of-Agents](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5434be94e82c54327bb9dcaf7fca52b6-Abstract-Conference.html) | ICLR 2025 Spotlight | `layered_refinement` | Strong claims include LLM-judged conversational tasks |
| Learned preference router | [RouteLLM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html) | ICLR 2025 | Future optional learned router | Requires representative preference supervision and drift checks |
| Same-model MoA | [Self-MoA](https://arxiv.org/abs/2502.00674) | Preprint, 2025-02-02 | `self_moa`, `self_moa_seq` | Main-conference peer review not established |
| Unified route/cascade | [Dekoninck et al.](https://icml.cc/virtual/2025/poster/46183) | ICML 2025 | Future quality-aware dynamic cascade | Quality estimation is the difficult component |
| Adaptive test-time compute | [Snell et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html) | ICLR 2025 oral | Adaptive call/sample budgets | Results focus on mathematics and verifier access |
| Large routing re-evaluation | [LLMRouterBench](https://arxiv.org/abs/2601.07206) | Preprint, 2026-01-12 | Simple-baseline and pool-curation guidance | Recent, not peer reviewed |
| Model plus length routing | [R2-Router](https://openreview.net/forum?id=S3m1tSp8F4) | Accepted ICML 2026, posted 2026-04-30 | Future output-budget action | Length instructions need backend enforcement |
| Unified model/compute action | [UniScale](https://arxiv.org/abs/2605.30898) | Accepted ICML 2026, posted 2026-05-29 | Future `(model, strategy, budget)` router | Online feedback and exploration are required |
| Shared reasoning budget | [CLEAR](https://arxiv.org/abs/2606.03092) | Accepted ICML 2026, posted 2026-06-02 | Multi-dimensional budget motivation | Global traffic objective differs from one request |
| System-level agent evaluation | [MASEval](https://aclanthology.org/2026.acl-demo.34/) | ACL 2026 System Demonstration | Evaluate orchestration, harness, and error handling as one system | Framework effects do not transfer automatically to OpenFusion |

See [Related work](research/related-work.md) for findings, dates, and limitations in
greater detail.

## What current research implies for OpenFusion

### Measure candidate quality and complementarity

Self-MoA results warn that adding a lower-quality provider for nominal diversity can hurt.
Router re-evaluations likewise find diminishing returns from larger pools. Panel selection
should therefore use direct-provider outcomes and complementary error patterns rather than
provider count alone.

OpenFusion should continue to compare:

- every provider directly;
- best single provider;
- same-model sampling;
- mixed-model fusion;
- a simple fallback/router baseline;
- equal-budget alternatives.

### Treat routing as estimation under drift

The route/cascade literature consistently depends on estimating response quality for a
specific query. Prices, latency, model aliases, and query distributions change. A future
learned router should be optional, versioned, trained only from explicit evaluation data,
and fall back to a readable policy when its calibration data is missing or stale.

### Allocate compute adaptively

Uniform best-of-N is simple but can overspend on easy or hopeless examples. Recent work
supports choosing model, sample count, strategy, and token budget together. Any OpenFusion
implementation must preserve the hard `max_total_calls` invariant and check every budget
before each call.

### Audit the evaluator

LLM judges are useful scalable measurements, not oracles. Primary research documents
position, verbosity, self-preference, multilingual, and adversarial biases:

- [Judging LLM-as-a-Judge](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html), NeurIPS 2023;
- [LLM Evaluators Recognize and Favor Their Own Generations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html), NeurIPS 2024;
- [Systematic Position-Bias Study](https://aclanthology.org/2025.ijcnlp-long.18/), IJCNLP-AACL 2025;
- [Multilingual Judge Reliability](https://aclanthology.org/2025.findings-emnlp.587/), Findings of EMNLP 2025;
- [BiasScope](https://openreview.net/forum?id=QGOw6AU8Lp), ICLR 2026.

Objective graders should take precedence. LLM comparisons should balance answer order,
record judge identity, expose abstention, and be checked against a human-reviewed sample.

### Do not equate verbal confidence with probability

[Can LLMs Express Their Uncertainty?](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6733cf15e10e2cd1d59af033c3bb8507-Abstract-Conference.html)
(ICLR 2024) found frequent overconfidence and task dependence in black-box confidence
elicitation. `uncertainty_cascade` is therefore an experimental selective-prediction
strategy. It should be evaluated using calibration and risk-coverage metrics, not only
final accuracy.

## v0.6 evaluation-reliability implementation

### Order-balanced, abstaining judges

The optional `llm_pairwise_swap` grader requests A/B and B/A orders when two calls remain
in the per-case budget and the first verdict is valid, then remaps them to stable answer
identities. It accepts a win, tie, or loss only when both judgments agree after remapping.
Contradictory judgments become `inconsistent`; insufficient budget or malformed/missing
judgments become `abstain` instead of silently becoming ties.

Records include normalized verdicts, judge provider/model, order consistency,
self-judge risk, resource use, and an abstention reason. The lower-cost single-pass
`llm_pairwise` mode remains available but now also abstains on malformed output.

### Wilson confidence intervals

Evaluation and Lab accuracy include a two-sided 95% Wilson interval. Lab complementarity
also includes Wilson intervals for oracle accuracy and all-provider co-failure. Paired
strategy comparisons still require a paired test; marginal intervals alone do not answer
the paired question.

### Cost accounting

Provider configuration accepts separate input/output USD prices per million tokens.
Calls, workflows, graders, evaluations, and Lab runs expose estimated cost. An aggregate
is `null` unless every executed call in its scope is priced. Cached, reasoning, image, and
other separately billed classes remain future work. No static cloud price table is treated
as permanently current.

### Panel complementarity report

Lab result-card schema v2 derives from direct-provider outcomes:

- best-observed-single and selection-oracle accuracy;
- all-provider co-failure rate;
- pairwise correctness disagreement;
- marginal oracle contribution from each provider;
- confidence intervals for binomial quantities.

Selection-oracle accuracy is a ceiling only for policies that select one direct member
answer. It is a diagnostic reference, not a ceiling, for generative synthesis.

The full protocol is specified in
[Evaluation methodology](concepts/evaluation-methodology.md) and
[Benchmarking OpenFusion](research/benchmarks.md).

## System comparison

| System | Public implementation | Self-hostable/local models | Learned orchestrator | Workflow transparency | Primary role |
|---|---|---|---|---|---|
| OpenFusion | Yes | Yes | No | Public bounded plan and operational trace | Local/cloud fusion research runtime |
| [OpenRouter Fusion](https://openrouter.ai/docs/guides/features/plugins/fusion) | Managed service | No direct local hosting | No public learned-router claim | Managed structured analysis | Hosted multi-model deliberation product |
| [Sakana Fugu](https://sakana.ai/fugu-release/) | No public implementation | No | Yes, product/research positioning | Not a local trace runtime | Learned orchestration system |
| [LiteLLM](https://github.com/BerriAI/litellm) | Yes | Through configured backends | No | Gateway observability | Provider gateway, routing, keys, and budgets |

OpenFusion can sit above a gateway such as LiteLLM. That arrangement separates workflow
semantics from provider access, key management, and load balancing.

## Observability direction

The official
[OpenTelemetry GenAI semantic-conventions repository](https://github.com/open-telemetry/semantic-conventions-genai)
now defines inference, workflow, agent, tool, evaluation, and MCP signals. As of the
research cutoff, the split repository's version metadata tracks core semantic conventions
v1.43.0. The GenAI conventions remain under active development without a formal
split-repository release or settled schema URL.

A future optional exporter should use a workflow root span and provider-call child spans,
pin its schema version, and emit operational metadata such as model, stage, tokens,
latency, remaining budget, and error type. Prompt, response, system-instruction, tool, and
candidate content can contain PII or secrets and must remain off by default. OpenFusion
will not emit hidden chain-of-thought.

## Research questions suitable for OpenFusion Lab

1. When does same-model sampling beat a mixed-provider panel at equal tokens?
2. Does a ranker identify the oracle-available answer often enough to justify its call?
3. Which providers add marginal oracle coverage rather than correlated errors?
4. At what calibrated risk level should an uncertainty cascade escalate?
5. Does a dynamic route/cascade policy beat a fixed provider order after accounting for
   estimator calls?
6. How quickly does a router become stale after a model alias or price change?
7. Do conclusions survive an independent judge family and order-balanced grading?

Negative results are welcome. They help identify tasks where the simplest direct model is
the correct deployment choice.

## Claim boundary

Do not claim an OpenFusion benchmark gain unless the result includes:

- a representative, versioned dataset;
- best-single and direct-provider baselines;
- an equal or explicitly reported budget;
- calls, tokens, cost assumptions, latency, and failures;
- the grader and judge protocol;
- uncertainty intervals and paired evidence where appropriate;
- exact provider/model identifiers and evaluation date.

Every conclusion must remain scoped to that experiment. More models and more calls can
improve, match, or degrade quality.
