# Related work

OpenFusion is an engineering runtime for bounded, API-level inference workflows. The
papers and systems below motivate individual design patterns; citing them does not mean
that OpenFusion reproduces their training setup, evaluation data, or reported gains.

The literature is unusually sensitive to model versions, judge choice, prompt templates,
and inference budget. Results from a preprint are identified as such, and numerical claims
should not be transferred to a different provider panel without a new experiment.

## Ranking and generative fusion

[LLM-Blender](https://aclanthology.org/2023.acl-long.792/) (ACL 2023,
peer reviewed) separates ensembling into a learned pairwise ranker and a generative fuser.
It is the closest academic precedent for rank-then-fuse workflows. OpenFusion's
`pairwise_rank_fuse` uses prompting rather than LLM-Blender's trained PairRanker, so the
two methods are not equivalent.

[Self-consistency](https://openreview.net/forum?id=1PL1NIMMrw) (ICLR 2023,
peer reviewed) samples multiple reasoning paths and aggregates their answers. It motivates
same-model sampling, voting, and `best_of_n`, but its gains were established on particular
reasoning tasks rather than arbitrary open-ended chat.

## Mixture of Agents and same-model sampling

[Mixture-of-Agents](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5434be94e82c54327bb9dcaf7fca52b6-Abstract-Conference.html)
(ICLR 2025 Spotlight; preprint first posted 2024-06-07) passes outputs from one
layer of models to the next. It
reported strong results on conversational benchmarks, including LLM-judged benchmarks.
OpenFusion's `layered_refinement` implements a bounded version of the information-flow
pattern; it is not the original Together AI implementation.

[Rethinking Mixture-of-Agents / Self-MoA](https://arxiv.org/abs/2502.00674)
(first posted 2025-02-02, unreviewed preprint/submission) found that repeated samples from a
strong model often beat panels that add weaker models. Its central warning is more durable
than any headline score: diversity helps only when the added candidates contribute useful,
selectable information. OpenFusion therefore treats `self_moa`, `self_moa_seq`, and mixed
panels as competing hypotheses to evaluate, not as an ordering of quality.

[More Agents Is All You Need](https://openreview.net/forum?id=bgzUSZ8aeg)
(TMLR, published 2024-10) reports scaling from repeated instantiated agents under several
aggregation settings. Its positive result does not remove the need to measure correlated
failures, selector quality, and diminishing returns.

Very recent work on the
[co-failure ceiling](https://arxiv.org/abs/2606.27288) (posted 2026-06-25,
unreviewed preprint) proposes the all-model-wrong rate as an upper bound for methods that
must select one member answer. This is a useful Lab diagnostic hypothesis, but the paper is
too recent to treat as settled evidence.

Two 2026 preprints sharpen *why* equal-weight mixing of unequal models can actively degrade
the strongest member. Work on structured multi-LLM message passing describes
[anchor corruption](https://arxiv.org/abs/2606.00405) (posted 2026, unreviewed preprint),
where naive aggregation drags a high-reliability model down toward weaker
neighbours, and mitigates it with asymmetric damping that shields trusted models. The
[consensus trap](https://arxiv.org/abs/2604.17139) (posted 2026, unreviewed preprint)
analyses response-level voting through epistemic social choice and shows that correlated
failures break the independence assumption that majority voting relies on. OpenFusion's
`uncertainty_cascade` and calibrated escalation (see the
[Spark case study](case-study-spark-escalation.md)) take the same lesson from the opposite
direction: rather than damp votes, detect low-confidence consensus and hand off to a
stronger model — a cheaper, transparent special case of trust-weighted aggregation.

## Routing and cascading

[Hybrid LLM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b47d93c99fa22ac0b377578af0a1f63a-Abstract-Conference.html)
(ICLR 2024, peer reviewed) routes between a small and a large model using predicted query
difficulty and a tunable quality target. It reported up to 40% fewer calls to the large
model without a quality drop in its experiments.

[RouterBench](https://arxiv.org/abs/2403.12031) (first posted 2024-03-18,
preprint) introduced a routing benchmark with more than 405,000 recorded model outcomes.
It is useful for comparing routers, but static historical outcomes cannot represent every
new model, price, latency regime, or production distribution.

[RouteLLM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html)
(ICLR 2025, peer reviewed) learns a strong-versus-weak router from preference data and
reported more than 2x cost reduction at matched quality on its evaluated settings. A
learned router also creates calibration, drift, and supervision requirements that
OpenFusion's current transparent heuristic planner does not have.

[A Unified Approach to Routing and Cascading for LLMs](https://icml.cc/virtual/2025/poster/46183)
(ICML 2025, peer reviewed) derives a combined cascade-routing policy and identifies a good
per-query quality estimator as the critical component. Its framework can skip or reorder
models instead of following only a fixed small-to-large chain.

[LLMRouterBench](https://arxiv.org/abs/2601.07206) (posted 2026-01-12,
preprint) re-evaluates ten routing baselines over more than 400,000 instances, 21 datasets,
and 33 models. It reports that several sophisticated routers do not reliably beat simple
baselines, larger model pools show diminishing returns, and model curation matters. These
findings are directly relevant to OpenFusion Lab, but remain preprint results.

[R2-Router](https://openreview.net/forum?id=S3m1tSp8F4) (accepted ICML 2026;
published on OpenReview 2026-04-30) jointly chooses the model and output-length budget. It
reported a 4--5x cost reduction relative to its routing baselines. The length instruction
is a soft control unless the serving backend also enforces a token limit.

[UniScale](https://arxiv.org/abs/2605.30898) (accepted ICML 2026; arXiv
version posted 2026-05-29) treats model routing and test-time scaling as one online action
space. It motivates choosing a tuple such as `(provider, strategy, samples, token budget)`
rather than choosing only a provider. Online adaptation should remain optional because it
adds feedback and exploration requirements.

## Budget-aware test-time compute

[Scaling LLM Test-Time Compute Optimally](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html)
(ICLR 2025 oral) found that the best allocation method varies with problem difficulty. Its
compute-optimal policy was more than 4x as efficient as uniform best-of-N on the studied
mathematics setting. The result supports adaptive budgets, not a universal claim that more
sampling improves every prompt.

[Zero-Overhead Introspection](https://openreview.net/forum?id=GqZYGOYuF2)
(ICLR 2026, peer reviewed) predicts reward and remaining length from otherwise unused
logits and adaptively continues samples. This is relevant to local, white-box backends; the
method is not directly available through most black-box OpenAI-compatible APIs.

[The Shadow Price of Reasoning](https://arxiv.org/abs/2606.03092) (accepted
ICML 2026; posted 2026-06-02) allocates a shared inference budget across a traffic stream.
It reported up to 3x aggregate accuracy versus uniform allocation under scarce budgets.
Its global optimization setting differs from OpenFusion's current per-request hard call
cap, but it motivates explicit cost and token ledgers.

## LLM-as-a-judge reliability

[MASEval](https://aclanthology.org/2026.acl-demo.34/) (ACL 2026 System
Demonstrations) treats the complete agentic system as the evaluation unit, including
topology, orchestration logic, context/harness design, and error handling. Its comparison
found framework choice mattered as much as model choice among the tested combinations.
That result supports measuring OpenFusion workflows end to end rather than attributing a
final score only to the underlying model; it does not establish the same effect size for
OpenFusion.

[Judging LLM-as-a-Judge](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)
(NeurIPS 2023 Datasets and Benchmarks) found useful agreement with humans while also
documenting position, verbosity, self-enhancement, and reasoning biases.

[LLM Evaluators Recognize and Favor Their Own Generations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html)
(NeurIPS 2024) provides evidence that evaluator self-recognition contributes to
self-preference. Evaluation should therefore record whether a judge also generated a
candidate and should prefer an independent judge family when practical.

[A Systematic Study of Position Bias](https://aclanthology.org/2025.ijcnlp-long.18/)
(IJCNLP-AACL 2025, published 2025-12) evaluated 15 judges over more than 150,000
judgments. It recommends measuring repetition stability, position consistency, and
preference fairness rather than assuming one prompt order is neutral.

[How Reliable is Multilingual LLM-as-a-Judge?](https://aclanthology.org/2025.findings-emnlp.587/)
(Findings of EMNLP 2025) evaluated 25 languages and reported average Fleiss' kappa of
about 0.3, with particularly weak consistency in lower-resource languages. English-only
judge calibration must not be generalized to multilingual evaluations.

[BiasScope](https://openreview.net/forum?id=QGOw6AU8Lp) (ICLR 2026) uses
automated perturbations to discover judge biases and reports error rates above 50% for
strong models on its adversarial JudgeBench-Pro extension. This reinforces the need for
abstention and objective graders; it does not imply that ordinary judge error is always
above 50%.

## Confidence and uncertainty

[Can LLMs Express Their Uncertainty?](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6733cf15e10e2cd1d59af033c3bb8507-Abstract-Conference.html)
(ICLR 2024) found that verbalized confidence is commonly overconfident and that no tested
black-box elicitation method dominates across difficult tasks. Consistency sampling helps,
but an emitted number should not be presented as a calibrated probability without held-out
calibration evidence.

[Taming Overconfidence in LLMs](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29fb6e1456b3d8b57ede5c45aa2c6537-Abstract-Conference.html)
(ICLR 2025) found that RLHF can amplify verbalized overconfidence. OpenFusion's
`uncertainty_cascade` should therefore combine confidence with format validation,
consistency, and task-specific checks.

## Managed and learned orchestration systems

[OpenRouter Fusion](https://openrouter.ai/docs/guides/features/plugins/fusion) is a
managed multi-model product. Its panel selection, provider integrations, and published
product comparisons are not an open training or evaluation recipe. OpenFusion's
`parallel_synthesis` is only a conceptual analogue.

[Sakana Fugu](https://sakana.ai/fugu-release/) is a managed learned-orchestration product;
its [technical report](https://arxiv.org/abs/2606.21228) was posted 2026-06-19.
[The Conductor](https://arxiv.org/abs/2512.04388) and
[TRINITY](https://openreview.net/forum?id=5HaRjXai12) are accepted ICLR 2026 work on
learned coordination through natural-language delegation and evolved model/role selection,
respectively. OpenFusion's `adaptive` strategy instead uses readable heuristics or a
constrained JSON plan and must not be described as a trained orchestrator.

## Observability standards

The [OpenTelemetry GenAI semantic-conventions repository](https://github.com/open-telemetry/semantic-conventions-genai)
now covers inference, workflows, agents, tools, evaluation, and MCP. As of 2026-07-10 the
split repository has no formal release and its schema URL remains unsettled; its version
metadata tracks core semantic conventions v1.43.0. The GenAI conventions remain under
active development, so an OpenFusion exporter should pin an explicitly verified schema
version. Prompt, response,
tool, and system-instruction content can contain secrets or PII and should remain disabled
by default.

## What the literature does not establish

No cited work establishes that:

- every extra model or sample improves an answer;
- an LLM judge is a ground-truth oracle;
- self-reported confidence is calibrated across tasks;
- a router trained on one provider pool remains calibrated after model or price changes;
- benchmark gains survive a different call, token, latency, or dollar budget.

OpenFusion consequently treats every strategy as an evaluation target and keeps workflow
plans bounded by `max_total_calls`.
