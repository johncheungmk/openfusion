# What is Model Fusion?

Model fusion means using more than one model call to produce, vote on, select, critique, or synthesize a final answer.

OpenFusion performs inference-time fusion. It does not merge model weights.

Fusion can improve some tasks, but it can also reduce quality or increase latency. Always compare against each base model and the best single-model baseline.
