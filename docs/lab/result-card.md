# Result Card

OpenFusion Lab writes `openfusion-lab-result-v2` JSON with experiment/dataset hashes,
model metadata, direct baselines, strategy summaries, Wilson accuracy intervals,
end-to-end latency percentiles, cost coverage, baseline comparisons, recommendations,
warnings, and a panel-complementarity report. The loader remains compatible with v1
cards by applying defaults for new fields.

The complementarity section reports selection-oracle accuracy, best-observed-single
accuracy, all-model co-failure, pairwise correctness disagreement/both-wrong rates, and
each provider's marginal oracle contribution. Selection-oracle accuracy assumes knowledge
of which direct answer is correct. It is a ceiling only for member-selection policies and
merely a diagnostic reference for generative synthesis.

The result card omits engine base URLs, API-key environment-variable names, and provider
headers. Do not add secrets, API keys, private endpoints, or raw private prompts to a
shareable card.

Schema v2 also stores the recommendation objective, latency cap, and call preference so
`lab recommend` can reproduce the original primary choice. The primary recommendation is
the configured objective's best eligible strategy; all four unfiltered objective leaders
remain visible for audit.
