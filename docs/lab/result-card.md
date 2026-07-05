# Result Card

OpenFusion Lab writes a result card JSON file for reproducible local experiments.

A result card includes:

- OpenFusion version;
- experiment metadata;
- platform and Python version;
- dataset metadata and hash;
- engine and model metadata;
- single-model baselines;
- strategy summaries;
- deltas versus fallback and best single model;
- recommendations and warnings.

## Privacy

By default, result cards do not include secrets, API keys, provider headers, or raw prompts.

Do not publish result cards containing private datasets unless you have reviewed the file.
