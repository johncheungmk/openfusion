# Evaluation guide

OpenFusion does not assume that more calls produce a better answer. Evaluate each strategy on representative tasks.

## Dataset format

One JSON object per line:

```json
{"id":"math-1","prompt":"Return only the answer: 2 + 2","reference":"4"}
{"id":"mcq-1","prompt":"Answer: A, B, C, or D","reference":["B","b"],"answer_regex":"Answer:\\s*([A-D])"}
```

Fields:

- `id`: unique case ID;
- `prompt`: user message;
- `reference`: string or list of accepted strings;
- `system`: optional system message;
- `answer_regex`: optional extraction regex; group 1 is used when present;
- `metadata`: optional object preserved by the loader.

## Run

```bash
openfusion evaluate examples/eval_sample.jsonl \
  --config openfusion.yaml \
  --strategy fallback \
  --output fallback.json

openfusion evaluate examples/eval_sample.jsonl \
  --config openfusion.yaml \
  --strategy weighted_vote \
  --output weighted-vote.json
```

Compare at least:

- accuracy or task-specific quality;
- total model calls;
- total tokens;
- wall-clock latency;
- financial cost;
- failure rate.

The built-in score is normalized exact match. Use a domain-specific test executor, citation checker, retrieval-grounding grader, or blinded human evaluation for open-ended tasks.
