# Slow CPU Models

Local CPU inference can be slow, especially on large models.

## Symptoms

- Requests take minutes.
- Fusion strategies are much slower than a single model.
- Self-MoA or parallel synthesis appears unusable.

## Fixes

Use smaller models first:

```text
llama3.2:3b
qwen3:latest
```

Reduce output length:

```yaml
max_tokens: 16
```

Reduce call budget:

```yaml
max_total_calls: 2
```

Use fewer strategies:

```yaml
strategies:
  - name: fallback
  - name: semantic_vote
  - name: uncertainty_cascade
```

## Reminder

Fusion may improve quality on some tasks, but it increases model calls and latency. Always compare against the best single-model baseline.
