# Slow CPU Models

CPU inference can be slow. Use smaller models first and keep `max_tokens` and `max_total_calls` low.

Recommended smoke-test models:

- `llama3.2:3b`
- `qwen3:latest`

For MCQ tests, use `max_tokens: 16`.
