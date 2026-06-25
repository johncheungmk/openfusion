#!/usr/bin/env bash
set -euo pipefail

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${OPENFUSION_API_KEY:-replace-with-a-long-random-token}" \
  -d '{
    "model": "openfusion/critique-revision",
    "messages": [
      {"role": "user", "content": "Give a short plan for building a RAG chatbot."}
    ],
    "max_tokens": 256,
    "fusion_max_total_calls": 6
  }' | python -m json.tool
