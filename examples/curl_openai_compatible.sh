#!/usr/bin/env bash
set -euo pipefail

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${OPENFUSION_API_KEY:-replace-with-a-long-random-token}" \
  -d '{
    "model": "openfusion/panel-judge",
    "messages": [
      {"role": "user", "content": "Give a short plan for building a RAG chatbot."}
    ],
    "fusion_strategy": "panel_judge"
  }' | python -m json.tool
