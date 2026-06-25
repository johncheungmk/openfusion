from __future__ import annotations

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="replace-with-a-long-random-token")

completion = client.chat.completions.create(
    model="openfusion/layered-refinement",
    messages=[{"role": "user", "content": "Compare vLLM and Ollama for a campus AI lab."}],
    max_tokens=256,
    extra_body={
        "fusion_samples_per_provider": 1,
        "fusion_refinement_rounds": 1,
        "fusion_max_total_calls": 8,
    },
)
print(completion.choices[0].message.content)
