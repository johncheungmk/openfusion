from __future__ import annotations

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="replace-with-a-long-random-token")

completion = client.chat.completions.create(
    model="openfusion/panel-judge",
    messages=[{"role": "user", "content": "Compare vLLM and Ollama for a campus AI lab."}],
    extra_body={
        "fusion_strategy": "panel_judge",
        # Optional: override the configured panel.
        # "fusion_panel": ["local-ollama", "cloud-openai"],
        # "fusion_judge": "cloud-openai",
    },
)
print(completion.choices[0].message.content)
