# Environment Variables

OpenFusion keeps secrets out of YAML files.

Common variables:

```dotenv
OPENFUSION_API_KEY=replace-with-a-long-random-token
OPENAI_API_KEY=
OPENROUTER_API_KEY=
OLLAMA_API_KEY=
LMSTUDIO_API_KEY=
```

For local-only Ollama tests, provider API keys are usually not required.

Set `OPENFUSION_API_KEY` before exposing the server beyond localhost.
