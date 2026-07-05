# Quickstart with Ollama

This guide uses one local Ollama server and two small models.

## Pull models

```powershell
ollama pull llama3.2:3b
ollama pull qwen3:latest
```

Check the models:

```powershell
ollama list
```

Check the OpenAI-compatible endpoint:

```powershell
curl.exe http://127.0.0.1:11434/v1/models
```

## Create OpenFusion config

```powershell
openfusion init --path openfusion.yaml
notepad openfusion.yaml
```

Example providers:

```yaml
providers:
  - name: ollama-llama32-3b
    type: openai_compatible
    enabled: true
    base_url: http://127.0.0.1:11434/v1
    api_key_env:
    model: llama3.2:3b
    timeout_seconds: 180
    weight: 1.0

  - name: ollama-qwen3
    type: openai_compatible
    enabled: true
    base_url: http://127.0.0.1:11434/v1
    api_key_env:
    model: qwen3:latest
    timeout_seconds: 240
    weight: 1.2
```

## Start the API server

```powershell
openfusion serve --config openfusion.yaml --port 8000
```

Test:

```powershell
curl.exe http://127.0.0.1:8000/health
```
