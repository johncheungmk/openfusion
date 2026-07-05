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

## Run the sample lab

```powershell
openfusion lab validate examples/lab_local_small.yaml
openfusion lab run examples/lab_local_small.yaml --out results-local-small.json
openfusion lab recommend results-local-small.json
```
