# Ollama Troubleshooting

Check installed models:

```powershell
ollama list
```

Check running models:

```powershell
ollama ps
```

Check the OpenAI-compatible endpoint:

```powershell
curl.exe http://127.0.0.1:11434/v1/models
```

If OpenFusion reports `model not found`, ensure the model name in YAML exactly matches `ollama list`.
