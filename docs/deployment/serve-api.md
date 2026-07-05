# API Server

Start the OpenAI-compatible API server:

```powershell
openfusion serve --config openfusion.yaml --port 8000
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Chat completions endpoint:

```text
POST /v1/chat/completions
```

Use `127.0.0.1` for local testing. Set `OPENFUSION_API_KEY` before exposing the service beyond localhost.
