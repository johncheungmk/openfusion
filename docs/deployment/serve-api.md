# API Server

Start the OpenAI-compatible API server:

```bash
openfusion serve --config openfusion.yaml --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```
