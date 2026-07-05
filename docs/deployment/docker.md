# Docker

Build:

```bash
docker build -t openfusion .
```

Run:

```bash
docker run --rm -p 8000:8000 openfusion
```

For local Ollama on the host, ensure the container can reach the host network endpoint.
