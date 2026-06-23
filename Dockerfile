FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config.example.yaml ./openfusion.yaml
EXPOSE 8000
CMD ["openfusion", "serve", "--config", "openfusion.yaml", "--host", "0.0.0.0"]
