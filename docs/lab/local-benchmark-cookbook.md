# Local Benchmark Cookbook

This cookbook runs a small local benchmark on two Ollama models.

## Windows PowerShell

```powershell
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

ollama pull llama3.2:3b
ollama pull qwen3:latest

openfusion lab validate examples/lab_local_small.yaml
openfusion lab engine-plan examples/lab_local_small.yaml
openfusion lab run examples/lab_local_small.yaml --out results-local-small.json
openfusion lab recommend results-local-small.json
```

## Linux/macOS

```bash
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

ollama pull llama3.2:3b
ollama pull qwen3:latest

openfusion lab validate examples/lab_local_small.yaml
openfusion lab engine-plan examples/lab_local_small.yaml
openfusion lab run examples/lab_local_small.yaml --out results-local-small.json
openfusion lab recommend results-local-small.json
```

## Interpretation

Look at the best single-model baseline, fallback baseline, delta versus best single model, average latency, call ratio, and token ratio.

If fusion does not improve accuracy, that is still a useful result.
