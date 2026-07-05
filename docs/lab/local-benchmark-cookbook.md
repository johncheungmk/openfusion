# Local Benchmark Cookbook

This cookbook runs a small local benchmark on two Ollama models.

The purpose is not to claim benchmark performance. The purpose is to check whether fusion improves results on your hardware and dataset.

## Windows PowerShell

```powershell
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Pull two models:

```powershell
ollama pull llama3.2:3b
ollama pull qwen3:latest
```

Run the sample lab:

```powershell
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

## How to interpret the result

Look at:

- best single-model baseline;
- fallback baseline;
- delta versus fallback;
- delta versus best single model;
- average latency;
- call ratio;
- token ratio.

If fusion does not improve accuracy, that is still a useful result.
