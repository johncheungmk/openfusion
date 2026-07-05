# Installation

## Windows PowerShell

```powershell
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Check the installation:

```powershell
openfusion --help
openfusion strategies
openfusion lab --help
```

## Linux/macOS

```bash
git clone https://github.com/johncheungmk/openfusion.git
cd openfusion

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Check the installation:

```bash
openfusion --help
openfusion strategies
openfusion lab --help
```

## Install from a release wheel

Download the `.whl` file from the GitHub release page, then install it:

```powershell
python -m pip install open_fusion_ai-0.5.2-py3-none-any.whl
```
