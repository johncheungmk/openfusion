# Contributing

Before opening a pull request, run:

```powershell
python -m compileall -q src tests
ruff check src tests
pytest -q --basetemp .pytest-tmp
python -m build
git diff --check
```

Do not commit local `.env`, `openfusion.yaml`, benchmark results, virtual environments, build artifacts, or cache folders.
