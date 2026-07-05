# Contributing

Run before submitting changes:

```bash
python -m compileall -q src tests
ruff check src tests
pytest -q --basetemp .pytest-tmp
python -m build
mkdocs build --strict
```
