# Contributing

1. Create a branch from `main`.
2. Install development dependencies with `python -m pip install -e ".[dev]"`.
3. Add tests that use fake providers; tests must not call external model APIs.
4. Run `ruff check src tests` and `pytest -q`.
5. Keep secrets and local configuration out of commits.
6. Describe model-call, latency, privacy, and compatibility effects in the pull request.

New orchestration strategies should have a hard call bound, a public trace, graceful provider-failure behavior, and documentation explaining which task types they suit.
