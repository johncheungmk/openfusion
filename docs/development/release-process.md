# Release Process

OpenFusion releases are cut from a reviewed commit on `main`. Use a release branch and pull
request, then tag the exact merged commit and publish a GitHub Release. Published tags are
immutable: correct a bad release with a new patch version rather than moving its tag.

## 1. Choose the version and freeze metadata

Update every current-version location in the same change:

- `pyproject.toml` (`project.version`);
- `src/openfusion/__init__.py` (`__version__`);
- version assertions in `tests/`;
- the current-version statement in `README.md` and any version-specific current docs;
- the new top entry in `CHANGELOG.md`;
- `CITATION.cff` (`version` and the actual `date-released`).

Do not rewrite historical changelog entries or archived migration documents just to make their
old version numbers match the new release. Before tagging, search for unexpected current-version
references and confirm the package and runtime agree:

```bash
python -c "import tomllib; from pathlib import Path; from openfusion import __version__; p = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8')); assert p['project']['version'] == __version__; print(__version__)"
```

## 2. Run the release gate

Create or activate a clean development environment, then install the declared test and
documentation dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Run the repository checks with no provider services or external APIs:

```bash
python -m compileall -q src tests
ruff check src tests
pytest -q
python -m build
mkdocs build --strict
git diff --check
```

Validate `CITATION.cff` against CFF 1.2.0 when `cffconvert` is available:

```bash
cffconvert --validate -i CITATION.cff
```

Inspect `dist/` and confirm the wheel and source archive both use the intended version. The source
archive must contain `README.md`, `CHANGELOG.md`, `CITATION.cff`, `LICENSE`, and the packaged
configuration example. Never include `.env`, `openfusion.yaml`, credentials, private endpoints, or
local result files.

## 3. Test the built wheel

Create a disposable virtual environment and install the wheel itself, not the editable checkout.
Replace `<wheel>` with the exact file from `dist/`.

```bash
python -m venv .release-venv
```

On Linux or macOS (replace `X.Y.Z` with the release version):

```bash
.release-venv/bin/python -m pip install "dist/open_fusion_ai-X.Y.Z-py3-none-any.whl"
.release-venv/bin/python -m pip check
.release-venv/bin/python -c "import openfusion; print(openfusion.__version__)"
.release-venv/bin/openfusion --help
```

On Windows PowerShell (replace `X.Y.Z` with the release version):

```powershell
.\.release-venv\Scripts\python.exe -m pip install "dist/open_fusion_ai-X.Y.Z-py3-none-any.whl"
.\.release-venv\Scripts\python.exe -m pip check
.\.release-venv\Scripts\python.exe -c "import openfusion; print(openfusion.__version__)"
.\.release-venv\Scripts\openfusion.exe --help
```

The printed version must match the wheel filename, `pyproject.toml`, and `CITATION.cff`.

## 4. Review, merge, tag, and publish

1. Push the release branch and open a pull request.
2. Review the complete diff, wait for CI and documentation checks, and merge it into `main`.
3. Pull `main` in a clean checkout and verify the commit SHA that passed CI.
4. Create an annotated tag such as `v0.6.0` on that merged commit and push the tag.
5. Publish a GitHub Release for the tag. Use the matching changelog entry as the release notes and
   attach the wheel and source archive produced from the tagged tree.
6. Confirm the GitHub Pages deployment completed and the public documentation describes the same
   version.
7. Install the published wheel or release asset once more in a fresh environment and run the import
   and CLI smoke checks.

If any post-publication defect changes behavior or metadata, prepare a new patch release. Do not
delete or retarget an existing public tag.
