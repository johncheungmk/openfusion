# Publishing OpenFusion v0.2.0 to GitHub

Use a release branch in a fresh clone. This avoids mixing old generated files, local secrets, or virtual environments into the release.

The examples assume the downloaded source archive is named `openfusion-v0.2.0-source.zip` and contains a top-level `openfusion/` directory.

## Before starting

- Back up any local `.env` and `openfusion.yaml`; neither belongs in Git.
- Confirm Git has your identity: `git config --global user.name` and `git config --global user.email`.
- GitHub CLI (`gh`) is optional. Without it, push the branch and create the pull request and release in the GitHub web interface.

## Windows PowerShell

### 1. Extract the clean source package

```powershell
$zip = "$env:USERPROFILE\Downloads\openfusion-v0.2.0-source.zip"
$extract = "$env:USERPROFILE\Downloads\openfusion-v0.2.0-source"

Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force

$source = Join-Path $extract "openfusion"
Get-ChildItem -Force $source
```

The listing must contain `README.md`, `pyproject.toml`, `src`, `tests`, `docs`, `assets`, and `.github`.

### 2. Create a release branch in a fresh clone

```powershell
cd $env:USERPROFILE
Remove-Item -Recurse -Force .\openfusion-release -ErrorAction SilentlyContinue
git clone https://github.com/johncheungmk/openfusion.git openfusion-release
cd .\openfusion-release
git checkout -b release/v0.2.0

# Replace the tracked v0.1 tree while preserving .git.
git rm -r --ignore-unmatch .
Get-ChildItem -Force $source | Copy-Item -Destination . -Recurse -Force
```

If Git has not been configured on this computer, set your real GitHub identity before committing:

```powershell
git config --global user.name "John Cheung"
git config --global user.email "YOUR-VERIFIED-GITHUB-EMAIL"
```

### 3. Validate the exact tree to be published

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m compileall -q src tests
ruff check src tests
pytest -q
python -m build
openfusion strategies

git add -A
git diff --cached --check
git status --short
```

Do not continue if tests, Ruff, the build, or `git diff --cached --check` fails. Confirm `.env`, `openfusion.yaml`, `.venv`, caches, and `dist/` are not staged.

### 4. Commit and push the release branch

```powershell
git commit -m "Release OpenFusion v0.2.0 orchestration workflows"
git push -u origin release/v0.2.0
```

### 5. Open a pull request

With GitHub CLI:

```powershell
gh auth status
gh pr create `
  --base main `
  --head release/v0.2.0 `
  --title "OpenFusion v0.2.0" `
  --body "Adds bounded multi-model orchestration, voting, critique-revision, layered refinement, adaptive planning, evaluation, and updated documentation."
```

Without GitHub CLI, open the repository in a browser. GitHub will offer **Compare & pull request** for `release/v0.2.0`. Review the **Files changed** tab, wait for CI, and merge only after it passes.

### 6. Tag the merged commit and create the GitHub release

After the pull request is merged:

```powershell
git checkout main
git pull --ff-only origin main

Remove-Item -Recurse -Force .\dist -ErrorAction SilentlyContinue
python -m build

git tag -a v0.2.0 -m "OpenFusion v0.2.0"
git push origin v0.2.0

gh release create v0.2.0 `
  .\dist\open_fusion_ai-0.2.0-py3-none-any.whl `
  .\dist\open_fusion_ai-0.2.0.tar.gz `
  --verify-tag `
  --title "OpenFusion v0.2.0" `
  --generate-notes
```

Without `gh`, open **Releases → Draft a new release**, select the existing `v0.2.0` tag, attach the two files from `dist`, generate release notes, and publish.

## Linux / macOS

### 1. Extract, clone, and replace the tree

```bash
zip="$HOME/Downloads/openfusion-v0.2.0-source.zip"
extract="$HOME/Downloads/openfusion-v0.2.0-source"

rm -rf "$extract" ~/openfusion-release
mkdir -p "$extract"
unzip "$zip" -d "$extract"
source_tree="$extract/openfusion"
ls -la "$source_tree"

cd ~
git clone https://github.com/johncheungmk/openfusion.git openfusion-release
cd openfusion-release
git checkout -b release/v0.2.0
git rm -r --ignore-unmatch .
cp -a "$source_tree"/. .
```

### 2. Validate, commit, and push

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m compileall -q src tests
ruff check src tests
pytest -q
python -m build
openfusion strategies

git add -A
git diff --cached --check
git status --short
git commit -m 'Release OpenFusion v0.2.0 orchestration workflows'
git push -u origin release/v0.2.0
```

### 3. Pull request, tag, and release

```bash
gh auth status
gh pr create \
  --base main \
  --head release/v0.2.0 \
  --title 'OpenFusion v0.2.0' \
  --body 'Adds bounded multi-model orchestration, voting, critique-revision, layered refinement, adaptive planning, evaluation, and updated documentation.'
```

After CI passes and the pull request is merged:

```bash
git checkout main
git pull --ff-only origin main
rm -rf dist
python -m build

git tag -a v0.2.0 -m 'OpenFusion v0.2.0'
git push origin v0.2.0

gh release create v0.2.0 \
  dist/open_fusion_ai-0.2.0-py3-none-any.whl \
  dist/open_fusion_ai-0.2.0.tar.gz \
  --verify-tag \
  --title 'OpenFusion v0.2.0' \
  --generate-notes
```

## Local files that must remain untracked

`.env`, `openfusion.yaml`, `.venv`, caches, test output, and `dist/` are ignored. Check `git status --short` before every commit and never publish credentials.
