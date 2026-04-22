# Development notes

## Clone and set up pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to ensure clean commits in the repository. To set up
pre-commit in a fresh repository clone, please run:

```bash
pre-commit install
```

This only needs to do once and ensures code formatting & checks are performed on each commit.

## Set up a development environment

[uv](https://docs.astral.sh/uv/) is a fast Python package manager from Astral (the team behind `ruff`).
It handles environment creation, dependency resolution, building, and publishing in a single tool.

Install uv (if not already installed):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or see https://docs.astral.sh/uv/getting-started/installation/ for more options
```

Create the environment and install all dependencies:

```bash
uv sync --group dev
```

This creates a `.venv` virtual environment and installs the project in editable mode with all dev
dependencies. To also include documentation dependencies:

```bash
uv sync --group dev --extra docs
```

Run commands in the project environment (no manual activation needed):

```bash
uv run pytest              # run tests
uv run ruff check .        # lint
uv run mypy src/           # type-check
uv run pre-commit install  # set up git hooks
```

> **Tip:** You can also activate the virtual environment directly with
> `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows),
> then run commands without the `uv run` prefix.

<details>
<summary><em>Legacy: conda-based setup</em></summary>

> **Note:** conda-based development is no longer actively maintained. The `dev/environment.yml`
> file is provided for convenience but may drift from the canonical dependencies in
> `pyproject.toml`. We recommend using uv instead.

```bash
conda create -y -n torchsurv python=3.10
conda env update -n torchsurv -f dev/environment.yml
conda activate torchsurv
```

</details>

## Test and develop the package

Run all tests:

```bash
uv run pytest
```

You can also use `dev/run-unittests.sh` which sets up `PYTHONPATH` automatically:

```bash
uv run ./dev/run-unittests.sh
```

## Building the package

```bash
uv build                   # outputs dist/*.whl and dist/*.tar.gz
```

To get a local installation for testing e.g. other tools that depend on `TorchSurv`
with a local development version of the code:

```bash
uv sync                    # editable install is the default
```

## Installing a local package build

To install a specific package build e.g. into a local virtual environment:

```bash
uv build
uv run --with dist/torchsurv-*.whl -- python -c "from torchsurv.loss import cox"
```

## Code formatting

We use `ruff` for code formatting and linting. To format and check the code, run:

```bash
uv run ruff format .
uv run ruff check .

# Or use the convenience script
uv run ./dev/codeqc.sh
```

Code should be properly formatted before merging with `dev` or `main`.

## Build and test the documentation locally

We use Sphinx to build the documentation. To build and view the documentation,
run the following script:

```bash
# just run ./dev/build-docs.sh to build but not start a webserver
uv run ./dev/build-docs.sh serve
```

You can then access the documentation at `http://localhost:8000` in your web browser.

When updating toctrees / adding content you may need to clean your local documentation
build:

```bash
cd docs
make clean
```

To build a single PDF of the documentation, run:

```bash
cd docs
make latexpdf LATEXMKOPTS="-silent -f"
```

There are a few errors in the PDF build due to layout issues,
but the PDF can still be used to summarize the package in a single
file.

## Steps to create a new release

### 1. On PyPI

Follow the steps below, ensure each one is successful.

1. Update the version number in `src/torchsurv/__init__.py`.
2. Ensure [CHANGELOG.md](CHANGELOG.md) is up-to-date with the new version number and changes.
3. Ensure all PRs to be included are merged with `main`, pushed to Github and that all tests & checks have run successfully.
4. Clean up any development artifacts: `rm -rf testenv testenv.bak dist build *.egg-info .pytest_cache __pycache__`
5. Build the release from the latest main branch:

    ```bash
    git checkout main && git pull
    uv build
    ```

6. Upload the package to TestPyPI and verify:

    ```bash
    uv publish --index testpypi
    ```

7. Check if the package can be installed (also check this installs the correct version):

    ```bash
    uv run --with torchsurv --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-project -- python -c "from torchsurv.loss import cox; from torchsurv.metrics.cindex import ConcordanceIndex; print('OK')"
    ```

8. Upload to PyPI:

    ```bash
    uv publish
    ```

9. Check that the package can be installed:

    ```bash
    uv run --with torchsurv --no-project -- python -c "from torchsurv.loss import cox; from torchsurv.metrics.cindex import ConcordanceIndex; print('OK')"
    ```

10. Create a new tag for the release, e.g. `git tag -a v0.1.3 -m "Version 0.1.3"`. Push the tag to Github: `git push origin v0.1.3`. Create a release on Github from the tag via <https://github.com/Novartis/torchsurv/releases>.

### 2. Update on conda-forge

1. Create a fork of <https://github.com/conda-forge/torchsurv-feedstock>
2. Create a branch on this fork, and update the version number in recipe/meta.yaml as well as the sha sum (obtained via `shasum -a 256 dist/torchsurv-<version>.tar.gz` from the file we uploaded to pypi)
3. Finish the checklist in the PR, ensure it builds.
4. Double-check and merge the PR.

The new release should become available some time after this.
