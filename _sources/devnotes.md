# Development notes

## Set up a development environment via conda

If you use Conda, you can install requirements into a conda environment
using the `environment.yml` file included in the `dev` subfolder of the source repository.

Using the package has the following dependencies which will be installed automatically via pip:

* [torch](https://pytorch.org/): Consider pre-installing if you have specific system requirements (CPU / GPU / CUDA version).
* [scipy](https://scipy.org/): We use some statistical helper functions to calculate metrics.
* [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/): We use some statistical helper functions to calculate metrics.

To run the tests and example notebooks, you need to install the following additional packages:

* [lifelines](https://lifelines.readthedocs.io/en/latest/)
* [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/)
* [pytorch_lightning](https://lightning.ai/docs/pytorch/stable/) (and [lightning](https://lightning.ai/))

To use conda to install all package dependencies locally and start development,
run the following commands in the root directory of the repository.

Create the environment:

```bash
conda create -y -n torchsurv python=3.10
conda env update -n torchsurv -f dev/environment.yml
```

Activate the environment:

```bash
conda activate torchsurv
```

## Test and develop the package

To run all unit tests either use `dev/run-unittests.sh` or run the
following command from the repository root directory:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

To run other scripts within the repo, you can run the following
from the top level directory of the repository clone, and then run
scripts from there that import `TorchSurv`. Alternatively you may give
an absolute path to the source directory.

```bash
export PYTHONPATH=src
```

To test building the package, run:

```bash
python -m build # builds the package as a tarball -> dist
```

To get a local installation for testing e.g. other tools that depend on `TorchSurv`
with a local development version of the code:

```bash
pip install -e <repo-clone-directory>  # install for development ("editable")
```

## Installing a local package build

To install a specific package build e.g. into a local conda environment / virtualenv:

```bash
python -mbuild
# ... activate virtualenv
# install the first tarball in dist
pip install "$(set -- dist/*.tar.gz; echo "$1")"
```

## Code formatting

We use `black` for code formatting. To format the code, run the following command:

```bash
./dev/codeqc.sh
```

Code should be properly formatted before merging with `dev` or `main`.

## Build and test the documentation locally

We use Sphinx to build the documentation. To build and view the documentation,
run the following script:

```bash
# just run ./dev/build-docs.sh to build but not start a webserver
./dev/build-docs.sh serve
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

### 1. On pypi

Follow the steps below, ensure each one is successful.

1. Update the version number in `pyproject.toml`.
2. Ensure [CHANGELOG.md](CHANGELOG.md) is up-to-date with the new version number and changes.
3. Ensure all PRs to be included are merged with `main`, pushed to Github and that all tests & checks have run successfully.
4. Build the release from the latest main branch: `git checkout main && git pull && rm -rf dist && python -m build`
5. Check the built package: `python -m twine check dist/*`
6. Upload the package to testPyPI: `python -m twine upload -r testpypi dist/*`
7. Check if the package can be installed (also check this installs the correct version):

    ```bash
    rm -rf testenv # ensure a new test env is created
    python -m virtualenv testenv
    . ./testenv/bin/activate
    # these don't simply install from testpypi
    pip install torch torchmetrics scipy numpy
    pip install -i https://test.pypi.org/simple/ torchsurv
    python
    >>> from torchsurv.loss import cox
    >>> from torchsurv.metrics.cindex import ConcordanceIndex
    ```

8. Upload to pypi: `python -m twine upload -r pypi dist/*`
9. Check that the package can be installed:

    ```bash
    rm -rf testenv # ensure a new test env is created
    python -m virtualenv testenv
    . ./testenv/bin/activate
    pip install torchsurv
    python
    >>> from torchsurv.loss import cox
    >>> from torchsurv.metrics.cindex import ConcordanceIndex
    ```

10. Create a new tag for the release, e.g. `git tag -a v0.1.3 -m "Version 0.1.3"`. Push the tag to Github: `git push origin v0.1.3`. Create a release on Github from the tag via <https://github.com/Novartis/torchsurv/releases>.

### 2. Update on conda-forge

1. Create a fork of <https://github.com/conda-forge/torchsurv-feedstock>
2. Create a branch on this fork, and update the version number in recipe/meta.yaml as well as the sha sum (obtained via `shasum -a 256 dist/torchsurv-<version>.tar.gz` from the file we uploaded to pypi)
3. Finish the checklist in the PR, ensure it builds.
4. Double-check and merge the PR.

The new release should become available some time after this.
