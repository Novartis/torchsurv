# Development notes

## Set up a development environment via conda

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
scripts from there that import torchsurv. Alternatively you may give
an absolute path to the source directory.

```bash
export PYTHONPATH=src
```

To test building the package, run:

```bash
python -m build # builds the package as a tarball -> dist
```

To get a local installation for testing e.g. other tools that depend on torchsurv
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
