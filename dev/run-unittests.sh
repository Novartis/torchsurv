#!/bin/bash
# Run all unit tests

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e

cd "${DIR}/.."

export PYTHONPATH="${DIR}/../src"

python -m unittest discover -s tests $@