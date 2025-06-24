#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e

cd "${DIR}/.."

if [ "$1" == "check" ]; then
    CHECK="--check"
else
    CHECK=""
fi

export PYTHONPATH=${DIR}/../src

isort ${CHECK} src tests
pylint src
