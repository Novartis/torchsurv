#!/bin/bash
# build & preview the documentation locally

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e
cd "${DIR}/../docs"
make html
make html
cd _build/html

if [ "$1" == "serve" ]; then
    python -m http.server --bind 127.0.0.1 8000
fi
