#!/bin/bash
# Run all unit tests

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e

cd "${DIR}/.."

export PYTHONPATH="${DIR}/../src"

files=$(find src -type f -name "*.py" -exec grep -l 'if __name__ == "__main__"' {} +)

for f in $files; do
    module=$(echo $f | sed 's/\//./g' | sed 's/\.py//g')
    echo "Running $f -> $module"
    python -m $module
done
