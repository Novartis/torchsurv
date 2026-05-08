#!/bin/bash
# Run code-quality checks: formatting, linting, and type checking.
# Usage: codeqc.sh [check]
#   check  Run ruff format in check-only mode (no writes). Default: format in-place.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src"

if [[ "${1-}" == "check" ]]; then
    ruff format --check .
else
    ruff format .
fi

ruff check .
mypy src/
