#!/bin/bash
# Build and optionally preview the documentation locally.
# Usage: build-docs.sh [serve]
#   serve  Start a local HTTP server on http://127.0.0.1:8000 after building.

set -euo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../docs" && pwd)"

cd "${DOCS_DIR}"
make clean
make html

if [[ "${1-}" == "serve" ]]; then
    cd _build/html
    python -m http.server --bind 127.0.0.1 8000
fi
