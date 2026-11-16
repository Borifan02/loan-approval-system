#!/usr/bin/env bash
# Automated test runner with coverage reporting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Ensure Matplotlib cache is writable
export MPLCONFIGDIR="/tmp/matplotlib"

echo "======================================================================"
echo " Running Automated Pytest Suite with Coverage"
echo "======================================================================"

if [ -f ".venv/bin/pytest" ]; then
    PYTEST_BIN=".venv/bin/pytest"
else
    PYTEST_BIN="pytest"
fi

"$PYTEST_BIN" -v --cov=src/loan_approval --cov-report=term-missing tests/

echo "======================================================================"
echo " All tests executed successfully!"
echo "======================================================================"
