#!/bin/bash
# Activate UV virtual environment for ContinualPTNCN
#
# The .venv lives at the repo root (one level up from ContPTNCN/).
# If the environment doesn't exist yet, run `uv sync` from the repo root first.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Activated UV virtual environment"
    echo "  Python: $(which python)"
    echo "  Location: $VENV_PATH"
    echo ""
    echo "You can now run:"
    echo "  cd ContPTNCN/src && python hyperparameter_tuning.py"
    echo "  cd ContPTNCN/src && python quick_trial.py --config ../configs/baseline.json"
else
    echo "✗ UV virtual environment not found at $VENV_PATH"
    echo ""
    echo "To create it, run:"
    echo "  cd $REPO_ROOT"
    echo "  uv sync"
    echo ""
    echo "Then source this script again."
    exit 1
fi
