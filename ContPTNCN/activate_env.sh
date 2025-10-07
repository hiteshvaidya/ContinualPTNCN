#!/bin/bash
# Activate UV virtual environment for running hyperparameter tuning

VENV_PATH="/data/hvaidya/ContinualPTNCN/.venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Activated UV virtual environment"
    echo "  Python: $(which python)"
    echo "  Location: $VENV_PATH"
    echo ""
    echo "You can now run:"
    echo "  cd src && python hyperparameter_tuning.py"
    echo "  cd src && python quick_trial.py --config ../configs/baseline.json"
else
    echo "✗ UV virtual environment not found at $VENV_PATH"
    echo ""
    echo "To create it, run:"
    echo "  cd /data/hvaidya/ContinualPTNCN"
    echo "  uv venv .venv"
    echo "  source .venv/bin/activate"
    echo "  uv pip install jax jaxlib numpy tqdm matplotlib"
    exit 1
fi
