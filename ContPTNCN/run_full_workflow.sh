#!/bin/bash
# Complete hyperparameter tuning workflow with UV environment

set -e  # Exit on error

PROJECT_DIR="/data/hvaidya/ContinualPTNCN/ContPTNCN"
VENV_PATH="/data/hvaidya/ContinualPTNCN/.venv"

echo "================================================================"
echo "RNN Hyperparameter Tuning - Complete Workflow"
echo "================================================================"
echo ""

# Step 1: Activate environment
echo "Step 1: Activating UV virtual environment..."
cd "$PROJECT_DIR"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "ERROR: UV virtual environment not found at $VENV_PATH!"
    echo "Please run: cd /data/hvaidya/ContinualPTNCN && uv venv .venv"
    exit 1
fi

source "$VENV_PATH/bin/activate"
echo "✓ Environment activated"
echo "  Python: $(which python)"
echo ""

# Step 2: Verify JAX installation
echo "Step 2: Verifying JAX installation..."
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())" || {
    echo "ERROR: JAX not properly installed"
    echo "Install with: uv pip install jax jaxlib"
    exit 1
}
echo ""

# Step 3: Run baseline test
echo "Step 3: Running baseline test (2 epochs for quick verification)..."
cd src

# Create a temporary quick test config
cat > /tmp/quick_test.json << EOF
{
  "embedding_dim": 64,
  "hidden_size": 128,
  "num_layers": 1,
  "learning_rate": 0.001,
  "batch_size": 32,
  "seq_len": 20,
  "num_epochs": 2,
  "activation": "tanh",
  "seed": 42,
  "max_train_batches": 50,
  "max_valid_batches": 10,
  "data_dir": "../data/ptb_char"
}
EOF

python quick_trial.py --config /tmp/quick_test.json || {
    echo "ERROR: Baseline test failed"
    exit 1
}

echo ""
echo "✓ Baseline test completed successfully!"
echo ""

# Step 4: Ask user what to do next
echo "================================================================"
echo "What would you like to do next?"
echo "================================================================"
echo ""
echo "1) Run full hyperparameter search (12 trials, ~1-2 hours)"
echo "2) Test a single configuration"
echo "3) Exit"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Starting full hyperparameter search..."
        echo "Results will be saved to: ../results/hyperparameter_tuning_*"
        echo ""
        python hyperparameter_tuning.py
        
        # Find the latest results directory
        RESULTS_DIR=$(ls -dt ../results/hyperparameter_tuning_* 2>/dev/null | head -1)
        
        if [ -n "$RESULTS_DIR" ]; then
            echo ""
            echo "================================================================"
            echo "Generating analysis plots..."
            echo "================================================================"
            python analyze_results.py "$RESULTS_DIR"
            
            echo ""
            echo "✓ Complete! Check results in: $RESULTS_DIR"
            echo ""
            echo "View summary:"
            echo "  cat $RESULTS_DIR/summary.json"
            echo ""
            echo "View plots:"
            echo "  xdg-open $RESULTS_DIR/training_curves.png"
            echo "  xdg-open $RESULTS_DIR/hyperparameter_effects.png"
        fi
        ;;
    2)
        echo ""
        echo "Available configurations:"
        echo "  1) baseline.json"
        echo "  2) large_model.json"
        echo "  3) high_lr.json"
        echo ""
        read -p "Choose config [1-3]: " config_choice
        
        case $config_choice in
            1) CONFIG="../configs/baseline.json" ;;
            2) CONFIG="../configs/large_model.json" ;;
            3) CONFIG="../configs/high_lr.json" ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac
        
        echo ""
        echo "Running trial with $CONFIG..."
        python quick_trial.py --config "$CONFIG"
        ;;
    3)
        echo "Exiting. Environment still activated."
        echo "To deactivate, run: deactivate"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Workflow completed!"
echo "================================================================"
echo ""
echo "To deactivate environment, run: deactivate"
