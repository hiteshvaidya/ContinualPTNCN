#!/bin/bash
# Parallel hyperparameter tuning for Vanilla RNN and Fast Weights RNN
# Uses 2 GPUs to run experiments in parallel

set -e

PROJECT_DIR="/data/hvaidya/ContinualPTNCN/ContPTNCN"
VENV_PATH="/data/hvaidya/ContinualPTNCN/.venv"
PYTHON_BIN="${VENV_PATH}/bin/python"
RESULTS_DIR="${PROJECT_DIR}/results"

echo "================================================================"
echo "Parallel Hyperparameter Tuning Experiments"
echo "================================================================"
echo ""
echo "Experiment 1: Vanilla RNN (50 epochs, GPU 0)"
echo "Experiment 2: Fast Weights RNN (50 epochs, GPU 1)"
echo ""
echo "================================================================"

# Verify UV Python environment
cd "$PROJECT_DIR"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: UV Python not found at $PYTHON_BIN!"
    exit 1
fi

echo "✓ Using Python: $PYTHON_BIN"
echo ""

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Change to src directory
cd src

# Function to run hyperparameter tuning
run_tuning() {
    local CONFIG_FILE=$1
    local EXPERIMENT_NAME=$2
    local GPU_ID=$3
    local LOG_FILE=$4
    
    echo "Starting $EXPERIMENT_NAME on GPU $GPU_ID..."
    echo "Log file: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" hyperparameter_tuning.py \
        --config "$CONFIG_FILE" \
        > "$LOG_FILE" 2>&1
    
    echo "✓ $EXPERIMENT_NAME completed!"
}

# Generate timestamped log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VANILLA_LOG="${RESULTS_DIR}/vanilla_rnn_${TIMESTAMP}.log"
FAST_LOG="${RESULTS_DIR}/fast_weights_${TIMESTAMP}.log"

echo "Starting parallel experiments..."
echo ""

# Run both experiments in parallel
run_tuning "../configs/vanilla_rnn_tuning.json" "Vanilla RNN" 0 "$VANILLA_LOG" &
PID_VANILLA=$!

run_tuning "../configs/fast_weights_tuning.json" "Fast Weights RNN" 1 "$FAST_LOG" &
PID_FAST=$!

# Monitor progress
echo "Both experiments launched!"
echo ""
echo "  Vanilla RNN (GPU 0)    - PID: $PID_VANILLA"
echo "  Fast Weights RNN (GPU 1) - PID: $PID_FAST"
echo ""
echo "Monitor progress with:"
echo "  tail -f $VANILLA_LOG"
echo "  tail -f $FAST_LOG"
echo ""
echo "Waiting for experiments to complete..."

# Wait for both to finish
wait $PID_VANILLA
VANILLA_STATUS=$?

wait $PID_FAST
FAST_STATUS=$?

echo ""
echo "================================================================"
echo "Experiments Completed!"
echo "================================================================"
echo ""

if [ $VANILLA_STATUS -eq 0 ]; then
    echo "✓ Vanilla RNN experiment succeeded"
    
    # Find the latest vanilla results directory
    VANILLA_RESULTS=$(ls -dt ${RESULTS_DIR}/vanilla_rnn_* 2>/dev/null | grep -v ".log" | head -1)
    if [ -n "$VANILLA_RESULTS" ]; then
        echo "  Results: $VANILLA_RESULTS"
        echo "  Analyzing results..."
        cd src
        "$PYTHON_BIN" analyze_results.py "$VANILLA_RESULTS" 2>/dev/null || echo "  (Analysis plots will be generated separately)"
        cd ..
    fi
else
    echo "✗ Vanilla RNN experiment failed (exit code: $VANILLA_STATUS)"
    echo "  Check log: $VANILLA_LOG"
fi

echo ""

if [ $FAST_STATUS -eq 0 ]; then
    echo "✓ Fast Weights RNN experiment succeeded"
    
    # Find the latest fast weights results directory
    FAST_RESULTS=$(ls -dt ${RESULTS_DIR}/fast_weights_* 2>/dev/null | grep -v ".log" | head -1)
    if [ -n "$FAST_RESULTS" ]; then
        echo "  Results: $FAST_RESULTS"
        echo "  Analyzing results..."
        cd src
        "$PYTHON_BIN" analyze_results.py "$FAST_RESULTS" 2>/dev/null || echo "  (Analysis plots will be generated separately)"
        cd ..
    fi
else
    echo "✗ Fast Weights RNN experiment failed (exit code: $FAST_STATUS)"
    echo "  Check log: $FAST_LOG"
fi

echo ""
echo "================================================================"
echo "Summary"
echo "================================================================"
echo ""
echo "Logs:"
echo "  Vanilla RNN:     $VANILLA_LOG"
echo "  Fast Weights:    $FAST_LOG"
echo ""

if [ $VANILLA_STATUS -eq 0 ] && [ -n "$VANILLA_RESULTS" ]; then
    echo "Vanilla RNN Results:"
    echo "  Directory:       $VANILLA_RESULTS"
    echo "  Summary:         cat $VANILLA_RESULTS/summary.json"
fi

echo ""

if [ $FAST_STATUS -eq 0 ] && [ -n "$FAST_RESULTS" ]; then
    echo "Fast Weights Results:"
    echo "  Directory:       $FAST_RESULTS"
    echo "  Summary:         cat $FAST_RESULTS/summary.json"
fi

echo ""
echo "To deactivate environment: deactivate"
