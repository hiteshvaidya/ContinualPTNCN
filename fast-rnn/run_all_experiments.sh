#!/bin/bash
# Simple experiment runner with logging for Fast Weights

# Configuration
DATASET="text8"
DATA_PATH="data/text8"
EPOCHS=10
BATCH_SIZE=64
HIDDEN_SIZE=256
GPU=0

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiment_logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
mkdir -p "checkpoints"

echo "=================================================="
echo "Fast Weights Experiment Suite"
echo "=================================================="
echo "Dataset: ${DATASET}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Hidden Size: ${HIDDEN_SIZE}"
echo "GPU: ${GPU}"
echo "Log Directory: ${LOG_DIR}"
echo "=================================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=${GPU}

# Function to run experiment
run_experiment() {
    local name=$1
    local script=$2
    shift 2
    local args=("$@")
    
    echo ""
    echo "=================================================="
    echo "Running: ${name}"
    echo "=================================================="
    echo "Command: python ${script} ${args[@]}"
    echo ""
    
    # Run and log
    python "${script}" "${args[@]}" 2>&1 | tee "${LOG_DIR}/${name}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ ${name} completed successfully"
    else
        echo "✗ ${name} failed"
    fi
}

# RNN Experiments
echo ""
echo "### RNN EXPERIMENTS ###"

# Standard RNN
run_experiment \
    "standard_rnn_${DATASET}" \
    "fast_weights.py" \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model "standard_rnn" \
    --hidden_size "${HIDDEN_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --save_path "checkpoints/standard_rnn_${DATASET}_${TIMESTAMP}.pt"

# Fast Weights RNN (S=1)
run_experiment \
    "fast_rnn_S1_${DATASET}" \
    "fast_weights.py" \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model "fast_rnn" \
    --hidden_size "${HIDDEN_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --S 1 \
    --lambda_decay 0.95 \
    --eta_lr 0.5 \
    --save_path "checkpoints/fast_rnn_S1_${DATASET}_${TIMESTAMP}.pt"

# Fast Weights RNN (S=2)
run_experiment \
    "fast_rnn_S2_${DATASET}" \
    "fast_weights.py" \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model "fast_rnn" \
    --hidden_size "${HIDDEN_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --S 2 \
    --lambda_decay 0.95 \
    --eta_lr 0.5 \
    --save_path "checkpoints/fast_rnn_S2_${DATASET}_${TIMESTAMP}.pt"

# LSTM Experiments
echo ""
echo "### LSTM EXPERIMENTS ###"

# Standard LSTM
run_experiment \
    "lstm_${DATASET}" \
    "fast_weights_lstm.py" \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model "lstm" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_layers 2 \
    --dropout 0.2 \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --save_path "checkpoints/lstm_${DATASET}_${TIMESTAMP}.pt"

# Fast Weights LSTM (S=1)
run_experiment \
    "fast_lstm_S1_${DATASET}" \
    "fast_weights_lstm.py" \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model "fast_lstm" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_layers 2 \
    --dropout 0.2 \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --S 1 \
    --lambda_decay 0.95 \
    --eta_lr 0.5 \
    --save_path "checkpoints/fast_lstm_S1_${DATASET}_${TIMESTAMP}.pt"

# Fast Weights LSTM (S=2)
run_experiment \
    "fast_lstm_S2_${DATASET}" \
    "fast_weights_lstm.py" \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model "fast_lstm" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_layers 2 \
    --dropout 0.2 \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --S 2 \
    --lambda_decay 0.95 \
    --eta_lr 0.3 \
    --save_path "checkpoints/fast_lstm_S2_${DATASET}_${TIMESTAMP}.pt"

# Generate summary
echo ""
echo "=================================================="
echo "EXPERIMENT SUMMARY"
echo "=================================================="
echo ""
echo "Extracting results from logs..."
echo ""

for log_file in "${LOG_DIR}"/*.log; do
    if [ -f "${log_file}" ]; then
        exp_name=$(basename "${log_file}" .log)
        echo "--- ${exp_name} ---"
        
        # Extract key metrics
        grep "Number of parameters:" "${log_file}" || echo "Parameters: N/A"
        grep "Train BPC:" "${log_file}" | tail -1 || echo "Train BPC: N/A"
        grep "Valid BPC:" "${log_file}" | tail -1 || echo "Valid BPC: N/A"
        grep "Test BPC:" "${log_file}" | tail -1 || echo "Test BPC: N/A"
        
        echo ""
    fi
done

echo "=================================================="
echo "All experiments completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "Checkpoints saved to: checkpoints/"
echo "=================================================="
