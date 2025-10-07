#!/bin/bash
# Convenience script for running hyperparameter tuning trials

# Activate UV virtual environment (at repository root)
VENV_PATH="/data/hvaidya/ContinualPTNCN/.venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated UV virtual environment at $VENV_PATH"
else
    echo "Warning: UV virtual environment not found at $VENV_PATH"
    echo "Please run: cd /data/hvaidya/ContinualPTNCN && uv venv .venv"
fi

SCRIPT_DIR="$(dirname "$0")"

cd "$SCRIPT_DIR/src"

echo "========================================="
echo "RNN Hyperparameter Tuning Helper"
echo "========================================="
echo ""
echo "Choose an option:"
echo "  1) Run full hyperparameter search (12 trials)"
echo "  2) Run quick baseline test"
echo "  3) Run large model test"
echo "  4) Run high learning rate test"
echo "  5) Run custom trial (interactive)"
echo "  6) Analyze existing results"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "Running full hyperparameter search..."
        echo "This will take approximately 1-2 hours"
        echo ""
        python hyperparameter_tuning.py
        ;;
    2)
        echo ""
        echo "Running baseline configuration..."
        python quick_trial.py --config ../configs/baseline.json
        ;;
    3)
        echo ""
        echo "Running large model configuration..."
        python quick_trial.py --config ../configs/large_model.json
        ;;
    4)
        echo ""
        echo "Running high learning rate configuration..."
        python quick_trial.py --config ../configs/high_lr.json
        ;;
    5)
        echo ""
        echo "Custom trial configuration:"
        read -p "Embedding dimension [64]: " emb_dim
        emb_dim=${emb_dim:-64}
        
        read -p "Hidden size [128]: " hidden
        hidden=${hidden:-128}
        
        read -p "Number of layers [1]: " layers
        layers=${layers:-1}
        
        read -p "Learning rate [0.001]: " lr
        lr=${lr:-0.001}
        
        read -p "Batch size [32]: " batch
        batch=${batch:-32}
        
        read -p "Sequence length [20]: " seqlen
        seqlen=${seqlen:-20}
        
        read -p "Number of epochs [15]: " epochs
        epochs=${epochs:-15}
        
        read -p "Activation (tanh/relu) [tanh]: " activation
        activation=${activation:-tanh}
        
        echo ""
        echo "Running custom trial with:"
        echo "  Embedding: $emb_dim, Hidden: $hidden, Layers: $layers"
        echo "  LR: $lr, Batch: $batch, SeqLen: $seqlen"
        echo "  Epochs: $epochs, Activation: $activation"
        echo ""
        
        python quick_trial.py \
            --embedding_dim $emb_dim \
            --hidden_size $hidden \
            --num_layers $layers \
            --learning_rate $lr \
            --batch_size $batch \
            --seq_len $seqlen \
            --num_epochs $epochs \
            --activation $activation
        ;;
    6)
        echo ""
        echo "Available results directories:"
        ls -1dt ../results/hyperparameter_tuning_* 2>/dev/null | head -5
        echo ""
        read -p "Enter results directory path: " results_dir
        
        if [ -d "$results_dir" ]; then
            python analyze_results.py "$results_dir"
        else
            echo "Error: Directory not found"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Done!"
