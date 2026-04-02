# Experiment Logging and Analysis

Scripts for running and analyzing Fast Weights experiments with comprehensive logging.

## Scripts

### 1. `run_experiments.py` - Python Experiment Runner

Automated experiment runner with structured JSON logging.

**Features:**
- Runs multiple experiments sequentially
- Captures all output and metrics
- Saves results to timestamped JSON files
- Generates summary report
- Handles timeouts and errors gracefully

**Usage:**

```bash
# Run all models on text8
python test/run_experiments.py --dataset text8 --epochs 10

# Run specific models
python test/run_experiments.py --models standard_rnn fast_rnn --epochs 20

# Quick test run (3 epochs)
python test/run_experiments.py --quick

# Use specific GPU
python test/run_experiments.py --gpu 1

# Penn Treebank dataset
python test/run_experiments.py \
    --dataset ptb \
    --data_path ContPTNCN/data/ptb_char \
    --epochs 50
```

**Arguments:**
- `--dataset`: Dataset to use (`text8` or `ptb`)
- `--data_path`: Path to dataset files
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--hidden_size`: Hidden layer size
- `--gpu`: GPU device ID
- `--quick`: Run with reduced epochs for testing
- `--models`: Specific models to run (default: all)

**Output:**
- JSON log file: `experiment_logs/experiments_YYYYMMDD_HHMMSS.json`
- Structured experiment data with all configurations and results

### 2. `run_all_experiments.sh` - Bash Experiment Runner

Simple bash script for running all experiments with text file logs.

**Features:**
- Runs all model variants (RNN and LSTM)
- Saves output to individual log files
- Generates summary from logs
- Easy to customize and extend

**Usage:**

```bash
# Edit configuration at top of script
# DATASET="text8"
# EPOCHS=10
# BATCH_SIZE=64
# etc.

# Run all experiments
./test/run_all_experiments.sh
```

**Output:**
- Log files: `experiment_logs/YYYYMMDD_HHMMSS/model_name.log`
- One log file per experiment with full output

### 3. `analyze_results.py` - Results Analysis

Analyzes experiment logs and generates comparison reports.

**Features:**
- Parses log files to extract metrics
- Compares model performance
- Calculates improvements from baseline
- Identifies best models
- Generates JSON summary

**Usage:**

```bash
# Analyze latest experiment
python test/analyze_results.py

# Analyze specific log directory
python test/analyze_results.py --log_dir experiment_logs/20251030_123456
```

**Output:**
```
==================================================================================================
EXPERIMENT RESULTS SUMMARY
==================================================================================================

### RNN MODELS ###

Model                          Params          Best Valid BPC       Test BPC       
--------------------------------------------------------------------------------------------------
standard_rnn_text8             106,795         1.5234               1.5421         
fast_rnn_S1_text8              106,795         1.4567               1.4723         
fast_rnn_S2_text8              106,795         1.4123               1.4301         

### LSTM MODELS ###

Model                          Params          Best Valid BPC       Test BPC       
--------------------------------------------------------------------------------------------------
lstm_text8                     534,811         1.4321               1.4512         
fast_lstm_S1_text8             534,811         1.3456               1.3621         
fast_lstm_S2_text8             534,811         1.3012               1.3187         

==================================================================================================
COMPARISON
==================================================================================================

🏆 Best Model (by Valid BPC): fast_lstm_S2_text8
   Valid BPC: 1.3012
   Test BPC: 1.3187

### Fast Weights Improvement ###

RNN Models:
  Baseline (Standard RNN): 1.5234 BPC
  fast_rnn_S1_text8: 1.4567 BPC (Δ -0.0667, -4.38%)
  fast_rnn_S2_text8: 1.4123 BPC (Δ -0.1111, -7.29%)

LSTM Models:
  Baseline (Standard LSTM): 1.4321 BPC
  fast_lstm_S1_text8: 1.3456 BPC (Δ -0.0865, -6.04%)
  fast_lstm_S2_text8: 1.3012 BPC (Δ -0.1309, -9.14%)
```

## Experiment Workflow

### Complete Experiment Suite

```bash
# 1. Run all experiments
python test/run_experiments.py \
    --dataset text8 \
    --data_path data/text8 \
    --epochs 20 \
    --batch_size 64 \
    --hidden_size 256 \
    --gpu 0

# 2. Analyze results
python test/analyze_results.py

# 3. View detailed logs
ls experiment_logs/*/
cat experiment_logs/20251030_*/fast_lstm_S2_text8.log
```

### Quick Test

```bash
# Quick validation run (3 epochs)
python test/run_experiments.py --quick --models fast_lstm

# Analyze
python test/analyze_results.py
```

### Custom Experiments

Edit `run_experiments.py` to add custom configurations:

```python
# Add custom experiment
run_experiment(
    "fast_weights_lstm.py",
    {
        **base_config,
        'model': 'fast_lstm',
        'S': 3,  # More inner loop iterations
        'lambda_decay': 0.98,  # Higher decay
        'eta_lr': 0.2,  # Lower learning rate
        'hidden_size': 512,  # Larger model
        'save_path': f'checkpoints/custom_experiment_{logger.timestamp}.pt'
    },
    "Custom Fast LSTM",
    logger
)
```

## Directory Structure

```
test/
├── fast_weights.py              # RNN implementation
├── fast_weights_lstm.py         # LSTM implementation
├── run_experiments.py           # Python experiment runner
├── run_all_experiments.sh       # Bash experiment runner
├── analyze_results.py           # Results analysis tool
├── experiment_logs/             # Experiment logs directory
│   └── YYYYMMDD_HHMMSS/        # Timestamped experiment session
│       ├── experiments.json     # Structured experiment data (Python runner)
│       ├── summary.json         # Analysis summary
│       ├── standard_rnn.log     # Individual experiment logs (Bash runner)
│       ├── fast_rnn_S1.log
│       └── ...
└── checkpoints/                 # Saved model checkpoints
    ├── standard_rnn_text8_TIMESTAMP.pt
    ├── fast_lstm_S2_text8_TIMESTAMP.pt
    └── ...
```

## Tips

### Running Long Experiments

Use `tmux` or `screen` to run experiments in background:

```bash
# Start tmux session
tmux new -s experiments

# Run experiments
python test/run_experiments.py --dataset text8 --epochs 50

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t experiments
```

### Monitoring Progress

```bash
# Watch log file in real-time
tail -f experiment_logs/20251030_*/fast_lstm_S2_text8.log

# Check progress
watch -n 5 'tail -20 experiment_logs/20251030_*/fast_lstm_S2_text8.log'
```

### Comparing Multiple Runs

```bash
# Run experiments on different dates
python test/run_experiments.py --dataset text8 --epochs 10
# ... next day ...
python test/run_experiments.py --dataset text8 --epochs 20

# Compare both runs
python test/analyze_results.py --log_dir experiment_logs/20251030_120000
python test/analyze_results.py --log_dir experiment_logs/20251031_140000
```

## Example Session

```bash
# Navigate to test directory
cd /data/hvaidya/ContinualPTNCN

# Activate environment
source .venv/bin/activate

# Run quick test
python test/run_experiments.py --quick --gpu 1

# Expected output:
# ==================================================
# FAST WEIGHTS EXPERIMENT SUITE
# ==================================================
# Dataset: text8
# Epochs: 3
# ...
# Running: Standard RNN - text8
# ...
# ✓ Standard RNN - text8 completed successfully
#   Best Valid BPC: 2.1234
#   Test BPC: 2.1456
# ...

# Analyze results
python test/analyze_results.py

# View best model checkpoint
ls -lh checkpoints/fast_lstm_S2_*
```

## Metrics Tracked

For each experiment:
- **Number of parameters**: Model size
- **Train BPC**: Bits per character on training set (per epoch)
- **Valid BPC**: Bits per character on validation set (per epoch)
- **Test BPC**: Final bits per character on test set
- **Best Valid BPC**: Lowest validation BPC across all epochs
- **Training time**: Duration of training
- **Checkpoint path**: Saved model location

## Troubleshooting

### No experiments found
```bash
# Make sure you ran experiments first
python test/run_experiments.py --quick

# Then analyze
python test/analyze_results.py
```

### Permission denied
```bash
chmod +x test/*.py test/*.sh
```

### Out of memory
```bash
# Reduce batch size
python test/run_experiments.py --batch_size 32

# Or reduce hidden size
python test/run_experiments.py --hidden_size 128
```

### Experiments taking too long
```bash
# Use --quick flag for fast testing
python test/run_experiments.py --quick

# Or specify fewer models
python test/run_experiments.py --models fast_lstm --epochs 5
```
