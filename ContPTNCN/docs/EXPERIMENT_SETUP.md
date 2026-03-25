# Parallel Hyperparameter Tuning Experiment Setup

## Overview
This setup runs two parallel hyperparameter tuning experiments comparing vanilla RNN and Fast Weights RNN on the Penn Treebank character-level language modeling task.

## Experiment Configuration

### Experiment 1: Vanilla RNN
- **Config File**: `configs/vanilla_rnn_tuning.json`
- **GPU**: GPU 0 (CUDA_VISIBLE_DEVICES=0)
- **Epochs**: 50
- **Trials**: 12
- **Fast Weights**: Disabled (fast_choice=False, S=0)
- **Output**: `../results/vanilla_rnn_YYYYMMDD_HHMMSS/`

**Search Space**:
- embedding_dim: [64, 128, 256]
- hidden_size: [128, 256, 512]
- num_layers: [1, 2]
- learning_rate: [0.001, 0.005, 0.01]
- batch_size: [32, 64]
- seq_len: [10, 20, 35]

### Experiment 2: Fast Weights RNN
- **Config File**: `configs/fast_weights_tuning.json`
- **GPU**: GPU 1 (CUDA_VISIBLE_DEVICES=1)
- **Epochs**: 50
- **Trials**: 12
- **Fast Weights**: Enabled (fast_choice=True)
- **S values**: [2, 5] (number of fast weights simulation steps)
- **Output**: `../results/fast_weights_YYYYMMDD_HHMMSS/`

**Search Space**: Same as Vanilla RNN + S parameter

## Running the Experiments

### Quick Start
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN
./run_parallel_tuning.sh
```

Or:
```bash
bash run_parallel_tuning.sh
```

### Manual Execution
If you want to run experiments separately:

```bash
# Terminal 1 - Vanilla RNN (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python src/hyperparameter_tuning.py --config configs/vanilla_rnn_tuning.json

# Terminal 2 - Fast Weights RNN (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python src/hyperparameter_tuning.py --config configs/fast_weights_tuning.json
```

> `uv run` automatically uses the `.venv` at the repo root without manual activation.
> To activate manually first: `cd /Users/hitesh/Documents/Research/ContinualPTNCN && uv sync && source .venv/bin/activate`

## Monitoring Progress

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Logs
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN

# Vanilla RNN logs
tail -f ../results/vanilla_rnn_*/logs/*.log

# Fast Weights RNN logs
tail -f ../results/fast_weights_*/logs/*.log
```

### Check Output Files
The script outputs progress to:
- `vanilla_rnn_output.log`
- `fast_weights_output.log`

```bash
tail -f vanilla_rnn_output.log fast_weights_output.log
```

## Expected Results

Each experiment will generate:
- **summary.json**: Overall statistics and best trial info
- **trial_XXX.json**: Individual trial results with hyperparameters and metrics
- **training_curves.png**: Loss and accuracy curves across trials
- **hyperparameter_effects.png**: Visualization of hyperparameter impact

## GPU Requirements
- **Total GPUs Used**: 2 out of 4 available
- **GPU Memory**: Each experiment uses ~20-30GB depending on model size
- **Available GPUs**: 4 x NVIDIA GPUs with 46GB memory each

## Estimated Runtime
- **Per Trial**: ~5-15 minutes (depends on hyperparameters)
- **Total per Experiment**: ~1-3 hours for 12 trials with 50 epochs each
- **Parallel Execution**: Both experiments complete in ~1-3 hours

## Comparing Results

After completion, compare the best models:

```python
import json

# Load results
with open('../results/vanilla_rnn_*/summary.json') as f:
    vanilla_results = json.load(f)

with open('../results/fast_weights_*/summary.json') as f:
    fast_weights_results = json.load(f)

# Compare best validation perplexity
print(f"Vanilla RNN Best Val Perplexity: {vanilla_results['best_trial']['val_perplexity']}")
print(f"Fast Weights Best Val Perplexity: {fast_weights_results['best_trial']['val_perplexity']}")

# Compare best bits per character
print(f"Vanilla RNN Best BPC: {vanilla_results['best_trial']['val_bpc']}")
print(f"Fast Weights Best BPC: {fast_weights_results['best_trial']['val_bpc']}")
```

## Troubleshooting

### Out of Memory Errors
Reduce batch_size or hidden_size in config files

### CUDA Out of Memory
Check GPU allocation:
```bash
nvidia-smi
```

Kill other processes if needed

### Training Divergence
- Check learning rate (may need to reduce)
- Check gradient clipping in train_ptb.py
- Verify fast weights decay parameter (λ)

## Fast Weights Implementation Details

The fast weights mechanism implements:
```
A(t) = λA(t-1) + ηh(t)h(t)^T
```

Where:
- A(t) is the fast weights matrix
- λ is the decay rate (default: 0.95)
- η is the learning rate for fast weights (default: 0.5)
- S is the number of inner-loop simulation steps

For more details, see `src/models/rnn.py` RNNCell.fast_forward()
