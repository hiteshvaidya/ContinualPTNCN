# Hyperparameter Tuning for RNN on Penn Treebank

This directory contains scripts for hyperparameter tuning of the RNN model on character-level Penn Treebank.

## Files

- **`hyperparameter_tuning.py`**: Runs multiple trials with different configurations automatically
- **`quick_trial.py`**: Runs a single trial with custom configuration
- **`configs/`**: Directory containing example configuration files

## Usage

### Option 1: Run Automated Hyperparameter Search

This will run 12 different configurations and compare results:

```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src
python hyperparameter_tuning.py
```

**What it does:**
- Tests different combinations of:
  - Embedding dimensions: [64, 96, 128]
  - Hidden sizes: [128, 192, 256]
  - Number of layers: [1, 2]
  - Learning rates: [0.0001, 0.001, 0.002, 0.003, 0.005, 0.01]
  - Batch sizes: [32, 48, 64]
  - Sequence lengths: [20, 25, 35]
  - Activations: [tanh, relu]
- Saves results to `../results/hyperparameter_tuning_YYYYMMDD_HHMMSS/`
- Generates summary with best configuration

**Output:**
```
Rank   Trial                      Valid BPC    Test BPC     Epochs   Time (s)
--------------------------------------------------------------------------------
1      Large_Hidden               1.2345       1.2567       12       456.78
2      Medium_LR                  1.2456       1.2678       15       523.45
...
```

### Option 2: Run Single Custom Trial

#### Using configuration file:

```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src
python quick_trial.py --config ../configs/baseline.json
```

#### Using command-line arguments:

```bash
python quick_trial.py \
  --embedding_dim 128 \
  --hidden_size 256 \
  --num_layers 1 \
  --learning_rate 0.005 \
  --batch_size 64 \
  --seq_len 35 \
  --num_epochs 20 \
  --activation tanh
```

## Configuration Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `embedding_dim` | Size of character embeddings | 32-256 |
| `hidden_size` | Size of RNN hidden state | 64-512 |
| `num_layers` | Number of stacked RNN layers | 1-3 |
| `learning_rate` | SGD learning rate | 0.0001-0.01 |
| `batch_size` | Training batch size | 16-128 |
| `seq_len` | Length of input sequences | 10-50 |
| `num_epochs` | Number of training epochs | 10-50 |
| `activation` | RNN activation function | tanh, relu, sigmoid |

## Example Configurations

### Baseline (Fast Training)
```json
{
  "embedding_dim": 64,
  "hidden_size": 128,
  "num_layers": 1,
  "learning_rate": 0.001,
  "batch_size": 32,
  "seq_len": 20,
  "num_epochs": 20
}
```
**Expected:** ~2.0 BPC, ~5 min training

### Large Model (Better Performance)
```json
{
  "embedding_dim": 128,
  "hidden_size": 256,
  "num_layers": 2,
  "learning_rate": 0.003,
  "batch_size": 64,
  "seq_len": 35,
  "num_epochs": 25
}
```
**Expected:** ~1.5-1.8 BPC, ~15 min training

### High Learning Rate (Risky but Fast)
```json
{
  "embedding_dim": 64,
  "hidden_size": 128,
  "num_layers": 1,
  "learning_rate": 0.01,
  "batch_size": 32,
  "seq_len": 20,
  "num_epochs": 20
}
```
**Expected:** May diverge or give ~1.8-2.2 BPC

## Understanding Results

### Metrics

- **BPC (Bits Per Character)**: Lower is better. Measures how many bits needed to encode each character.
  - Random guessing: ~5.64 BPC (log₂(50) for 50-char vocab)
  - Good model: 1.3-1.8 BPC
  - State-of-the-art: <1.3 BPC

- **Perplexity**: Alternative metric, exponential of BPC
  - Lower is better
  - Perplexity = 2^(BPC)

### What to Look For

1. **Training doesn't diverge**: Loss shouldn't increase or become NaN
2. **Validation improves**: Valid BPC should decrease over epochs
3. **No severe overfitting**: Train BPC shouldn't be much lower than valid BPC
4. **Reasonable time**: Balance performance vs training time

## Tips for Tuning

### If loss is too high (>3.0 BPC):
- Increase `hidden_size` (128 → 256)
- Increase `num_layers` (1 → 2)
- Increase `num_epochs` (15 → 25)
- Try different `activation` (tanh → relu)

### If training is unstable:
- Decrease `learning_rate` (0.01 → 0.001)
- Decrease `seq_len` (35 → 20)
- Check gradient clipping (already set to 5.0)

### If training is too slow:
- Decrease `batch_size` (64 → 32)
- Decrease `seq_len` (35 → 20)
- Decrease `num_layers` (2 → 1)
- Reduce `max_train_batches` for faster epochs

### If overfitting (train << valid):
- Currently no regularization implemented
- Could add dropout (requires model modification)
- Reduce model capacity (`hidden_size`, `num_layers`)

## Recommended Tuning Strategy

1. **Start with baseline** to verify everything works
2. **Grid search learning rate**: [0.0001, 0.001, 0.005, 0.01]
3. **Try larger models** with best LR
4. **Optimize batch size** and sequence length
5. **Fine-tune** the best configuration

## Results Directory Structure

```
results/hyperparameter_tuning_YYYYMMDD_HHMMSS/
├── trial_001.json          # Individual trial results
├── trial_002.json
├── ...
├── trial_012.json
└── summary.json            # Overall summary and best config
```

Each trial file contains:
- Configuration used
- Training history (loss per epoch)
- Best validation loss and epoch
- Final test loss
- Total training time

## Quick Commands

```bash
# Run full hyperparameter search
python hyperparameter_tuning.py

# Test baseline configuration
python quick_trial.py --config ../configs/baseline.json

# Test large model
python quick_trial.py --config ../configs/large_model.json

# Quick test with custom params
python quick_trial.py --hidden_size 256 --learning_rate 0.005 --num_epochs 10
```

## Notes

- Fixed parameters (as per requirements):
  - `cell_type='rnn'` (basic RNN, not LSTM)
  - `task='next_char'` (language modeling)
  - `fast_choice=False` (no fast weights)
- All trials use character-level Penn Treebank data
- Results are reproducible with same `seed` value
- GPU will be used if available, otherwise CPU
