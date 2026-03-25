# Hyperparameter Tuning - Quick Start Guide

## What You Have

A complete hyperparameter tuning framework for your RNN with the following fixed settings:
- **cell_type**: `'rnn'` (basic RNN, not LSTM)
- **task**: `'next_char'` (character-level language modeling)
- **fast_choice**: `False` (no fast weights)

---

## Prerequisites

### Activate UV Virtual Environment First

The `pyproject.toml` and `uv.lock` live at the **repo root**. Run `uv sync` once to create `.venv`:

```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv sync
```

Then activate for the session:

```bash
# Option 1: Use the activation script (from ContPTNCN/)
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN
source activate_env.sh

# Option 2: Activate manually from repo root
cd /Users/hitesh/Documents/Research/ContinualPTNCN
source .venv/bin/activate

# Option 3: Skip activation entirely — use uv run
uv run python ContPTNCN/src/quick_trial.py --config ContPTNCN/configs/baseline.json
```

Verify activation:
```bash
which python
# Should show: /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/python
```

---

## Files

```
ContinualPTNCN/          (repo root)
├── pyproject.toml       # dependencies (jax, numpy, tqdm, matplotlib, scipy)
├── uv.lock              # pinned lockfile
├── .venv/               # created by `uv sync`
└── ContPTNCN/
    ├── src/
    │   ├── hyperparameter_tuning.py    # Automated search (12 trials)
    │   ├── quick_trial.py              # Single trial runner
    │   └── analyze_results.py          # Result visualization
    ├── configs/
    │   ├── baseline.json               # Safe baseline config
    │   ├── large_model.json            # Larger capacity model
    │   └── high_lr.json                # Higher learning rate
    ├── activate_env.sh                 # Activation helper
    ├── run_tuning.sh                   # Interactive menu script
    └── docs/TUNING_README.md           # Detailed documentation
```

---

## Quick Start (3 Options)

**Activate UV environment first:**
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN && source .venv/bin/activate
```

### Option 1: Interactive Menu (Easiest)
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN
./run_tuning.sh  # auto-activates .venv
```
Choose from the menu:
1. Full automated search
2. Quick baseline test
3. Large model test
4. High LR test
5. Custom interactive trial
6. Analyze results

### Option 2: Run Full Automated Search
```bash
source /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN/src
python hyperparameter_tuning.py
```

This will:
- Run 12 different configurations automatically
- Save results to `../results/hyperparameter_tuning_TIMESTAMP/`
- Generate a summary ranking all trials
- Take ~1-2 hours on GPU

### Option 3: Test Single Configuration
```bash
source /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN/src

# Using config file
python quick_trial.py --config ../configs/baseline.json

# Or with command-line args
python quick_trial.py --hidden_size 256 --learning_rate 0.005
```

---

## Hyperparameters Being Tuned

| Parameter | Purpose | Range Tested |
|-----------|---------|--------------|
| `embedding_dim` | Character embedding size | 64, 96, 128 |
| `hidden_size` | RNN hidden state size | 128, 192, 256 |
| `num_layers` | Stacked RNN layers | 1, 2 |
| `learning_rate` | SGD step size | 0.0001 - 0.01 |
| `batch_size` | Training batch size | 32, 48, 64 |
| `seq_len` | Input sequence length | 20, 25, 35 |
| `activation` | RNN activation | tanh, relu |

---

## Understanding Results

### Metrics
- **BPC (Bits Per Character)**: Lower is better
  - Random: ~5.64 BPC
  - Good: 1.3-1.8 BPC
  - Great: <1.3 BPC

### What Gets Saved
```
results/hyperparameter_tuning_20251007_120000/
├── trial_001.json    # Each trial's full results
├── trial_002.json
├── ...
├── trial_012.json
└── summary.json      # Best configuration summary
```

### Analyzing Results
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN/src
python analyze_results.py ../results/hyperparameter_tuning_TIMESTAMP/
```

This generates:
- `training_curves.png` — Loss curves for all trials
- `hyperparameter_effects.png` — Impact of each hyperparameter
- Console output with detailed statistics

---

## Example Workflow

### 1. First-time setup
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv sync
source .venv/bin/activate
which python  # Should show .venv/bin/python
```

### 2. Start with baseline to verify setup
```bash
cd ContPTNCN/src
python quick_trial.py --config ../configs/baseline.json
```
Expected: ~2.0 BPC in 5-10 minutes

### 3. Run full hyperparameter search
```bash
python hyperparameter_tuning.py
```
Expected: Best trial around 1.5-1.8 BPC

### 4. Analyze results
```bash
python analyze_results.py ../results/hyperparameter_tuning_*/
```

### 5. Fine-tune the best configuration
```bash
# Copy best config from summary.json to a new file, then:
python quick_trial.py --config ../configs/best_config.json
```

---

## Customizing Trials

### Edit existing config
```bash
nano /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN/configs/baseline.json
```

### Create new config
```json
{
  "embedding_dim": 96,
  "hidden_size": 192,
  "num_layers": 1,
  "learning_rate": 0.003,
  "batch_size": 48,
  "seq_len": 25,
  "num_epochs": 20,
  "activation": "tanh"
}
```

### Modify trial list in code
Edit `generate_trial_configurations()` in `hyperparameter_tuning.py`

---

## Troubleshooting

### `.venv` doesn't exist yet
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv sync
```

### Out of memory
- Reduce `batch_size`: 64 → 32
- Reduce `hidden_size`: 256 → 128
- Reduce `seq_len`: 35 → 20

### Training too slow
- Reduce `max_train_batches`: 500 → 200
- Reduce `num_epochs`: 20 → 10
- Use smaller model

### Loss not decreasing
- Increase `learning_rate`: 0.001 → 0.005
- Increase `hidden_size`: 128 → 256
- Increase `num_epochs`: 15 → 25

### NaN or exploding loss
- Decrease `learning_rate`: 0.01 → 0.001
- Check data loader (ensure no invalid indices)
- Verify gradient clipping is enabled (already set to 5.0)

---

## Expected Timeline

- **Quick baseline**: 5-10 minutes
- **Single trial (20 epochs)**: 10-15 minutes
- **Full search (12 trials)**: 1-2 hours
- **Deep search (custom)**: 3-4 hours

---

## Next Steps

1. **Run baseline** to verify everything works
2. **Run full search** overnight or during break
3. **Analyze results** to find best configuration
4. **Fine-tune** best config with more epochs
5. **Document** final hyperparameters for your paper/report

## Tips

- Start with small trials to verify setup
- Use GPU if available (automatic detection)
- Check intermediate results during long runs
- Save promising configurations
- Don't over-optimize on validation set

## More Info

- `docs/TUNING_README.md` — Detailed documentation
- `src/hyperparameter_tuning.py` — See trial definitions
- `src/quick_trial.py` — Understand single trial
- `src/analyze_results.py` — Result processing
- `UV_COMMANDS.md` — Full UV environment reference
