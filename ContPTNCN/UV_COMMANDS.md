# UV Environment - Command Reference

## Quick Reference

All commands assume you're using the UV virtual environment at `/data/hvaidya/ContinualPTNCN/.venv/`.

## Activation Commands

### One-time activation (manual)
```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
```

### Using activation helper
```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN
source activate_env.sh
```

### Verify activation
```bash
which python
# Should output: /data/hvaidya/ContinualPTNCN/.venv/bin/python
```

## Running Hyperparameter Tuning

### Method 1: Interactive Menu (Auto-activates UV env)
```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN
./run_tuning.sh
```

### Method 2: Full Automated Workflow (Auto-activates UV env)
```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN
./run_full_workflow.sh
```

### Method 3: Manual Commands (Requires manual activation)
```bash
# Activate first
cd /data/hvaidya/ContinualPTNCN/ContPTNCN
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate

# Then run commands
cd src
python hyperparameter_tuning.py                           # Full search
python quick_trial.py --config ../configs/baseline.json   # Single trial
python analyze_results.py ../results/hyperparameter_tuning_*  # Analyze
```

## Common Commands

### Run baseline test
```bash
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src
python quick_trial.py --config ../configs/baseline.json
```

### Run full hyperparameter search
```bash
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src
python hyperparameter_tuning.py
```

### Run custom trial
```bash
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src
python quick_trial.py \
    --embedding_dim 128 \
    --hidden_size 256 \
    --num_layers 1 \
    --learning_rate 0.003 \
    --batch_size 64 \
    --seq_len 35 \
    --num_epochs 20
```

### Analyze results
```bash
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src
python analyze_results.py ../results/hyperparameter_tuning_YYYYMMDD_HHMMSS/
```

## Troubleshooting

### Environment not activated
**Symptom:** `ModuleNotFoundError: No module named 'jax'`

**Solution:**
```bash
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
```

### Wrong Python version
**Symptom:** Using system Python instead of UV env

**Check:**
```bash
which python  # Should show /data/hvaidya/ContinualPTNCN/.venv/bin/python
```

**Fix:**
```bash
deactivate  # If in another env
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
```

### Missing dependencies
**Symptom:** Import errors for packages

**Solution:**
```bash
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
uv pip install jax jaxlib numpy tqdm matplotlib
```

## File Structure

```
ContinualPTNCN/
├── .venv/                          # UV virtual environment (at repo root)
│   └── bin/
│       ├── activate                # Activation script
│       └── python                  # Python interpreter
└── ContPTNCN/
    ├── activate_env.sh             # Helper to activate env
    ├── run_tuning.sh               # Interactive menu (auto-activates)
    ├── run_full_workflow.sh        # Complete workflow (auto-activates)
    ├── src/
    │   ├── hyperparameter_tuning.py    # Need .venv activated
    │   ├── quick_trial.py              # Need .venv activated
    │   └── analyze_results.py          # Need .venv activated
    └── configs/
        ├── baseline.json
        ├── large_model.json
        └── high_lr.json
```

## Recommended Workflow

### First Time Setup
```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN

# Verify UV env exists
ls /data/hvaidya/ContinualPTNCN/.venv/bin/activate

# Test activation
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
which python
python --version

# Verify JAX
python -c "import jax; print(jax.devices())"
```

### Daily Usage

**Option A: Use automated scripts (easiest)**
```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN
./run_tuning.sh  # Choose from menu
```

**Option B: Manual control**
```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
cd src
python quick_trial.py --config ../configs/baseline.json
```

### When Done
```bash
deactivate  # Exit UV environment
```

## Scripts That Auto-Activate UV Environment

These scripts automatically activate `.venv` - no manual activation needed:

1. `./run_tuning.sh` - Interactive menu
2. `./run_full_workflow.sh` - Complete automated workflow

## Scripts Requiring Manual Activation

These require you to activate `.venv` first:

1. `python hyperparameter_tuning.py`
2. `python quick_trial.py`
3. `python analyze_results.py`

## Quick Command Cheat Sheet

```bash
# Activate environment
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate

# Quick test (2 epochs)
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src && python quick_trial.py --config ../configs/baseline.json --num_epochs 2

# Full search
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src && python hyperparameter_tuning.py

# Custom trial
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src && python quick_trial.py --hidden_size 256 --learning_rate 0.005

# Analyze latest results
cd /data/hvaidya/ContinualPTNCN/ContPTNCN/src && python analyze_results.py $(ls -dt ../results/hyperparameter_tuning_* | head -1)

# Deactivate
deactivate
```

## Environment Variables (Optional)

To always use UV environment, add to `~/.bashrc`:

```bash
# Auto-activate UV env when entering project directory
cd() {
    builtin cd "$@"
    if [[ "$PWD" == /data/hvaidya/ContinualPTNCN* ]] && [ -f "/data/hvaidya/ContinualPTNCN/.venv/bin/activate" ]; then
        source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
    fi
}
```

Then:
```bash
source ~/.bashrc
cd /data/hvaidya/ContinualPTNCN/ContPTNCN  # Auto-activates!
```
