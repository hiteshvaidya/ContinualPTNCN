# UV Environment - Command Reference

## Quick Reference

The UV environment is managed via `pyproject.toml` and `uv.lock` at the **repo root**:
```
/Users/hitesh/Documents/Research/ContinualPTNCN/
```

The `.venv` is created there by running `uv sync`.

---

## First-Time Setup

```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv sync
```

This reads `pyproject.toml` + `uv.lock` and creates `.venv` with all dependencies pinned.

---

## Activation Commands

### Option A: Source the helper script (from ContPTNCN/)
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN
source activate_env.sh
```

### Option B: Activate manually (from repo root)
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
source .venv/bin/activate
```

### Option C: Use `uv run` without activating
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv run python ContPTNCN/src/quick_trial.py --config ContPTNCN/configs/baseline.json
```

### Verify activation
```bash
which python
# Should output: /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/python
python --version
# Should output: Python 3.10.x
```

---

## Running Hyperparameter Tuning

### Method 1: Interactive Menu (auto-activates)
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN
./run_tuning.sh
```

### Method 2: Full Automated Workflow (auto-activates)
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN
./run_full_workflow.sh
```

### Method 3: Manual Commands (requires activation first)
```bash
# Activate
cd /Users/hitesh/Documents/Research/ContinualPTNCN
source .venv/bin/activate

# Then run from src/
cd ContPTNCN/src
python hyperparameter_tuning.py                           # Full search
python quick_trial.py --config ../configs/baseline.json   # Single trial
python analyze_results.py ../results/hyperparameter_tuning_*  # Analyze
```

---

## Common Commands

### Run baseline test
```bash
source /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN/src
python quick_trial.py --config ../configs/baseline.json
```

### Run full hyperparameter search
```bash
source /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN/src
python hyperparameter_tuning.py
```

### Run custom trial
```bash
source /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN/src
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
source /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN/src
python analyze_results.py ../results/hyperparameter_tuning_YYYYMMDD_HHMMSS/
```

---

## Troubleshooting

### Environment not activated
**Symptom:** `ModuleNotFoundError: No module named 'jax'`

**Solution:**
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv sync          # ensures all packages are installed
source .venv/bin/activate
```

### Wrong Python version
**Symptom:** Using system Python instead of `.venv`

**Check:**
```bash
which python  # Should show /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/python
```

**Fix:**
```bash
deactivate  # Exit any other active env
cd /Users/hitesh/Documents/Research/ContinualPTNCN
source .venv/bin/activate
```

### Missing or outdated dependencies
**Symptom:** Import errors for packages

**Solution:** Re-sync the environment from the lockfile:
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv sync
```

To add a new package:
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv add <package-name>   # updates pyproject.toml + uv.lock
```

---

## File Structure

```
ContinualPTNCN/                         # Repo root — run uv sync here
├── pyproject.toml                      # Project metadata & dependencies
├── uv.lock                             # Pinned dependency lockfile
├── .python-version                     # Python 3.10
├── .venv/                              # Created by `uv sync`
│   └── bin/
│       ├── activate
│       └── python
└── ContPTNCN/                          # Project code
    ├── activate_env.sh                 # Helper to activate env
    ├── UV_COMMANDS.md                  # This file
    ├── run_tuning.sh                   # Interactive menu (auto-activates)
    ├── run_full_workflow.sh            # Complete workflow (auto-activates)
    ├── src/
    │   ├── hyperparameter_tuning.py
    │   ├── quick_trial.py
    │   └── analyze_results.py
    └── configs/
        ├── baseline.json
        ├── large_model.json
        └── high_lr.json
```

---

## Recommended Workflow

### First-Time Setup
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN

# Create .venv and install all locked dependencies
uv sync

# Verify
source .venv/bin/activate
which python    # .venv/bin/python
python --version  # Python 3.10.x
python -c "import jax; print(jax.devices())"
```

### Daily Usage

**Option A: Use automated scripts (easiest)**
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN/ContPTNCN
./run_tuning.sh  # Choose from menu
```

**Option B: Manual control**
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
source .venv/bin/activate
cd ContPTNCN/src
python quick_trial.py --config ../configs/baseline.json
```

**Option C: `uv run` (no activation needed)**
```bash
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv run python ContPTNCN/src/quick_trial.py --config ContPTNCN/configs/baseline.json
```

### When Done
```bash
deactivate  # Exit UV environment
```

---

## Scripts That Auto-Activate UV Environment

These scripts automatically activate `.venv` — no manual activation needed:

1. `./run_tuning.sh` — Interactive menu
2. `./run_full_workflow.sh` — Complete automated workflow

## Scripts Requiring Manual Activation

These require you to activate `.venv` (or use `uv run`) first:

1. `python hyperparameter_tuning.py`
2. `python quick_trial.py`
3. `python analyze_results.py`

---

## Quick Command Cheat Sheet

```bash
# First-time setup
cd /Users/hitesh/Documents/Research/ContinualPTNCN && uv sync

# Activate environment
source /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate

# Quick test (2 epochs)
cd ContPTNCN/src && python quick_trial.py --config ../configs/baseline.json --num_epochs 2

# Full search
python hyperparameter_tuning.py

# Custom trial
python quick_trial.py --hidden_size 256 --learning_rate 0.005

# Analyze latest results
python analyze_results.py $(ls -dt ../results/hyperparameter_tuning_* | head -1)

# Deactivate
deactivate
```

## Environment Variables (Optional)

To auto-activate whenever you `cd` into the project, add to `~/.zshrc`:

```zsh
# Auto-activate UV env when entering the ContinualPTNCN project
cd() {
    builtin cd "$@"
    if [[ "$PWD" == /Users/hitesh/Documents/Research/ContinualPTNCN* ]] && \
       [ -f "/Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate" ]; then
        source /Users/hitesh/Documents/Research/ContinualPTNCN/.venv/bin/activate
    fi
}
```

Then:
```bash
source ~/.zshrc
cd /Users/hitesh/Documents/Research/ContinualPTNCN  # Auto-activates!
```
