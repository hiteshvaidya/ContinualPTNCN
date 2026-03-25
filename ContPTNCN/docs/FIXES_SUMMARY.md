# Experiment Fixes Summary

## Issues Fixed

### 1. Fast Weights NaN Loss Issue
**Problem**: Training was producing `nan` BPC values due to numerical instability in fast weights implementation.

**Root Causes**:
- `self.A` matrix was being modified inside JIT-compiled functions (violates JAX purity)
- Wrong dimensions for fast weights matrix: `(batch_size, batch_size)` instead of `(hidden_size, hidden_size)`
- Unbounded accumulation in the fast weights loop causing gradient explosion
- Missing bias term in fast forward computation

**Fixes Applied**:
- Removed all instance variable modifications (`self.A`, `self.t`, `self.hidden_states`)
- Changed fast weights matrix `A` to be part of `params` dict with correct dimensions
- Added proper activation function and normalization in fast weights loop
- Made fast weights optional via `use_fast_weights` parameter
- Updated `RNN.init_params()` to accept `use_fast_weights` flag
- Modified `train_ptb.py` to pass the flag during initialization

### 2. Hyperparameter Tuning TypeError
**Problem**: `TypeError: '>=' not supported between instances of 'int' and 'NoneType'`

**Root Cause**:
- `max_train_batches` was set to `null` in config files
- Code tried to compare `num_batches >= None`

**Fixes Applied**:
- Updated comparison logic to check `if max_train_batches is not None` before comparison
- Changed `max_train_batches` from `null` to `200` in both config files

### 3. Directory Navigation Issue in run_parallel_tuning.sh
**Problem**: `cd: src: No such file or directory` when trying to run analysis

**Root Cause**:
- Script used relative path `cd src` instead of absolute path

**Fix Applied**:
- Changed to `cd "$PROJECT_DIR/src"` and `cd "$PROJECT_DIR"` for proper navigation

## Files Modified

1. **ContPTNCN/src/models/rnn.py**
   - `RNNCell.__init__`: Removed instance variables
   - `RNNCell.init_params`: Added `use_fast_weights` parameter
   - `RNNCell.fast_forward`: Completely rewritten for JAX compatibility
   - `RNNCell.init_hidden`: Removed `self.A` initialization
   - `RNN.init_params`: Added `use_fast_weights` flag propagation
   - `RNN.forward_step`: Removed `cell._increment_time()` call
   - `RNN.forward_sequence`: Fixed state propagation bug

2. **ContPTNCN/src/train_ptb.py**
   - Updated params initialization to use `use_fast_weights` flag
   - Added logging for fast weights configuration

3. **ContPTNCN/src/hyperparameter_tuning.py**
   - Fixed `max_train_batches` comparison to handle `None` values

4. **ContPTNCN/configs/vanilla_rnn_tuning.json**
   - Changed `max_train_batches` from `null` to `200`

5. **ContPTNCN/configs/fast_weights_tuning.json**
   - Changed `max_train_batches` from `null` to `200`

6. **ContPTNCN/run_parallel_tuning.sh**
   - Fixed directory navigation with absolute paths

## How to Run Experiments

### Quick Start
```bash
cd /data/hvaidya/ContinualPTNCN/ContPTNCN
./run_parallel_tuning.sh
```

### Monitor Progress
```bash
# Use the monitoring script
./monitor_experiments.sh

# Or manually follow logs
tail -f results/vanilla_rnn_*.log
tail -f results/fast_weights_*.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Expected Results

Each experiment will:
- Run 12 trials with different hyperparameter combinations
- Train for 50 epochs per trial
- Use 200 batches per epoch (for speed)
- Save results to `results/vanilla_rnn_50epochs_*` and `results/fast_weights_50epochs_*`
- Generate:
  - `summary.json`: Best trial and overall statistics
  - `trial_XXX.json`: Individual trial results
  - Training curves and hyperparameter analysis plots

## Validation

Training is working correctly when you see:
- ✅ BPC values between 4-7 (reasonable for character-level LM)
- ✅ BPC improving over epochs
- ✅ No `nan` values in loss
- ✅ Generated text (initially gibberish, improving with training)

## Current Status

- ✅ Fast weights implementation fixed for JAX compatibility
- ✅ Training produces valid BPC values
- ✅ Hyperparameter tuning configs updated
- ✅ Parallel execution script working
- ✅ Ready to run full 50-epoch experiments

## Notes

- Experiments use GPU 0 for Vanilla RNN and GPU 1 for Fast Weights RNN
- Each trial takes approximately 5-15 minutes depending on hyperparameters
- Total experiment time: ~1-3 hours for 12 trials × 50 epochs
- Results will show if fast weights improve performance over vanilla RNN
