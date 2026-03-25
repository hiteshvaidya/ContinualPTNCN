# PTNCN PyTorch Port Notes

## Which dataset to start with

Start with `ContPTNCN/data/ptb_char`.

Why this is the right first dataset for the port:

- It already exists in the repo, so we do not block on downloading or preprocessing.
- The original TensorFlow PTNCN training script already targets this exact format.
- It lets us compare the PyTorch port against the TensorFlow baseline with minimal confounds.
- The `subX.txt` split is small enough for smoke tests, while `trainX.txt` is available when we want a longer run.

If you want an even tighter sanity check after this, add a synthetic copy task next. That is useful for debugging fast weights, but `ptb_char` is the best first end-to-end port target because it matches the existing code path.

## How the PyTorch PTNCN is structured

The new PyTorch files are:

- `ContPTNCN/src/models/ptncn_torch.py`
- `ContPTNCN/src/utils/discrete_sequence_torch.py`
- `ContPTNCN/src/train_ptncn_torch.py`

The implementation follows the TensorFlow model closely:

1. `forward_step()` keeps explicit latent states:
   - `zf0` is the current one-hot input
   - `zf1`, `zf2` are the two latent states
   - `y1`, `y2` are the local targets after error correction
   - `ex`, `e1`, `e1v`, `e2v` are local error terms

2. Fast weights are added at both latent layers:
   - `A1` and `A2` are updated with a decayed outer-product rule
   - the inner loop refines `zf1` and `zf2` for a small number of fast-weight steps

3. `compute_updates()` returns manual synaptic updates:
   - `W1`, `W2` are top-down prediction weights
   - `E1`, `E2` are error weights
   - `M1`, `M2`, `V1`, `V2`, `U1` handle temporal and top-down driving terms

4. Training uses an optimizer only as an update applicator:
   - we do not call `loss.backward()`
   - instead, we write the local update tensors into `parameter.grad`
   - then `optimizer.step()` applies momentum and learning rate

This keeps the local-learning rule explicit while still letting PyTorch handle parameter storage and optimization.

## Important implementation detail

Padded timesteps are converted to all-zero input vectors, not token `0`. This matches the TensorFlow path more closely and avoids injecting bogus symbols into masked positions.

## Smoke-test command

Run this from `ContPTNCN/src`:

```bash
../../.venv/bin/python train_ptncn_torch.py \
  --device cpu \
  --hidden-size 64 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --nesterov
```

This is only a functional smoke test. After that, scale up `hidden-size`, `max-train-batches`, and `epochs`.
