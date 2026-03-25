# Research Program: rec-LRA & Fast Weights on P-TNCN (JAX)

## Project Overview

This research program targets the **Parallel Temporal Neural Coding Network (P-TNCN)**
reimplemented in JAX, augmenting it with two independent but potentially complementary
research threads:

1. **rec-LRA on P-TNCN**: Apply recursive Local Representation Alignment to the
   P-TNCN's weight matrices, specifically exploring low-rank factorizations of the
   error synapse matrices (E) and the forward synapse matrices (W, R).

2. **Fast Weights integration into P-TNCN via backpropagation**: Explore whether
   Hebbian-style fast weight buffers can augment the P-TNCN's temporal memory
   without breaking the local learning rule.

**Codebase**: JAX implementation from scratch. Baseline is `train_discrete_ptncn.py`
in the original TensorFlow implementation (for reference).

**Primary metric**: Bits-per-character (BPC) on Penn Treebank (PTB) validation set.
Lower is better. Baseline target: BPC < 1.45 (original PTNCN result).

**GPU**: NVIDIA L40 (48GB VRAM). Training budget per experiment: 10 minutes wall clock.

---

## Architecture Specification (P-TNCN)

The P-TNCN is a 3-layer locally recurrent predictive coding network:

- **State equations** (per layer ℓ, time step t):
  ```
  z^ℓ_t = φ(W^ℓ · z^{ℓ-1}_t + R^ℓ · z^ℓ_{t-1})   # forward + recurrent
  z^ℓ_{t,o} = φ(W^{ℓ-1}_o · z^ℓ_t)                 # prediction
  e^{ℓ-1} = z^{ℓ-1}_t - z^{ℓ-1}_{t,o}              # error unit
  y^ℓ_{t,z} = z^ℓ_t + E^ℓ · e^{ℓ-1}               # corrected state (K steps)
  ```

- **Weight update rules** (LRA, locally Hebbian):
  ```
  ΔW^ℓ = (y^ℓ - z^ℓ) · (z^{ℓ-1})^T
  ΔR^ℓ = (y^ℓ - z^ℓ) · (z^ℓ_{t-1})^T
  ΔE^ℓ = -β · (z^ℓ · (e^{ℓ-1})^T)    # error synapse update
  ```

- **Key hyperparameters** (start from these baselines):
  - `n_layers = 3`, `hidden_dim = 512`
  - `K = 20` (inference steps per time step)
  - `lr_W = 1e-3`, `lr_E = 1e-4` (error synapses update slower)
  - `beta = 0.1` (error synapse modulation)
  - `weight_decay_W = 1e-4`, `weight_decay_E = 1e-5`

---

## Research Thread 1: rec-LRA Weight Factorization

### Hypothesis
The error synapse matrices E^ℓ (shape: hidden_dim × hidden_dim) are low-rank in
practice — they serve as global "routing" matrices for error signals, analogous to
how LoRA factors work in transformers. Decomposing E^ℓ ≈ A^ℓ · B^ℓ (rank r ≪ d)
should reduce parameters without degrading BPC, and may improve generalization.

### Agent Search Space (modify these only — do not touch JAX infrastructure)

**Phase 1 — Rank search for E matrices (freeze W, R):**
Vary `rank_E` in {4, 8, 16, 32, 64, 128} while keeping `hidden_dim=512`.
Log: `val_bpc`, `n_params_E`, `E_rank_effective` (computed via SVD).

**Phase 2 — Rank search for W matrices:**
Given best `rank_E` from Phase 1, vary `rank_W` in {32, 64, 128, 256}.
Log: `val_bpc`, `n_params_total`, training stability (std of last 100 BPC values).

**Phase 3 — Joint factorization + update rule:**
Test whether the local Hebbian update decomposes cleanly:
  Option A: update A and B independently (ΔA = err · B^T, ΔB = A^T · err)
  Option B: update only A, freeze B per 10 steps
  Option C: singular value regularization on A·B

**Ablations to run automatically:**
- With/without weight decay on A, B separately
- Initialization: random Gaussian vs. SVD of pretrained E
- `K` ∈ {10, 20, 50} for each best configuration

---

## Research Thread 2: Fast Weights in P-TNCN

### Hypothesis
Fast weights (Ba et al. 2016) create a rapidly-decaying Hebbian memory:
  `A_t = λ · A_{t-1} + η · h_t · h_t^T`
  `h̃_t = LayerNorm(h_t + A_t · h_t)`

Integrating fast weights into the P-TNCN's state dynamics (as an additive
correction to `z^ℓ_t`) should improve long-range dependency capture without
requiring BPTT. Crucially, this thread uses standard backpropagation (JAX autograd)
to train W, R, while keeping the E matrices local.

### Agent Search Space

**Phase 1 — Placement study:**
Where to inject fast weights in P-TNCN?
  Option A: After state update (before inference correction)
  Option B: As additional term in error unit computation
  Option C: As modulation of E matrices (dynamic E_t = E_base + A_t)

**Phase 2 — Decay & learning rate:**
Vary `lambda_fw` ∈ {0.9, 0.95, 0.99} and `eta_fw` ∈ {0.1, 0.5, 1.0}.

**Phase 3 — Hybrid training:**
- Backprop for {W, R, fast_weight_scalars}
- LRA for {E matrices}
Log: whether hybrid training is stable, BPC, convergence speed.

---

## JAX Implementation Notes (for agent)

The agent should implement in JAX with the following structure:

```
ptncn_jax/
├── model.py          # PTNCN as a flax.linen.Module (or pure JAX)
├── train.py          # jit-compiled training step, this file is what you edit
├── data.py           # PTB dataloader (numpy-based, JAX-compatible)
├── metrics.py        # BPC calculation
└── config.py         # Hyperparameter dataclass (BaseConfig, ExpConfig)
```

**JAX idioms to follow:**
- Use `jax.jit` on the full train step for XLA fusion
- Use `jax.vmap` for batched inference correction (K steps)
- State management via `flax.struct.PyTreeNode` or `optax` `TrainState`
- All RNG keys must be threaded explicitly
- Weight matrices: `jnp.ndarray`, shapes annotated in comments
- Prefer `jax.lax.scan` over Python loops for time-step unrolling
- Use `jax.debug.print` for mid-JIT debugging (not Python print)

**JAX beginner notes (for human reading this):**
- `jax.grad` computes gradients just like PyTorch `.backward()`
- `jax.jit` = `torch.compile()` — wrap your train_step with it
- JAX arrays are immutable — use `jax.tree_util.tree_map` to update pytrees
- `optax` is the standard optimizer library (Adam, SGD, etc.)
- Key difference: JAX has no implicit state — carry everything explicitly

---

## Evaluation Protocol

Every experiment must log to `results/exp_{id}.json`:
```json
{
  "exp_id": "unique_id",
  "config": {...hyperparams...},
  "val_bpc_curve": [...],
  "val_bpc_final": float,
  "n_params": int,
  "wall_time_min": float,
  "notes": "agent's 1-sentence summary of what changed and why"
}
```

After each run, compare `val_bpc_final` to the current best. Keep the best
`train.py` as `train_best.py`. Add a 1-line entry to `experiment_log.md`.

---

## Agent Instructions

1. Read `program.md` (this file) before each experiment.
2. Read `experiment_log.md` to understand what has already been tried.
3. Choose ONE modification from the search space above. Prefer unexplored
   combinations. Do not repeat experiments.
4. Edit `train.py` only. Do not modify `data.py`, `metrics.py`, or `config.py`.
5. Run training: `python train.py --exp_id <id>`. It will terminate in ~10 min.
6. Read the output JSON, compare to best, update `experiment_log.md`.
7. If BPC improved: copy current `train.py` → `train_best.py`.
8. Loop.

**Hard constraints for the agent:**
- Never allocate more than 40GB GPU memory (leave 8GB headroom on L40)
- Never reduce `K` below 5 (model loses its inference correction property)
- `hidden_dim` must be a multiple of 64 for XLA efficiency
- Always report `n_params` so we track parameter efficiency

---

## Current Best Results

| Experiment | Config | Val BPC | Notes |
|------------|--------|---------|-------|
| baseline   | TF original | 1.43 | Reference from paper |
| jax_baseline | K=20, d=512, no LRA factorization | TBD | First JAX run |

---

## Open Research Questions (for human)

1. Does the local Hebbian update for E^ℓ remain valid when E^ℓ = A·B?
   (Mathematical analysis in `math/lra_ptncn_derivation.tex`)
2. Does fast weight injection interact constructively or destructively with
   the P-TNCN's K-step inference correction?
3. Can rec-LRA + fast weights achieve BPC competitive with BPTT-trained LSTMs
   on PTB (target: BPC ≈ 1.30)?
