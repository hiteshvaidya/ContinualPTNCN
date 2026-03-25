# PTNCN Experiments And Autoresearch

## What the PyTorch trainer now saves

Each run creates its own directory under:

- `ContPTNCN/results/ptncn_torch/<run_name>/`

Artifacts per run:

- `config.json` — exact run configuration
- `history.csv` — epoch-by-epoch metrics, easy to open in pandas or Excel
- `history.json` — same history in JSON form
- `checkpoint_best.pt` — best validation checkpoint
- `checkpoint_last.pt` — final checkpoint from the latest epoch
- `metrics.png` — training curves for BPC and accuracy
- `final_metrics.json` — best validation and final test metrics

## Recommended first real GPU run

From `ContPTNCN/src`:

```bash
python train_ptncn_torch.py \
  --device cuda \
  --hidden-size 256 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --epochs 5 \
  --max-train-batches 200 \
  --max-eval-batches 5 \
  --nesterov \
  --run-name ptncn_gpu_trial_01
```

This is still a development run, not a final benchmark. It is large enough to produce meaningful logs and plots while still finishing quickly.

## How to learn from the code while it runs

The most useful places to read are:

1. `src/models/ptncn_torch.py`
   This is the model itself. Focus on `forward_step()` first, then `compute_updates()`.

2. `src/train_ptncn_torch.py`
   This shows how the local update rule is applied in practice. There is no `loss.backward()` call. Instead, the local updates are written into `parameter.grad` before `optimizer.step()`.

3. `docs/PTNCN_PYTORCH_PORT.md`
   This gives a compact map from the TensorFlow PTNCN to the PyTorch port.

## Ollama autoresearch: do we need opencode?

Short answer: no, not for the setup you described.

Why:

- `autoresearch-local-llm/agent.py` already talks directly to Ollama over HTTP.
- The model is selected through the `AUTORESEARCH_MODEL` environment variable.
- That means any model you have pulled in Ollama can be used without adding a separate coding agent runtime.

So if your intended local model is something like `rnj:latest` in Ollama, the normal launch pattern is:

```bash
cd /home/hvaidya/documents/autoresearch-local-llm
export AUTORESEARCH_MODEL=rnj:latest
bash run_pipeline.sh
```

Use the exact Ollama tag that `ollama list` shows.

## When opencode might help

`opencode` is optional. It may help only if you specifically want:

- a richer coding-agent interface
- tool calling or sandboxing beyond the current Python agent
- a different orchestration layer than the simple `agent.py` loop

It is not required for a first PTNCN autoresearch setup.

## What should change before running autoresearch on PTNCN

The current autoresearch repo is built around a GPT pretraining script, not PTNCN. Before launching autonomous experiments, we should adapt it so the search space matches PTNCN.

The main changes should be:

1. Replace its `train.py` with a PTNCN runner that trains for a fixed wall-clock budget.

2. Keep a single scalar objective:
   - use validation BPC for PTB-char
   - lower is better

3. Rewrite `program.md` so the agent explores PTNCN-specific ideas:
   - hidden size
   - learning rate and momentum
   - `beta`, `alpha_error`, and `gamma`
   - fast-weight parameters: `fast_steps`, `fast_eta`, `fast_lambda`
   - optional normalization or clipping choices

4. Keep the first search small:
   - PTB-char only
   - short run budget
   - one GPU
   - save every result to a TSV and keep best checkpoints

## Recommended sequencing

1. Run a few manual GPU experiments and inspect the saved plots.
2. Stabilize the PTNCN training defaults.
3. Adapt `autoresearch-local-llm` to call the PTNCN trainer.
4. Start autonomous search only after the manual baseline is reproducible.

That sequencing lets you learn the model while still offloading the repetitive trial-and-error loop once the baseline is trustworthy.
