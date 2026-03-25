# PTNCN Autoresearch Pipeline

## Exact pipeline

The PTNCN local autoresearch loop now lives in:

- `ContPTNCN/autoresearch_ptncn_local/`

The execution flow is:

1. `run_pipeline.sh`
   - checks local prerequisites with `prepare.py`
   - aborts if the git working tree is dirty
   - creates or switches to a git branch for the search
   - starts `agent.py`

2. `agent.py`
   - runs a baseline experiment first
   - reads `results.tsv`
   - reads `program.md`
   - reads the current `train.py`
   - asks Ollama for one small edit to `train.py`
   - commits the edit
   - runs `uv run train.py`
   - parses `val_bpb` and `peak_vram_mb`
   - logs the result to `results.tsv`
   - keeps the commit if it improved the metric
   - otherwise resets back to the previous commit
   - repeats forever until you stop it

3. `train.py`
   - is a thin wrapper around `src/train_ptncn_torch.py`
   - contains editable PTNCN hyperparameters as simple constants
   - launches a fixed PTNCN training/evaluation run
   - prints the scalar metrics that `agent.py` needs

4. `src/train_ptncn_torch.py`
   - performs the actual PTNCN training
   - saves checkpoints, logs, and plots

## Why do we need the language model?

Strictly speaking, we do not *need* it.

You can search PTNCN settings without any LLM using:

- random search
- grid search
- Bayesian optimization
- Optuna-style tuning

The language model is useful because it acts as a proposal engine:

- it reads the history of what already worked or failed
- it proposes the next edit
- it can make coupled decisions across several settings at once
- it can eventually move beyond pure numeric tuning into lightweight code changes

For your goal, the local LLM is helpful because it automates the repetitive “try, run, log, compare, keep or discard” loop while you focus on understanding the PTNCN code and the results.

## When an LLM is overkill

If you only want to tune a short list of numeric PTNCN hyperparameters, a classical tuner is simpler and often more reliable than a small LLM.

The local LLM becomes more attractive when:

- you want a zero-API-cost autonomous loop
- you want the search to use experiment history semantically
- you want it to suggest more human-like follow-up experiments

## Zero-token-cost setup

To keep the loop at zero ChatGPT/Claude token cost:

- run Ollama locally
- use a local Ollama model only
- do not call OpenAI or Anthropic APIs from the agent

Example:

```bash
cd /home/hvaidya/documents/ContinualPTNCN
source .venv/bin/activate
ollama serve
ollama pull rnj:latest
cd ContPTNCN/autoresearch_ptncn_local
export AUTORESEARCH_MODEL=rnj:latest
bash run_pipeline.sh
```

Replace `rnj:latest` with the exact tag from `ollama list`.
