# PTNCN Autoresearch With A Local LLM

This folder contains a PTNCN-specific autoresearch loop that runs fully locally:

- Ollama serves the language model
- `agent.py` proposes edits to `train.py`
- `train.py` launches the PTNCN PyTorch trainer with a fixed evaluation setup
- `results.tsv` records keep/discard outcomes

No OpenAI or Anthropic API is required.

## Quick start

From the repo root:

```bash
source .venv/bin/activate
ollama serve
ollama pull rnj:latest
cd ContPTNCN/autoresearch_ptncn_local
export AUTORESEARCH_MODEL=rnj:latest
bash run_pipeline.sh
```

Replace `rnj:latest` with the exact tag shown by `ollama list`.

Run this on a clean git working tree or in a dedicated clone/worktree. The agent keeps or discards experiments via git commits and resets.

## Files

- `train.py` — the only file the agent edits
- `agent.py` — the autonomous experiment loop
- `evaluate_results.py` — summarizes one or more `results.tsv` files
- `prepare.py` — validates local prerequisites
- `program.md` — PTNCN-specific search guidance for the local model
- `run_pipeline.sh` — sets up the branch and starts the loop

## Why use an LLM here?

The language model is the proposal engine, not the trainer.

It reads:

- the current `train.py`
- previous results in `results.tsv`
- the PTNCN search instructions in `program.md`

Then it proposes the next experiment by editing the hyperparameters in `train.py`.

If you only want numeric hyperparameter tuning, you do not strictly need an LLM. A random search or Bayesian optimizer could also work. The local LLM becomes useful when you want the search to reason over coupled choices and occasionally propose structural changes instead of only sampling numbers.

## Small-model safety

The agent is intentionally constrained to edit only the `# BEGIN_TUNABLES` ... `# END_TUNABLES` block in `train.py`.

That makes small local models such as `qwen3.5-9b` or `rnj-1` less likely to break the training wrapper.

## Evaluating a run

From `ContPTNCN/autoresearch_ptncn_local`:

```bash
python evaluate_results.py results.tsv
```

To compare two runs:

```bash
python evaluate_results.py /path/to/runA/results.tsv /path/to/runB/results.tsv
```
