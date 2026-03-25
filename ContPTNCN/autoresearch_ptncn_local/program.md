# PTNCN Autoresearch Program

You are optimizing a PTNCN PyTorch training wrapper for character-level Penn Treebank.

## Goal

Lower `val_bpb`.

Lower is better.

## Scope

You may only edit `train.py`.
Only edit the lines between `# BEGIN_TUNABLES` and `# END_TUNABLES`.
Do not change imports, functions, printing, logging, parsing, or any code outside that block.

You may change:

- hidden size
- batch size
- eval batch size
- epochs
- max train batches
- max eval batches
- learning rate
- momentum
- nesterov on/off
- beta
- alpha error
- gamma
- zeta
- update radius
- param radius
- fast steps
- fast eta
- fast lambda

Keep changes small and deliberate.

## Good search behavior

- Start from the current best run.
- Make one focused change at a time.
- Prefer simple changes that are easy to explain.
- If a change crashes, back off rather than repeatedly forcing larger settings.
- Use validation BPC as the main metric.

## PTNCN-specific guidance

- `hidden_size`, `learning_rate`, and `beta` are high-leverage parameters.
- Fast weights matter, but aggressive `fast_eta` can destabilize learning.
- Increasing `max_train_batches` improves signal but also increases experiment time.
- Batch size and hidden size interact with GPU memory.
- If validation BPC improves while keeping the code simple, keep it.

## Output style

Use SEARCH/REPLACE blocks only.

Explain the experiment in one short sentence before the blocks.
