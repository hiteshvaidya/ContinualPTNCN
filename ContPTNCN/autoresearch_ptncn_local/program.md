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
- Do not change learning rate in back-to-back experiments unless the last improvement was clearly strong.
- Rotate across parameter families instead of staying on one knob.
- If a change crashes, back off rather than repeatedly forcing larger settings.
- Use validation BPC as the main metric.

## PTNCN-specific guidance

- `hidden_size`, `learning_rate`, and `beta` are high-leverage parameters.
- Fast weights matter, but aggressive `fast_eta` can destabilize learning.
- Increasing `max_train_batches` improves signal but also increases experiment time.
- Batch size and hidden size interact with GPU memory.
- Activation, initialization, and predictive-coding parameters are valid search directions, not just learning rate.
- If validation BPC improves while keeping the code simple, keep it.

## Exploration rotation

Prefer to rotate among these families:

1. optimization
   - `LEARNING_RATE`, `MOMENTUM`, `USE_NESTEROV`
2. model size / runtime
   - `HIDDEN_SIZE`, `BATCH_SIZE`, `EVAL_BATCH_SIZE`, `MAX_TRAIN_BATCHES`, `MAX_EVAL_BATCHES`
3. fast weights
   - `FAST_STEPS`, `FAST_ETA`, `FAST_LAMBDA`
4. predictive coding
   - `BETA`, `ALPHA_ERROR`, `GAMMA`, `ZETA`, `UPDATE_RADIUS`, `PARAM_RADIUS`
5. representation dynamics
   - `ACTIVATION`, `INIT_TYPE`

After one experiment in one family, try a different family unless there is a very strong reason not to.

## Output style

Use SEARCH/REPLACE blocks only.

Explain the experiment in one short sentence before the blocks.
