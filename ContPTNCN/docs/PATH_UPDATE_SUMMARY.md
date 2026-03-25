# UV Environment Path Update Summary

## Current Setup (as of March 2026)

The project has been migrated from the research server to a local macOS machine.

**Repo root:** `/Users/hitesh/Documents/Research/ContinualPTNCN/`

All UV tooling (`pyproject.toml`, `uv.lock`, `.venv`) lives at the repo root.

### Directory Structure
```
/Users/hitesh/Documents/Research/ContinualPTNCN/   # Repo root
├── pyproject.toml          # Project metadata & dependencies
├── uv.lock                 # Pinned lockfile
├── .python-version         # Python 3.10
├── .venv/                  # Created by `uv sync` (not committed)
│   └── bin/
│       ├── activate
│       └── python
└── ContPTNCN/              # Project code
    ├── activate_env.sh
    ├── UV_COMMANDS.md
    ├── run_tuning.sh
    ├── run_full_workflow.sh
    ├── src/
    └── configs/
```

### Activation

```bash
# First-time setup (creates .venv from lockfile)
cd /Users/hitesh/Documents/Research/ContinualPTNCN
uv sync

# Activate
source .venv/bin/activate

# Or use the helper script from ContPTNCN/
source ContPTNCN/activate_env.sh

# Or skip activation and use uv run directly
uv run python ContPTNCN/src/quick_trial.py --config ContPTNCN/configs/baseline.json
```

---

## Previous Setup (archived)

Previously the project lived on a remote server at:
```
/data/hvaidya/ContinualPTNCN/
```

The `.venv` was manually created with:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install jax jaxlib numpy tqdm matplotlib
```

All scripts and documentation that previously referenced `/data/hvaidya/ContinualPTNCN/` have been updated to use `/Users/hitesh/Documents/Research/ContinualPTNCN/`.

---

## Updated Files

### Shell Scripts
1. **activate_env.sh** — now auto-detects repo root relative to its own location
2. **run_tuning.sh** — updated `.venv` path
3. **run_full_workflow.sh** — updated `.venv` path

### Documentation
1. **docs/QUICK_START.md** — updated all paths and workflow
2. **docs/EXPERIMENT_SETUP.md** — updated all paths
3. **docs/TUNING_README.md** — updated all paths, added UV setup section
4. **UV_COMMANDS.md** — updated all paths, updated to `uv sync` workflow

---

## Key Differences: Old vs New

| | Old (server) | New (local) |
|---|---|---|
| Repo root | `/data/hvaidya/ContinualPTNCN/` | `/Users/hitesh/Documents/Research/ContinualPTNCN/` |
| Setup | `uv venv .venv && uv pip install ...` | `uv sync` (from lockfile) |
| `.venv` path | `/data/hvaidya/ContinualPTNCN/.venv` | `/Users/hitesh/Documents/Research/ContinualPTNCN/.venv` |
| Dependencies | Manually specified | Locked in `uv.lock` via `pyproject.toml` |
| Python | System | 3.10 (pinned in `.python-version`) |
