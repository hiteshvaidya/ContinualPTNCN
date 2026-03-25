#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("AUTORESEARCH_MODEL", "rnj:latest")
TRAIN_SCRIPT = "train.py"
PROGRAM_FILE = "program.md"
RESULTS_FILE = "results.tsv"
RUN_LOG = "run.log"
RUN_TIMEOUT = 1800
MAX_CONSECUTIVE_CRASHES = 3


def query_llm(prompt: str, max_tokens: int = 2048) -> str:
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.6,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("response", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Ollama query failed: {exc}")
        return ""


def git_run(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=30, check=False)
    return result.stdout.strip()


def git_commit(message: str) -> str:
    subprocess.run(["git", "add", TRAIN_SCRIPT], check=True, timeout=30)
    subprocess.run(["git", "commit", "-m", message], check=True, timeout=30)
    return git_run("rev-parse", "--short", "HEAD")


def git_reset_hard(commit: str) -> None:
    subprocess.run(["git", "reset", "--hard", commit], check=True, timeout=30)


def get_current_commit() -> str:
    return git_run("rev-parse", "--short", "HEAD")


def run_experiment() -> tuple[float | None, float | None]:
    print(f"  Running experiment with timeout {RUN_TIMEOUT}s")
    try:
        with open(RUN_LOG, "w", encoding="utf-8") as handle:
            process = subprocess.run(
                [sys.executable, TRAIN_SCRIPT],
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT,
                check=False,
            )
        if process.returncode != 0:
            return None, None
    except subprocess.TimeoutExpired:
        return None, None

    val_bpb = None
    peak_vram = None
    with open(RUN_LOG, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("val_bpb:"):
                val_bpb = float(line.split(":", 1)[1].strip())
            if line.startswith("peak_vram_mb:"):
                peak_vram = float(line.split(":", 1)[1].strip())
    return val_bpb, peak_vram


def get_crash_info() -> str:
    path = Path(RUN_LOG)
    if not path.exists():
        return "run.log not found"
    return "".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[-80:])


def init_results() -> None:
    path = Path(RESULTS_FILE)
    if not path.exists():
        path.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")


def log_result(commit: str, val_bpb: float, memory_gb: float, status: str, description: str) -> None:
    with open(RESULTS_FILE, "a", encoding="utf-8") as handle:
        handle.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def get_results_history() -> str:
    path = Path(RESULTS_FILE)
    if not path.exists():
        return "No results yet."
    return path.read_text(encoding="utf-8")


def get_best_bpb() -> float:
    best = float("inf")
    path = Path(RESULTS_FILE)
    if not path.exists():
        return best
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            try:
                best = min(best, float(parts[1]))
            except ValueError:
                pass
    return best


def read_train_py() -> str:
    return Path(TRAIN_SCRIPT).read_text(encoding="utf-8")


def write_train_py(code: str) -> None:
    Path(TRAIN_SCRIPT).write_text(code, encoding="utf-8")


def validate_syntax(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


def apply_search_replace(code: str, search: str, replace: str) -> tuple[str, bool]:
    if search in code:
        return code.replace(search, replace, 1), True
    return code, False


def parse_search_replace_blocks(response: str) -> list[tuple[str, str]]:
    pattern = r"<<<SEARCH\n(.*?)>>>\s*<<<REPLACE\n(.*?)>>>"
    return [(search.rstrip("\n"), replace.rstrip("\n")) for search, replace in re.findall(pattern, response, re.DOTALL)]


def extract_tunable_block(code: str) -> str:
    start = code.find("# BEGIN_TUNABLES")
    end = code.find("# END_TUNABLES")
    if start == -1 or end == -1 or end <= start:
        return code
    end += len("# END_TUNABLES")
    return code[start:end]


def build_experiment_prompt(train_code: str, results_history: str, best_bpb: float, crash_info: str | None = None) -> str:
    program_text = Path(PROGRAM_FILE).read_text(encoding="utf-8")
    tunable_block = extract_tunable_block(train_code)
    crash_section = f"Last crash info:\n{crash_info}\n" if crash_info else ""
    return f"""You are an autonomous ML researcher running local PTNCN experiments.

Current best val_bpb: {best_bpb:.6f}

You may only edit the tunable block inside train.py.

Reference instructions:
{program_text}

Current tunable block from train.py:
```python
{tunable_block}
```

Results history:
{results_history}

{crash_section}

Reply with:
1. One sentence explaining the experiment.
2. SEARCH/REPLACE blocks only.

Format:
<<<SEARCH
exact old text
>>>
<<<REPLACE
new text
>>>

Make one focused change. Keep edits small and valid Python."""


def main() -> None:
    print("=" * 60)
    print("PTNCN Local Autoresearch Agent")
    print(f"Model: {MODEL}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    init_results()
    experiment_num = 0
    consecutive_crashes = 0

    results = get_results_history()
    if "baseline" not in results.lower():
        print("\n--- Experiment 0: Baseline ---")
        base_commit = get_current_commit()
        val_bpb, peak_vram = run_experiment()
        if val_bpb is None:
            log_result(base_commit, 0.0, 0.0, "crash", "baseline failed")
            print("  Baseline failed. The agent will try fixes.")
        else:
            log_result(base_commit, val_bpb, (peak_vram or 0.0) / 1024.0, "keep", "baseline")
            print(f"  Baseline val_bpb: {val_bpb:.6f}")
        experiment_num = 1

    while True:
        print(f"\n{'=' * 60}")
        print(f"Experiment {experiment_num}")
        print(f"Time: {datetime.now().isoformat()}")

        best_bpb = get_best_bpb()
        base_commit = get_current_commit()
        train_code = read_train_py()
        results_history = get_results_history()
        crash_context = get_crash_info() if consecutive_crashes > 0 else None

        prompt = build_experiment_prompt(train_code, results_history, best_bpb, crash_context)
        print("  Querying local model...")
        response = query_llm(prompt)
        if not response:
            print("  Empty response from model, waiting before retry")
            time.sleep(10)
            experiment_num += 1
            continue

        blocks = parse_search_replace_blocks(response)
        if not blocks:
            print("  No SEARCH/REPLACE blocks found")
            experiment_num += 1
            continue

        modified = train_code
        applied = True
        for search, replace in blocks:
            modified, success = apply_search_replace(modified, search, replace)
            if not success:
                applied = False
                break

        if not applied:
            print("  Could not apply proposed edit cleanly")
            consecutive_crashes += 1
            experiment_num += 1
            continue

        valid, error = validate_syntax(modified)
        if not valid:
            print(f"  Proposed edit is invalid Python: {error}")
            consecutive_crashes += 1
            experiment_num += 1
            continue

        description = next(
            (
                line.strip()
                for line in response.splitlines()
                if line.strip() and not line.startswith("<<<") and not line.startswith(">>>")
            ),
            f"experiment {experiment_num}",
        )[:120]

        write_train_py(modified)
        try:
            commit_hash = git_commit(f"exp{experiment_num}: {description}")
        except Exception as exc:
            print(f"  Git commit failed: {exc}")
            git_reset_hard(base_commit)
            experiment_num += 1
            continue

        val_bpb, peak_vram = run_experiment()
        if val_bpb is None:
            log_result(commit_hash, 0.0, 0.0, "crash", description)
            print("  Experiment crashed")
            git_reset_hard(base_commit)
            consecutive_crashes += 1
        else:
            consecutive_crashes = 0
            memory_gb = (peak_vram or 0.0) / 1024.0
            if val_bpb < best_bpb:
                log_result(commit_hash, val_bpb, memory_gb, "keep", description)
                print(f"  KEEP: {val_bpb:.6f} improved over {best_bpb:.6f}")
            else:
                log_result(commit_hash, val_bpb, memory_gb, "discard", description)
                print(f"  DISCARD: {val_bpb:.6f} >= {best_bpb:.6f}")
                git_reset_hard(base_commit)

        experiment_num += 1
        print(f"  Best val_bpb so far: {get_best_bpb():.6f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
