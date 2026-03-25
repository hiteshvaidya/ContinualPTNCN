#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ContPTNCN" / "data" / "ptb_char"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def main() -> None:
    required_files = [
        DATA_DIR / "trainX.txt",
        DATA_DIR / "subX.txt",
        DATA_DIR / "validX.txt",
        DATA_DIR / "testX.txt",
        DATA_DIR / "vocab.txt",
    ]

    print("Checking PTNCN autoresearch prerequisites...")
    for path in required_files:
        if not path.exists():
            raise SystemExit(f"Missing required dataset file: {path}")

    if not VENV_PYTHON.exists():
        raise SystemExit(f"Missing project virtualenv python: {VENV_PYTHON}")

    if shutil.which("ollama") is None:
        raise SystemExit("Ollama CLI not found in PATH. Install Ollama before starting the local agent.")

    print(f"Dataset directory: {DATA_DIR}")
    print(f"Project python: {VENV_PYTHON}")
    print("Prerequisites look good.")


if __name__ == "__main__":
    main()
