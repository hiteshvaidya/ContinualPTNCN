#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "ContPTNCN" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_ptncn_torch import build_parser, train as run_ptncn_train  # noqa: E402


# ---------------------------------------------------------------------------
# Editable experiment configuration
# ---------------------------------------------------------------------------

# BEGIN_TUNABLES
DATA_DIR = str(ROOT / "ContPTNCN" / "data" / "ptb_char")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "results")
RUN_NAME = "current_trial"
DEVICE = "cuda"

HIDDEN_SIZE = 64
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
EPOCHS = 1
MAX_TRAIN_BATCHES = 20
MAX_EVAL_BATCHES = 2

LEARNING_RATE = 0.025
MOMENTUM = 0.9
USE_NESTEROV = True

ACTIVATION = "tanh"
INIT_TYPE = "normal"
WEIGHT_STD = 0.05
ERR_WEIGHT_STD = 0.05

BETA = 0.1
ALPHA_ERROR = 0.001
GAMMA = 1.0
ZETA = 1.0
UPDATE_RADIUS = 1.0
PARAM_RADIUS = -1.0

FAST_STEPS = 2
FAST_ETA = 0.01
FAST_LAMBDA = 0.9

SEED = 1234
# END_TUNABLES


def make_args():
    argv = [
        "--data-dir",
        DATA_DIR,
        "--output-dir",
        OUTPUT_DIR,
        "--run-name",
        RUN_NAME,
        "--device",
        DEVICE,
        "--seed",
        str(SEED),
        "--batch-size",
        str(BATCH_SIZE),
        "--eval-batch-size",
        str(EVAL_BATCH_SIZE),
        "--hidden-size",
        str(HIDDEN_SIZE),
        "--epochs",
        str(EPOCHS),
        "--learning-rate",
        str(LEARNING_RATE),
        "--momentum",
        str(MOMENTUM),
        "--activation",
        ACTIVATION,
        "--init-type",
        INIT_TYPE,
        "--weight-std",
        str(WEIGHT_STD),
        "--err-weight-std",
        str(ERR_WEIGHT_STD),
        "--beta",
        str(BETA),
        "--alpha-error",
        str(ALPHA_ERROR),
        "--gamma",
        str(GAMMA),
        "--zeta",
        str(ZETA),
        "--update-radius",
        str(UPDATE_RADIUS),
        "--param-radius",
        str(PARAM_RADIUS),
        "--fast-steps",
        str(FAST_STEPS),
        "--fast-eta",
        str(FAST_ETA),
        "--fast-lambda",
        str(FAST_LAMBDA),
        "--max-train-batches",
        str(MAX_TRAIN_BATCHES),
        "--max-eval-batches",
        str(MAX_EVAL_BATCHES),
    ]
    if USE_NESTEROV:
        argv.append("--nesterov")
    return build_parser().parse_args(argv)


def main() -> None:
    start = time.time()
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args = make_args()
    result = run_ptncn_train(args)

    training_seconds = time.time() - start
    peak_vram_mb = 0.0
    if DEVICE == "cuda" and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    summary = {
        "val_bpb": result["best_valid_bpc"],
        "test_bpc": result["test_bpc"],
        "training_seconds": training_seconds,
        "peak_vram_mb": peak_vram_mb,
        "run_dir": result["run_dir"],
    }

    print("---")
    print(f"val_bpb:          {summary['val_bpb']:.6f}")
    print(f"test_bpc:         {summary['test_bpc']:.6f}")
    print(f"training_seconds: {summary['training_seconds']:.1f}")
    print(f"peak_vram_mb:     {summary['peak_vram_mb']:.1f}")
    print(f"run_dir:          {summary['run_dir']}")
    print("config_json:")
    print(json.dumps(
        {
            "hidden_size": HIDDEN_SIZE,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "epochs": EPOCHS,
            "max_train_batches": MAX_TRAIN_BATCHES,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "use_nesterov": USE_NESTEROV,
            "beta": BETA,
            "alpha_error": ALPHA_ERROR,
            "gamma": GAMMA,
            "zeta": ZETA,
            "update_radius": UPDATE_RADIUS,
            "param_radius": PARAM_RADIUS,
            "fast_steps": FAST_STEPS,
            "fast_eta": FAST_ETA,
            "fast_lambda": FAST_LAMBDA,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
