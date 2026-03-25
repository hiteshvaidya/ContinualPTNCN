#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.ptncn_torch import PTNCN
from utils.discrete_sequence_torch import PaddedSequenceLoader, SequenceBatch, Vocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_batches(loader: PaddedSequenceLoader, max_batches: int | None):
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


def build_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ptncn_torch_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_history_row(path: Path, row: dict[str, float | int]) -> None:
    fieldnames = list(row.keys())
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    *,
    model: PTNCN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def maybe_plot_history(history: list[dict[str, float | int]], plot_path: Path, cache_dir: Path) -> bool:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    epochs = [int(row["epoch"]) for row in history]
    train_bpc = [float(row["train_bpc"]) for row in history]
    valid_bpc = [float(row["valid_bpc"]) for row in history]
    subtrain_bpc = [float(row["subtrain_bpc"]) for row in history]
    train_acc = [float(row["train_acc"]) for row in history]
    valid_acc = [float(row["valid_acc"]) for row in history]

    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(epochs, train_bpc, marker="o", label="train_bpc")
    axes[0].plot(epochs, subtrain_bpc, marker="o", label="subtrain_bpc")
    axes[0].plot(epochs, valid_bpc, marker="o", label="valid_bpc")
    axes[0].set_ylabel("Bits Per Character")
    axes[0].set_title("PTNCN Training Curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[1].plot(epochs, valid_acc, marker="o", label="valid_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return True


def one_hot_from_indices(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    safe_indices = indices.clamp_min(0)
    one_hot = F.one_hot(safe_indices, num_classes=vocab_size).float()
    return one_hot * (indices >= 0).unsqueeze(1).float()


def token_nll(probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    safe_targets = targets.clamp_min(0)
    picked = probs.gather(dim=1, index=safe_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
    nll = -(picked.log() * mask)
    preds = probs.argmax(dim=1)
    correct = (preds == safe_targets).float() * mask
    return nll.sum(), correct.sum()


def run_sequence(
    model: PTNCN,
    batch: SequenceBatch,
    vocab_size: int,
    *,
    device: torch.device,
    beta: float,
    alpha_error: float,
    t_prime: int,
    optimizer: torch.optim.Optimizer | None = None,
    gamma: float = 1.0,
    update_radius: float = -1.0,
    param_radius: float = -1.0,
) -> tuple[float, float, float]:
    tokens = batch.tokens.to(device)
    mask = batch.mask.to(device)
    total_nll = 0.0
    total_correct = 0.0
    total_count = 0.0

    with torch.no_grad():
        for timestep in range(tokens.size(1)):
            current_targets = tokens[:, timestep]
            current_mask = mask[:, timestep].unsqueeze(1)
            x_t = one_hot_from_indices(current_targets, vocab_size).to(device)

            _, probs = model.forward_step(x_t, current_mask, beta=beta, alpha=alpha_error)
            if timestep < t_prime:
                continue

            step_nll, step_correct = token_nll(probs, current_targets, current_mask.squeeze(1))
            total_nll += float(step_nll.item())
            total_correct += float(step_correct.item())
            total_count += float(current_mask.sum().item())

            if optimizer is not None:
                updates = model.compute_updates(gamma=gamma, update_radius=update_radius)
                batch_scale = max(tokens.size(0), 1)
                optimizer.zero_grad(set_to_none=True)
                for parameter, update in zip(model.parameter_list(), updates):
                    parameter.grad = update / batch_scale
                optimizer.step()

                if param_radius > 0.0:
                    for parameter in model.parameter_list():
                        row_norms = parameter.norm(dim=1, keepdim=True).clamp_min(1e-8)
                        clipped = parameter * torch.clamp(param_radius / row_norms, max=1.0)
                        parameter.copy_(clipped)

    model.reset_state(reset_fast_weights=True)
    return total_nll, total_correct, total_count


def evaluate(
    model: PTNCN,
    loader: PaddedSequenceLoader,
    vocab_size: int,
    *,
    device: torch.device,
    beta: float,
    alpha_error: float,
    t_prime: int,
    max_batches: int | None,
) -> dict[str, float]:
    total_nll = 0.0
    total_correct = 0.0
    total_count = 0.0

    for batch in maybe_limit_batches(loader, max_batches):
        seq_nll, seq_correct, seq_count = run_sequence(
            model,
            batch,
            vocab_size,
            device=device,
            beta=beta,
            alpha_error=alpha_error,
            t_prime=t_prime,
            optimizer=None,
        )
        total_nll += seq_nll
        total_correct += seq_correct
        total_count += seq_count

    mean_nll = total_nll / max(total_count, 1.0)
    return {
        "nll": mean_nll,
        "bpc": mean_nll / math.log(2.0),
        "accuracy": total_correct / max(total_count, 1.0),
    }


def train(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    run_dir = build_run_dir(args)
    history_csv = run_dir / "history.csv"
    history_json = run_dir / "history.json"
    config_json = run_dir / "config.json"
    best_ckpt = run_dir / "checkpoint_best.pt"
    last_ckpt = run_dir / "checkpoint_last.pt"
    plot_path = run_dir / "metrics.png"
    mpl_cache_dir = run_dir / ".mplcache"
    final_metrics_json = run_dir / "final_metrics.json"
    vocab = Vocab(data_dir / "vocab.txt")
    save_json(config_json, vars(args))

    train_loader = PaddedSequenceLoader(data_dir / "trainX.txt", args.batch_size, shuffle=True)
    subtrain_loader = PaddedSequenceLoader(data_dir / "subX.txt", args.eval_batch_size, shuffle=False)
    valid_loader = PaddedSequenceLoader(data_dir / "validX.txt", args.eval_batch_size, shuffle=False)
    test_loader = PaddedSequenceLoader(data_dir / "testX.txt", args.eval_batch_size, shuffle=False)

    model = PTNCN(
        x_dim=vocab.size,
        hid_dim=args.hidden_size,
        act_fun=args.activation,
        out_fun="softmax",
        init_type=args.init_type,
        wght_sd=args.weight_std,
        err_wght_sd=args.err_weight_std,
        zeta=args.zeta,
        fast_steps=args.fast_steps,
        fast_eta=args.fast_eta,
        fast_lambda=args.fast_lambda,
        device=device,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameter_list(),
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=args.nesterov,
    )

    print("=" * 72)
    print("PyTorch PTNCN")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Vocab size: {vocab.size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Dataset: {data_dir}")
    print(f"Run directory: {run_dir}")

    best_valid_bpc = float("inf")
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch:02d}/{args.epochs} started")
        epoch_nll = 0.0
        epoch_correct = 0.0
        epoch_count = 0.0

        for batch in maybe_limit_batches(train_loader, args.max_train_batches):
            seq_nll, seq_correct, seq_count = run_sequence(
                model,
                batch,
                vocab.size,
                device=device,
                beta=args.beta,
                alpha_error=args.alpha_error,
                t_prime=args.t_prime,
                optimizer=optimizer,
                gamma=args.gamma,
                update_radius=args.update_radius,
                param_radius=args.param_radius,
            )
            epoch_nll += seq_nll
            epoch_correct += seq_correct
            epoch_count += seq_count

        train_nll = epoch_nll / max(epoch_count, 1.0)
        train_bpc = train_nll / math.log(2.0)
        train_acc = epoch_correct / max(epoch_count, 1.0)

        subtrain_metrics = evaluate(
            model,
            subtrain_loader,
            vocab.size,
            device=device,
            beta=args.beta,
            alpha_error=args.alpha_error,
            t_prime=args.t_prime,
            max_batches=args.max_eval_batches,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            vocab.size,
            device=device,
            beta=args.beta,
            alpha_error=args.alpha_error,
            t_prime=args.t_prime,
            max_batches=args.max_eval_batches,
        )

        epoch_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_nll": train_nll,
            "train_bpc": train_bpc,
            "train_acc": train_acc,
            "subtrain_nll": subtrain_metrics["nll"],
            "subtrain_bpc": subtrain_metrics["bpc"],
            "subtrain_acc": subtrain_metrics["accuracy"],
            "valid_nll": valid_metrics["nll"],
            "valid_bpc": valid_metrics["bpc"],
            "valid_acc": valid_metrics["accuracy"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        append_history_row(history_csv, row)
        save_json(history_json, {"epochs": history})

        print(
            f"Epoch {epoch:02d} finished in {epoch_seconds:.2f}s | "
            f"train_bpc={train_bpc:.4f} train_acc={train_acc:.4f} | "
            f"subtrain_bpc={subtrain_metrics['bpc']:.4f} | "
            f"valid_bpc={valid_metrics['bpc']:.4f} valid_acc={valid_metrics['accuracy']:.4f}"
        )

        checkpoint_metrics = {
            "train_bpc": train_bpc,
            "train_acc": train_acc,
            "valid_bpc": valid_metrics["bpc"],
            "valid_acc": valid_metrics["accuracy"],
            "subtrain_bpc": subtrain_metrics["bpc"],
        }
        save_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=checkpoint_metrics,
            args=args,
        )

        if valid_metrics["bpc"] < best_valid_bpc:
            best_valid_bpc = valid_metrics["bpc"]
            best_metrics = valid_metrics
            save_checkpoint(
                best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=checkpoint_metrics,
                args=args,
            )
            print(f"  New best checkpoint saved: {best_ckpt.name} (valid_bpc={best_valid_bpc:.4f})")

        if not args.no_plots:
            plotted = maybe_plot_history(history, plot_path, mpl_cache_dir)
            if epoch == 1 and not plotted:
                print("  Plot skipped: matplotlib is not installed in this environment")

    test_metrics = evaluate(
        model,
        test_loader,
        vocab.size,
        device=device,
        beta=args.beta,
        alpha_error=args.alpha_error,
        t_prime=args.t_prime,
        max_batches=args.max_eval_batches,
    )

    result = {
        "best_valid_bpc": best_valid_bpc,
        "best_valid_accuracy": best_metrics.get("accuracy", 0.0),
        "test_bpc": test_metrics["bpc"],
        "test_accuracy": test_metrics["accuracy"],
        "run_dir": str(run_dir),
        "history_csv": str(history_csv),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
    }

    save_json(final_metrics_json, result)
    print("Final metrics:")
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PyTorch PTNCN on PTB-char.")
    parser.add_argument("--data-dir", type=str, default="../data/ptb_char")
    parser.add_argument("--output-dir", type=str, default="../results/ptncn_torch")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--init-type", type=str, default="normal")
    parser.add_argument("--weight-std", type=float, default=0.05)
    parser.add_argument("--err-weight-std", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha-error", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--t-prime", type=int, default=1)
    parser.add_argument("--update-radius", type=float, default=1.0)
    parser.add_argument("--param-radius", type=float, default=-1.0)
    parser.add_argument("--fast-steps", type=int, default=2)
    parser.add_argument("--fast-eta", type=float, default=0.01)
    parser.add_argument("--fast-lambda", type=float, default=0.9)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
