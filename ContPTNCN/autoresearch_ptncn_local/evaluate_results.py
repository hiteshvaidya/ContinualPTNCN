#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def resolve_results_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        candidate = path / "results.tsv"
        if candidate.exists():
            return candidate
    return path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize(path: Path) -> dict[str, object]:
    rows = load_rows(path)
    total = len(rows)
    keeps = [row for row in rows if row.get("status") == "keep"]
    discards = [row for row in rows if row.get("status") == "discard"]
    crashes = [row for row in rows if row.get("status") == "crash"]

    baseline_row = next((row for row in rows if row.get("description", "").lower() == "baseline"), None)
    baseline_bpb = to_float(baseline_row["val_bpb"]) if baseline_row else None

    best_keep = None
    if keeps:
        best_keep = min(keeps, key=lambda row: to_float(row["val_bpb"], default=float("inf")))

    best_bpb = to_float(best_keep["val_bpb"], default=float("inf")) if best_keep else None
    improvement = None
    if baseline_bpb is not None and best_bpb is not None:
        improvement = baseline_bpb - best_bpb

    summary = {
        "path": str(path),
        "total_runs": total,
        "keeps": len(keeps),
        "discards": len(discards),
        "crashes": len(crashes),
        "keep_rate": (len(keeps) / total) if total else 0.0,
        "crash_rate": (len(crashes) / total) if total else 0.0,
        "baseline_bpb": baseline_bpb,
        "best_bpb": best_bpb,
        "improvement_over_baseline": improvement,
        "best_description": best_keep["description"] if best_keep else None,
        "best_commit": best_keep["commit"] if best_keep else None,
        "avg_keep_bpb": (
            sum(to_float(row["val_bpb"]) for row in keeps) / len(keeps)
            if keeps
            else None
        ),
    }
    return summary


def print_single(summary: dict[str, object]) -> None:
    print(f"Results file: {summary['path']}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Keeps: {summary['keeps']}")
    print(f"Discards: {summary['discards']}")
    print(f"Crashes: {summary['crashes']}")
    print(f"Keep rate: {summary['keep_rate']:.2%}")
    print(f"Crash rate: {summary['crash_rate']:.2%}")
    if summary["baseline_bpb"] is not None:
        print(f"Baseline BPC: {summary['baseline_bpb']:.6f}")
    if summary["best_bpb"] is not None:
        print(f"Best BPC: {summary['best_bpb']:.6f}")
    if summary["improvement_over_baseline"] is not None:
        print(f"Improvement over baseline: {summary['improvement_over_baseline']:.6f}")
    if summary["best_description"]:
        print(f"Best experiment: {summary['best_description']}")
    if summary["best_commit"]:
        print(f"Best commit: {summary['best_commit']}")


def print_comparison(summaries: list[dict[str, object]]) -> None:
    headers = [
        "label",
        "runs",
        "keeps",
        "crashes",
        "keep_rate",
        "crash_rate",
        "best_bpb",
        "improvement",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            [
                Path(str(summary["path"])).parent.name,
                str(summary["total_runs"]),
                str(summary["keeps"]),
                str(summary["crashes"]),
                f"{summary['keep_rate']:.2%}",
                f"{summary['crash_rate']:.2%}",
                f"{summary['best_bpb']:.6f}" if summary["best_bpb"] is not None else "n/a",
                (
                    f"{summary['improvement_over_baseline']:.6f}"
                    if summary["improvement_over_baseline"] is not None
                    else "n/a"
                ),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PTNCN autoresearch results.tsv files.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["results.tsv"],
        help="Path(s) to results.tsv files or directories containing results.tsv",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    resolved = [resolve_results_path(path_str) for path_str in args.paths]
    for path in resolved:
        if not path.exists():
            raise SystemExit(f"Results file not found: {path}")

    summaries = [summarize(path) for path in resolved]

    if args.json:
        print(json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2))
        return

    if len(summaries) == 1:
        print_single(summaries[0])
    else:
        print_comparison(summaries)


if __name__ == "__main__":
    main()
