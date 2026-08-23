#!/usr/bin/env python3
"""Run the fixed one-factor 96-RBF spiral capacity control.

This control changes only the number of RBF features from 48 to 96 relative to
the registered higher-capacity spiral experiment.  All other architecture,
optimizer, solver, damping, stopping, seed, and epoch settings remain fixed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_experiment(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_suite.py")),
        "--profile",
        "paper",
        "--datasets",
        "spirals",
        "--chain-sizes",
        "17",
        "--seeds",
        "101,202,303,404,505",
        "--methods",
        "implicit_id,standard_eqprop,boundary_dynamic",
        "--n-input-nodes",
        "12",
        "--n-rbf",
        "96",
        "--rbf-sigma",
        "0.55",
        "--epochs",
        "150",
        "--learning-rate-u",
        "0.01",
        "--learning-rate-structure",
        "0.001",
        "--damping-schedule",
        "graded_trace",
        "--damping-trace",
        "5.0",
        "--max-steps",
        "40000",
        "--tolerance",
        "2e-5",
        "--endpoint-step-multiplier",
        "10",
        "--phase-protocol",
        "exact_free_boundary_pair",
        "--workers",
        str(args.workers),
        "--output-dir",
        str(args.output_dir),
        "--resume",
    ]
    subprocess.run(command, check=True)


def summarize(output_dir: Path) -> None:
    runs = pd.read_csv(output_dir / "benchmark_runs.csv")
    valid_flag = runs["valid_dynamic_run"].astype(str).str.lower().eq("true")
    valid = runs.loc[(runs["run_status"] == "completed") & valid_flag].copy()
    if valid.empty:
        raise RuntimeError("no completed valid runs are available")

    summary = (
        valid.groupby(["dataset", "method"], as_index=False)
        .agg(
            n_runs=("test_accuracy", "size"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_accuracy_min=("test_accuracy", "min"),
            test_accuracy_max=("test_accuracy", "max"),
            gradient_relative_error_mean=(
                "gradient_relative_error_mean",
                "mean",
            ),
            gradient_relative_error_max=(
                "gradient_relative_error_max",
                "max",
            ),
            gradient_cosine_min=("gradient_cosine_min", "min"),
            endpoint_convergence_fraction=(
                "endpoint_convergence_fraction",
                "mean",
            ),
            wall_time_seconds_mean=("wall_time_seconds", "mean"),
        )
        .sort_values(["dataset", "method"])
    )
    summary.to_csv(output_dir / "spirals_96rbf_summary.csv", index=False)

    payload = {
        "registered_change": "n_rbf increased from 48 to 96",
        "unchanged": {
            "dataset": "spirals",
            "chain_size": 17,
            "n_input_nodes": 12,
            "rbf_sigma": 0.55,
            "epochs": 150,
            "seeds": [101, 202, 303, 404, 505],
            "damping_profile": "graded_trace",
            "damping_trace": 5.0,
            "endpoint_tolerance": 2e-5,
            "phase_protocol": "exact_free_boundary_pair",
        },
        "n_valid_runs": int(len(valid)),
        "n_failed_runs": int((runs["run_status"] != "completed").sum()),
    }
    (output_dir / "spirals_96rbf_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/high_capacity_spirals_96rbf_control"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summarize_only:
        run_experiment(args)
    summarize(args.output_dir)


if __name__ == "__main__":
    main()
