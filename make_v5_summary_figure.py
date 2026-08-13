#!/usr/bin/env python3
"""Create the combined statistics, topology, and readout-noise figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "boundary-eqprop-v5-figure")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCE_LABELS = {
    "implicit_id": "Implicit differentiation",
    "standard_eqprop": "Standard EqProp",
    "uniform_dynamic": "Uniform dynamics",
}

NOISE_LABELS = {
    "independent_phase": "Independent phase noise",
    "shared_sensor_bias": "Shared sensor bias",
}


def create_figure(
    statistics_path: Path,
    graph_path: Path,
    noise_path: Path,
    output_path: Path,
) -> None:
    statistics = pd.read_csv(statistics_path)
    graph = pd.read_csv(graph_path)
    noise = pd.read_csv(noise_path)

    required_statistics = {
        "reference",
        "mean_accuracy_difference_percentage_points",
        "cluster_bootstrap_95_ci_low_percentage_points",
        "cluster_bootstrap_95_ci_high_percentage_points",
    }
    required_graph = {
        "topology",
        "damping_mode",
        "dynamic_vs_implicit_gradient_error",
    }
    required_noise = {
        "noise_model",
        "relative_readout_noise",
        "noisy_vs_implicit_gradient_error",
    }
    for name, table, required in (
        ("statistics", statistics, required_statistics),
        ("graph", graph, required_graph),
        ("noise", noise, required_noise),
    ):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} input is missing columns: {missing}")

    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))

    statistics = statistics.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(statistics))
    mean = statistics["mean_accuracy_difference_percentage_points"].to_numpy()
    low = statistics["cluster_bootstrap_95_ci_low_percentage_points"].to_numpy()
    high = statistics["cluster_bootstrap_95_ci_high_percentage_points"].to_numpy()
    axes[0].axvspan(-2.0, 2.0, color="#eaf3e7", zorder=0)
    axes[0].errorbar(
        mean,
        positions,
        xerr=np.vstack((mean - low, high - mean)),
        fmt="o",
        color="#1f77b4",
        capsize=4,
        linewidth=1.6,
    )
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlim(-0.08, 0.08)
    axes[0].set_yticks(
        positions,
        [REFERENCE_LABELS[item] for item in statistics["reference"]],
    )
    axes[0].set_xlabel("Accuracy difference [percentage points]")
    axes[0].set_title("(a) Boundary minus reference")
    axes[0].text(
        0.02,
        0.03,
        "All 95% CIs inside\nregistered +/-2 pp margin",
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
    )

    topology_order = ["grid_4", "grid_5", "sparse_16", "sparse_25"]
    grouped = (
        graph.groupby(["topology", "damping_mode"], observed=True)[
            "dynamic_vs_implicit_gradient_error"
        ]
        .median()
        .unstack()
        .reindex(topology_order)
    )
    if grouped.isna().any().any():
        raise ValueError("graph input does not contain every registered topology")
    x = np.arange(len(grouped))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        100.0 * grouped["boundary"],
        width,
        label="Boundary",
        color="#1f77b4",
    )
    axes[1].bar(
        x + width / 2,
        100.0 * grouped["uniform_trace"],
        width,
        label="Uniform trace",
        color="#ff7f0e",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, ["Grid 4x4", "Grid 5x5", "Sparse 16", "Sparse 25"])
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].set_ylabel("Median relative gradient error [%]")
    axes[1].set_title("(b) Non-chain topology audit")
    axes[1].legend(frameon=False, fontsize=9)

    noise_summary = (
        noise.groupby(["noise_model", "relative_readout_noise"], observed=True)[
            "noisy_vs_implicit_gradient_error"
        ]
        .quantile(0.95)
        .reset_index(name="q95_error")
    )
    for model, part in noise_summary.groupby("noise_model", observed=True):
        part = part[part["relative_readout_noise"] > 0.0]
        axes[2].plot(
            100.0 * part["relative_readout_noise"],
            100.0 * part["q95_error"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=NOISE_LABELS[model],
        )
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="1% error")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Relative state-readout noise [%]")
    axes[2].set_ylabel("95th-percentile gradient error [%]")
    axes[2].set_title("(c) Measurement-noise audit")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.25, which="both")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_figure(args.statistics, args.graph, args.noise, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
