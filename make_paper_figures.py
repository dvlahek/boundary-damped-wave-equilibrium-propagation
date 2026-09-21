#!/usr/bin/env python3
"""Generate manuscript figures from locked numerical result tables.

The script does not rerun simulations. It reads the CSV outputs produced by the
registered numerical audits, creates publication figures, and writes a
provenance JSON containing the exact input paths and SHA-256 hashes used for
each output figure.

Expected result directories can be supplied explicitly. Defaults match the
current paper-scale audit output names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_input(provenance: dict, figure: str, path: Path) -> None:
    provenance.setdefault(figure, {}).setdefault("inputs", []).append(
        {"path": str(path), "sha256": sha256(path)}
    )


def finish_figure(fig, output: Path, provenance: dict, inputs: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    for path in inputs:
        register_input(provenance, output.name, path)
    provenance[output.name]["outputs"] = [
        str(output),
        str(output.with_suffix(".pdf")),
    ]


def architecture_overview(output_dir: Path, provenance: dict) -> None:
    """Create a data-free schematic of the physical learning architecture."""
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    ax.set_xlim(-0.7, 9.8)
    ax.set_ylim(-1.4, 2.4)
    ax.axis("off")

    x = np.arange(8)
    y = np.zeros_like(x, dtype=float)
    ax.plot(x, y, linewidth=1.8)
    ax.scatter(x, y, s=420, zorder=3)
    for i in range(7):
        ax.text(i + 0.5, 0.18, "conservative", ha="center", va="bottom", fontsize=9)

    ax.text(0, -0.62, "input support", ha="center", fontsize=10)
    ax.annotate(
        "input",
        xy=(0, 0.15),
        xytext=(-0.4, 1.25),
        arrowprops={"arrowstyle": "->", "linewidth": 1.3},
        ha="center",
        fontsize=10,
    )

    ax.text(7, -0.62, "readout / damping support", ha="center", fontsize=10)
    for node in (5, 6, 7):
        ax.annotate(
            "",
            xy=(node, -0.05),
            xytext=(node, -0.95),
            arrowprops={"arrowstyle": "->", "linewidth": 1.4},
        )
    ax.text(6, -1.18, "localized dissipation", ha="center", fontsize=10)

    ax.text(3.5, 1.72, "free phase  $\\beta=0$", ha="center", fontsize=11)
    ax.text(5.6, 1.18, "$+\\beta$", ha="center", fontsize=11)
    ax.text(7.2, 1.18, "$-\\beta$", ha="center", fontsize=11)
    ax.annotate(
        "",
        xy=(5.35, 0.35),
        xytext=(5.6, 0.98),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
    )
    ax.annotate(
        "",
        xy=(7.0, 0.35),
        xytext=(7.2, 0.98),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
    )
    ax.text(
        8.55,
        0.35,
        "local centered\nparameter contrast",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.annotate(
        "",
        xy=(8.0, 0.2),
        xytext=(7.45, 0.2),
        arrowprops={"arrowstyle": "->", "linewidth": 1.3},
    )
    ax.set_title(
        "Conservative transport, localized relaxation, and centered EqProp update",
        fontsize=12,
    )

    output = output_dir / "figure_architecture_overview.png"
    finish_figure(fig, output, provenance, [])
    provenance[output.name]["note"] = "Data-free schematic generated directly by make_paper_figures.py"


def damping_support_figure(
    support_dir: Path, output_dir: Path, provenance: dict
) -> None:
    runs_path = support_dir / "runs.csv"
    targets_path = support_dir / "targets.csv"
    if not runs_path.exists() or not targets_path.exists():
        raise FileNotFoundError(
            f"Missing support inputs: {runs_path} or {targets_path}"
        )

    runs = pd.read_csv(runs_path)
    targets = pd.read_csv(targets_path)
    supports = sorted(runs["support_width"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    grouped = (
        runs.groupby(["support_width", "physical_time"], as_index=False)
        .agg(
            median_error=("dynamic_vs_exact_centered_relative_error", "median"),
            maximum_error=("dynamic_vs_exact_centered_relative_error", "max"),
        )
        .sort_values(["support_width", "physical_time"])
    )
    for width in supports:
        part = grouped[grouped["support_width"] == width]
        axes[0].loglog(
            part["physical_time"],
            100.0 * part["median_error"],
            marker="o",
            label=f"m={int(width)}",
        )
    axes[0].axhline(1.0, linestyle="--", linewidth=1.1)
    axes[0].set(
        xlabel="physical relaxation time",
        ylabel="median relaxation-gradient error (%)",
        title="Finite-time gradient fidelity",
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8, ncol=2)

    one = targets[
        np.isclose(targets["target_relative_error"].to_numpy(float), 0.01)
    ].copy()
    reached = one[one["target_reached"]].copy()
    first = (
        reached.groupby("support_width", as_index=False)
        .agg(median_first_time=("first_physical_time", "median"))
    )
    full_fraction = (
        one.groupby("support_width", as_index=False)
        .agg(reached_fraction=("target_reached", "mean"))
    )
    axes[1].plot(
        first["support_width"],
        first["median_first_time"],
        marker="o",
        linewidth=1.8,
    )
    for _, row in full_fraction.iterrows():
        if row["reached_fraction"] < 1.0:
            axes[1].annotate(
                f"{int(round(6*row['reached_fraction']))}/6",
                (row["support_width"], first.loc[
                    first["support_width"] == row["support_width"],
                    "median_first_time",
                ].iloc[0] if row["support_width"] in set(first["support_width"]) else 16000),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="number of damped nodes $m$",
        ylabel="median first time to <1% error",
        title="Relaxation resource to target",
    )
    axes[1].grid(alpha=0.25, which="both")

    spectral = (
        runs.groupby("support_width", as_index=False)
        .agg(
            minimum_visibility=("minimum_eigenspace_visibility", "min"),
            minimum_decay=("minimum_spectral_decay_rate", "min"),
        )
        .sort_values("support_width")
    )
    axes[2].semilogy(
        spectral["support_width"],
        spectral["minimum_decay"],
        marker="o",
        label="spectral decay rate",
    )
    twin = axes[2].twinx()
    twin.semilogy(
        spectral["support_width"],
        spectral["minimum_visibility"],
        marker="s",
        linestyle="--",
        label="modal visibility",
    )
    axes[2].set(
        xlabel="number of damped nodes $m$",
        ylabel="minimum spectral decay rate",
        title="Modal bottleneck",
    )
    twin.set_ylabel("minimum modal visibility")
    axes[2].grid(alpha=0.25, which="both")
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    output = output_dir / "figure_damping_support_resource.png"
    finish_figure(fig, output, provenance, [runs_path, targets_path])


def graph_placement_figure(
    graph_dir: Path, output_dir: Path, provenance: dict
) -> None:
    evaluation_path = graph_dir / "evaluation.csv"
    random_path = graph_dir / "random_spectral_pool.csv"
    if not evaluation_path.exists() or not random_path.exists():
        raise FileNotFoundError(
            f"Missing graph inputs: {evaluation_path} or {random_path}"
        )

    data = pd.read_csv(evaluation_path)
    random_pool = pd.read_csv(random_path)
    order = ["calibration_modal", "topology_only", "random_0"]
    labels = ["calibration-modal", "topology-only", "random control"]

    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.6))
    positions = np.arange(len(order))

    med_vis = [
        data.loc[data["strategy"] == strategy, "unseen_minimum_visibility"].median()
        for strategy in order
    ]
    axes[0].bar(positions, med_vis)
    axes[0].set_yscale("log")
    axes[0].set(
        ylabel="median unseen minimum visibility",
        title="Unseen modal visibility",
    )

    med_steps = [
        data.loc[data["strategy"] == strategy, "total_active_sample_steps"].median()
        for strategy in order
    ]
    axes[1].bar(positions, med_steps)
    axes[1].set_yscale("log")
    axes[1].set(
        ylabel="median active sample steps",
        title="Finite-time relaxation cost",
    )

    max_err = [
        data.loc[
            data["strategy"] == strategy,
            "dynamic_vs_exact_centered_gradient_error",
        ].max()
        for strategy in order
    ]
    axes[2].bar(positions, 100.0 * np.asarray(max_err))
    axes[2].set_yscale("log")
    axes[2].set(
        ylabel="maximum relaxation-gradient error (%)",
        title="Worst-case finite-time error",
    )

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=12)
        axis.grid(axis="y", alpha=0.25, which="both")

    output = output_dir / "figure_graph_placement_generalization.png"
    finish_figure(fig, output, provenance, [evaluation_path, random_path])

    # Record the random-pool percentile context without adding a fourth panel.
    percentiles = []
    for (topology, seed), block in data.groupby(["topology", "seed"]):
        modal = block[block["strategy"] == "calibration_modal"].iloc[0]
        pool = random_pool[
            (random_pool["topology"] == topology)
            & (random_pool["seed"] == seed)
        ]
        percentiles.append(
            float(
                np.mean(
                    pool["unseen_minimum_visibility"].to_numpy(float)
                    <= float(modal["unseen_minimum_visibility"])
                )
            )
        )
    provenance[output.name]["median_modal_visibility_random_percentile"] = float(
        np.median(percentiles)
    )


def modal_rate_figure(
    modal_dir: Path, output_dir: Path, provenance: dict
) -> None:
    audit_path = modal_dir / "audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    data = pd.read_csv(audit_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    variant_labels = {
        "paper_boundary": "boundary layer",
        "single_terminal": "endpoint only",
        "uniform_trace": "distributed, fixed trace",
    }
    for variant, part in data.groupby("variant"):
        grouped = (
            part.groupby("eta", as_index=False)
            .agg(
                exact=("exact_decay_rate", "median"),
                predicted=("first_order_predicted_decay_rate", "median"),
            )
            .sort_values("eta")
        )
        axes[0].loglog(
            grouped["eta"], grouped["exact"], marker="o", label=f"{variant_labels.get(variant, variant)}: exact"
        )
        axes[0].loglog(
            grouped["eta"],
            grouped["predicted"],
            linestyle="--",
            label=f"{variant_labels.get(variant, variant)}: first order",
        )
    axes[0].set(
        xlabel="damping scale $\\eta$",
        ylabel="slowest decay rate",
        title="Weak-damping modal law",
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=7)

    smallest = data.loc[
        data.groupby(["seed", "chain_size", "variant"])["eta"].idxmin()
    ].copy()
    for variant, part in smallest.groupby("variant"):
        grouped = (
            part.groupby("chain_size", as_index=False)
            .agg(coeff=("first_order_decay_coefficient", "median"))
            .sort_values("chain_size")
        )
        axes[1].loglog(
            grouped["chain_size"], grouped["coeff"], marker="o", label=variant_labels.get(variant, variant)
        )
    axes[1].set(
        xlabel="chain size",
        ylabel="slowest modal participation coefficient",
        title="Boundary participation bottleneck",
    )
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)

    output = output_dir / "figure_modal_relaxation.png"
    finish_figure(fig, output, provenance, [audit_path])


def finite_time_figure(
    finite_dir: Path, output_dir: Path, provenance: dict
) -> None:
    tolerance_path = finite_dir / "tolerance_sweep.csv"
    horizon_path = finite_dir / "horizon_sweep.csv"
    if not tolerance_path.exists() or not horizon_path.exists():
        raise FileNotFoundError(
            f"Missing finite-time inputs: {tolerance_path} or {horizon_path}"
        )
    tol = pd.read_csv(tolerance_path)
    horizon = pd.read_csv(horizon_path)

    valid = tol[
        np.isfinite(tol["residual_bound_relative"])
        & (tol["residual_bound_relative"] > 0)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    lower = max(
        1e-16,
        float(
            min(
                valid["dynamic_vs_implicit_relative_error"].min(),
                valid["residual_bound_relative"].min(),
            )
        ),
    )
    upper = float(
        max(
            valid["dynamic_vs_implicit_relative_error"].max(),
            valid["residual_bound_relative"].max(),
        )
    )
    axes[0].loglog(
        valid["residual_bound_relative"],
        valid["dynamic_vs_implicit_relative_error"],
        "o",
        alpha=0.7,
    )
    axes[0].loglog([lower, upper], [lower, upper], "k--", linewidth=1.0)
    axes[0].set(
        xlabel="decomposed a posteriori upper bound",
        ylabel="observed error vs implicit gradient",
        title="Observed error versus bound",
    )
    axes[0].grid(alpha=0.25, which="both")

    grouped = (
        tol.groupby("beta", as_index=False)
        .agg(
            total=("dynamic_vs_implicit_relative_error", "median"),
            finite_beta=("exact_truncation_relative_error", "median"),
            relaxation_bound=("residual_relaxation_bound_relative", "median"),
        )
        .sort_values("beta")
    )
    axes[1].loglog(grouped["beta"], grouped["total"], marker="o", label="total error")
    axes[1].loglog(
        grouped["beta"], grouped["finite_beta"], marker="o", label="finite-$\\beta$ bias"
    )
    axes[1].loglog(
        grouped["beta"],
        grouped["relaxation_bound"],
        marker="o",
        label="relaxation bound",
    )
    axes[1].set(
        xlabel="$\\beta$",
        ylabel="relative gradient error / bound",
        title="Finite-nudge and relaxation terms",
    )
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)

    output = output_dir / "figure_finite_time_bound_validation.png"
    finish_figure(fig, output, provenance, [tolerance_path, horizon_path])


def timestep_figure(
    timestep_dir: Path, output_dir: Path, provenance: dict
) -> None:
    input_path = timestep_dir / "audit.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    data = pd.read_csv(input_path)

    grouped = (
        data.groupby(["support_width", "dt"], as_index=False)
        .agg(
            median_relaxation_error=(
                "dynamic_vs_exact_centered_gradient_error", "median"
            ),
            max_gradient_difference=(
                "gradient_difference_vs_finest_dt", "max"
            ),
            max_energy_balance=(
                "maximum_relative_split_energy_balance_error", "max"
            ),
        )
        .sort_values(["support_width", "dt"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for width, part in grouped.groupby("support_width"):
        axes[0].loglog(
            part["dt"],
            100.0 * part["median_relaxation_error"],
            marker="o",
            label=f"m={int(width)}",
        )
        axes[1].loglog(
            part["dt"],
            part["max_energy_balance"],
            marker="o",
            label=f"m={int(width)}",
        )
    axes[0].set(
        xlabel="time step",
        ylabel="median relaxation-gradient error (%)",
        title="Fixed physical-time sensitivity",
    )
    axes[1].set(
        xlabel="time step",
        ylabel="maximum split energy-balance residual",
        title="Integration consistency",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
        axis.invert_xaxis()

    output = output_dir / "figure_timestep_refinement.png"
    finish_figure(fig, output, provenance, [input_path])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--modal-dir", type=Path, required=True)
    parser.add_argument("--finite-time-dir", type=Path, required=True)
    parser.add_argument("--timestep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance: dict[str, dict] = {
        "_generator": {
            "script": "make_paper_figures.py",
            "purpose": "Manuscript figure generation from locked numerical CSV outputs",
        }
    }
    architecture_overview(args.output_dir, provenance)
    damping_support_figure(args.support_dir, args.output_dir, provenance)
    graph_placement_figure(args.graph_dir, args.output_dir, provenance)
    modal_rate_figure(args.modal_dir, args.output_dir, provenance)
    finite_time_figure(args.finite_time_dir, args.output_dir, provenance)
    timestep_figure(args.timestep_dir, args.output_dir, provenance)

    provenance_path = args.output_dir / "figure_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
