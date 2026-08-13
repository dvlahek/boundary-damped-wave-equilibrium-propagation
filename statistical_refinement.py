#!/usr/bin/env python3
"""Block-aware equivalence analysis for the existing synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v5-statistics")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon


REFERENCES = ("implicit_id", "standard_eqprop", "uniform_dynamic")
KEYS = ["dataset", "seed", "chain_size"]
BLOCK_KEYS = ["dataset", "seed"]


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def tost_from_blocks(values: np.ndarray, margin: float) -> dict[str, float | bool]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1)) if n > 1 else 0.0
    standard_error = standard_deviation / np.sqrt(n) if n else np.nan
    if standard_error <= np.finfo(float).eps:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        ci_low = ci_high = mean
    else:
        degrees = n - 1
        lower_statistic = (mean + margin) / standard_error
        upper_statistic = (mean - margin) / standard_error
        p_lower = float(t.sf(lower_statistic, degrees))
        p_upper = float(t.cdf(upper_statistic, degrees))
        critical = float(t.ppf(0.95, degrees))
        ci_low = mean - critical * standard_error
        ci_high = mean + critical * standard_error
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": float(standard_error),
        "tost_p_lower": p_lower,
        "tost_p_upper": p_upper,
        "tost_p_value": max(p_lower, p_upper),
        "tost_equivalent_unadjusted": bool(max(p_lower, p_upper) < 0.05),
        "tost_90_ci_low": float(ci_low),
        "tost_90_ci_high": float(ci_high),
    }


def cluster_bootstrap(
    paired: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Resample dataset-seed blocks and retain all chain sizes inside a block."""
    # Every registered block contains the same four chain sizes.  Averaging
    # within a block first is therefore exactly equivalent to concatenating
    # the selected complete blocks, and permits a deterministic vectorized
    # bootstrap instead of 50,000 DataFrame concatenations.
    block_means = (
        paired.groupby(BLOCK_KEYS, sort=True, observed=True)["difference"]
        .mean()
        .to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    selected = rng.integers(
        0, len(block_means), size=(n_bootstrap, len(block_means))
    )
    return np.mean(block_means[selected], axis=1)


def analyze(
    input_csv: Path,
    output_dir: Path,
    *,
    margin: float = 0.02,
    n_bootstrap: int = 50_000,
    seed: int = 20260813,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(input_csv)
    required = set(KEYS + ["method", "test_accuracy", "wall_time_seconds"])
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"input CSV is missing columns: {missing}")
    duplicate = runs.duplicated(KEYS + ["method"], keep=False)
    if duplicate.any():
        raise ValueError("duplicate dataset-seed-chain-method rows were found")
    pivot_accuracy = runs.pivot(index=KEYS, columns="method", values="test_accuracy")
    pivot_time = runs.pivot(index=KEYS, columns="method", values="wall_time_seconds")
    expected_methods = {"boundary_dynamic", *REFERENCES}
    if not expected_methods.issubset(pivot_accuracy.columns):
        raise ValueError("not all four benchmark methods are present")
    if pivot_accuracy[list(expected_methods)].isna().any().any():
        raise ValueError("the benchmark is not complete and paired")

    detail_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    tost_p_values: list[float] = []
    for reference_index, reference in enumerate(REFERENCES):
        paired = pivot_accuracy.reset_index()[KEYS].copy()
        paired["boundary_accuracy"] = pivot_accuracy["boundary_dynamic"].to_numpy()
        paired["reference_accuracy"] = pivot_accuracy[reference].to_numpy()
        paired["difference"] = paired["boundary_accuracy"] - paired["reference_accuracy"]
        paired["reference"] = reference
        detail_rows.append(paired)

        block = (
            paired.groupby(BLOCK_KEYS, as_index=False, observed=True)["difference"]
            .mean()
            .rename(columns={"difference": "block_mean_difference"})
        )
        tost = tost_from_blocks(block["block_mean_difference"].to_numpy(), margin)
        tost_p_values.append(float(tost["tost_p_value"]))
        bootstrap = cluster_bootstrap(
            paired,
            n_bootstrap=n_bootstrap,
            seed=seed + 1009 * reference_index,
        )
        try:
            wilcoxon_result = wilcoxon(
                block["block_mean_difference"].to_numpy(),
                alternative="two-sided",
                zero_method="zsplit",
                method="approx",
            )
            wilcoxon_p = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        time_ratio = (
            pivot_time["boundary_dynamic"] / pivot_time[reference]
        ).replace([np.inf, -np.inf], np.nan)
        summary_rows.append(
            {
                "reference": reference,
                "n_paired_configurations": int(len(paired)),
                "n_dataset_seed_blocks": int(len(block)),
                "equivalence_margin_accuracy": margin,
                "mean_accuracy_difference": float(paired["difference"].mean()),
                "mean_accuracy_difference_percentage_points": float(
                    100.0 * paired["difference"].mean()
                ),
                "cluster_bootstrap_95_ci_low": float(
                    np.quantile(bootstrap, 0.025)
                ),
                "cluster_bootstrap_95_ci_high": float(
                    np.quantile(bootstrap, 0.975)
                ),
                "cluster_bootstrap_95_ci_low_percentage_points": float(
                    100.0 * np.quantile(bootstrap, 0.025)
                ),
                "cluster_bootstrap_95_ci_high_percentage_points": float(
                    100.0 * np.quantile(bootstrap, 0.975)
                ),
                "bootstrap_ci_inside_equivalence_margin": bool(
                    np.quantile(bootstrap, 0.025) > -margin
                    and np.quantile(bootstrap, 0.975) < margin
                ),
                **tost,
                "wilcoxon_block_p_value": wilcoxon_p,
                "median_boundary_to_reference_wall_time_ratio": float(
                    np.nanmedian(time_ratio)
                ),
            }
        )

    tost_adjusted = holm_adjust(tost_p_values)
    wilcoxon_adjusted = holm_adjust(
        [float(row["wilcoxon_block_p_value"]) for row in summary_rows]
    )
    for row, adjusted_tost, adjusted_wilcoxon in zip(
        summary_rows, tost_adjusted, wilcoxon_adjusted
    ):
        row["tost_p_value_holm"] = adjusted_tost
        row["tost_equivalent_holm_0_05"] = bool(adjusted_tost < 0.05)
        row["wilcoxon_block_p_value_holm"] = adjusted_wilcoxon

    detailed = pd.concat(detail_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    detailed.to_csv(output_dir / "paired_accuracy_differences.csv", index=False)
    summary.to_csv(output_dir / "block_equivalence_summary.csv", index=False)
    with (output_dir / "statistical_refinement_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_file": str(input_csv),
                "analysis_seed": seed,
                "cluster_bootstrap_replicates": n_bootstrap,
                "block_definition": "dataset and seed; all chain sizes retained within a block",
                "number_of_runs": int(len(runs)),
                "number_of_paired_configurations": int(len(pivot_accuracy)),
                "number_of_dataset_seed_blocks": int(
                    runs[BLOCK_KEYS].drop_duplicates().shape[0]
                ),
                "equivalence_margin_accuracy": margin,
                "comparisons": summary.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(summary))
    mean = summary["mean_accuracy_difference_percentage_points"].to_numpy()
    low = summary["cluster_bootstrap_95_ci_low_percentage_points"].to_numpy()
    high = summary["cluster_bootstrap_95_ci_high_percentage_points"].to_numpy()
    axis.errorbar(
        mean,
        y,
        xerr=np.vstack((mean - low, high - mean)),
        fmt="o",
        capsize=5,
        color="#1f77b4",
    )
    axis.axvline(-100.0 * margin, color="#b22222", linestyle="--")
    axis.axvline(100.0 * margin, color="#b22222", linestyle="--")
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(y, summary["reference"])
    axis.set_xlabel("boundary minus reference accuracy [percentage points]")
    axis.set_title("Dataset-seed cluster bootstrap intervals")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "statistical_equivalence.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("input/benchmark_runs.csv"))
    parser.add_argument("--output", type=Path, default=Path("results_statistical_refinement"))
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    summary = analyze(
        args.input,
        args.output,
        margin=args.margin,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
