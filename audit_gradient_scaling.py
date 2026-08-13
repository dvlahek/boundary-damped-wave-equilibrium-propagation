#!/usr/bin/env python3
"""Audit REP v3 Theorem 2: centered EqProp has O(beta^2) bias."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v3-gradient"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from theory_audit_core import (
    exact_gradient_triplet,
    fit_loglog_slope,
    hessian_summary,
    make_audit_problem,
    write_json,
)


PROFILES = {
    "quick": {
        "sizes": (3, 5, 9),
        "seeds": (17,),
        "betas": (0.08, 0.04, 0.02, 0.01),
        "n_samples": 12,
    },
    "paper": {
        # Longer chains are handled by the separate scaling audit.  Beyond
        # n=17 this static drive is attenuated close to machine precision, so
        # a relative gradient ratio no longer measures centered-difference
        # order reliably.
        "sizes": (3, 5, 9, 17),
        "seeds": (17, 29, 43),
        "betas": (0.12, 0.08, 0.05, 0.035, 0.02, 0.01, 0.005, 0.0025),
        "n_samples": 24,
    },
}


def run(profile: str, output_dir: Path) -> dict[str, object]:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in settings["seeds"]:
        for chain_size in settings["sizes"]:
            config, params, features, targets = make_audit_problem(
                chain_size,
                seed,
                n_samples=settings["n_samples"],
                boundary_variant="paper_boundary",
            )
            for beta in settings["betas"]:
                result = exact_gradient_triplet(
                    params, features, targets, config, beta
                )
                plus_curvature = hessian_summary(
                    np.asarray(result["q_plus"]), params, config, beta
                )
                minus_curvature = hessian_summary(
                    np.asarray(result["q_minus"]), params, config, -beta
                )
                rows.append(
                    {
                        "profile": profile,
                        "seed": seed,
                        "chain_size": chain_size,
                        "beta": beta,
                        "relative_gradient_error": result["relative_error"],
                        "gradient_cosine_similarity": result["cosine"],
                        "implicit_gradient_norm": float(
                            np.linalg.norm(result["implicit_gradient"])
                        ),
                        "centered_gradient_norm": float(
                            np.linalg.norm(result["centered_gradient"])
                        ),
                        "minimum_plus_hessian_eigenvalue": plus_curvature[
                            "minimum_hessian_eigenvalue"
                        ],
                        "minimum_minus_hessian_eigenvalue": minus_curvature[
                            "minimum_hessian_eigenvalue"
                        ],
                        "free_residual": result["free_residual"],
                        "plus_residual": result["plus_residual"],
                        "minus_residual": result["minus_residual"],
                    }
                )
    table = pd.DataFrame(rows)
    slope_rows = []
    for (seed, chain_size), part in table.groupby(["seed", "chain_size"]):
        # Avoid the largest perturbation in the article profile.  The remaining
        # interval is the local centered-difference regime used by the theorem.
        fit_part = part[(part["beta"] <= 0.08) & (part["beta"] >= 0.01)]
        slope, r_squared = fit_loglog_slope(
            fit_part["beta"].to_numpy(float),
            fit_part["relative_gradient_error"].to_numpy(float),
        )
        slope_rows.append(
            {
                "seed": seed,
                "chain_size": chain_size,
                "n_fit_points": int(len(fit_part)),
                "loglog_slope": slope,
                "loglog_r_squared": r_squared,
                "minimum_relative_gradient_error": float(
                    part["relative_gradient_error"].min()
                ),
                "minimum_gradient_cosine": float(
                    part["gradient_cosine_similarity"].min()
                ),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    table.to_csv(output_dir / "gradient_beta_sweep.csv", index=False)
    slopes.to_csv(output_dir / "gradient_scaling_slopes.csv", index=False)

    summary = {
        "version": "3.0",
        "audit": "centered EqProp beta-squared scaling",
        "profile": profile,
        "n_cases": int(len(table)),
        "n_scaling_fits": int(len(slopes)),
        "median_loglog_slope": float(slopes["loglog_slope"].median()),
        "minimum_loglog_slope": float(slopes["loglog_slope"].min()),
        "maximum_loglog_slope": float(slopes["loglog_slope"].max()),
        "median_loglog_r_squared": float(slopes["loglog_r_squared"].median()),
        "maximum_relative_gradient_error": float(
            table["relative_gradient_error"].max()
        ),
        "maximum_finest_beta_relative_gradient_error": float(
            table.loc[
                table["beta"] == table["beta"].min(),
                "relative_gradient_error",
            ].max()
        ),
        "minimum_gradient_cosine": float(
            table["gradient_cosine_similarity"].min()
        ),
        "minimum_small_beta_gradient_cosine": float(
            table.loc[
                table["beta"] <= 0.02, "gradient_cosine_similarity"
            ].min()
        ),
        "minimum_finest_beta_gradient_cosine": float(
            table.loc[
                table["beta"] == table["beta"].min(),
                "gradient_cosine_similarity",
            ].min()
        ),
        "minimum_audited_hessian_eigenvalue": float(
            min(
                table["minimum_plus_hessian_eigenvalue"].min(),
                table["minimum_minus_hessian_eigenvalue"].min(),
            )
        ),
    }
    summary["passes_core_gate"] = bool(
        1.7 <= summary["median_loglog_slope"] <= 2.3
        and summary["minimum_finest_beta_gradient_cosine"] >= 0.999
        and summary["maximum_finest_beta_relative_gradient_error"] <= 0.02
        and summary["minimum_audited_hessian_eigenvalue"] > 0.0
        and summary["median_loglog_r_squared"] >= 0.98
    )
    write_json(output_dir / "gradient_scaling_summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    grouped = table.groupby(["chain_size", "beta"], as_index=False).agg(
        median_error=("relative_gradient_error", "median"),
        minimum_error=("relative_gradient_error", "min"),
        maximum_error=("relative_gradient_error", "max"),
    )
    for chain_size, part in grouped.groupby("chain_size"):
        axes[0].loglog(
            part["beta"],
            part["median_error"],
            marker="o",
            label=f"n={chain_size}",
        )
    beta_reference = np.asarray(sorted(set(table["beta"])))
    anchor = float(grouped["median_error"].median()) / float(
        np.median(beta_reference**2)
    )
    axes[0].loglog(
        beta_reference,
        anchor * beta_reference**2,
        "k--",
        label=r"reference $\beta^2$",
    )
    axes[0].set(
        xlabel=r"nudging strength $\beta$",
        ylabel="relative gradient error",
        title="Centered EqProp convergence",
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].scatter(
        slopes["chain_size"],
        slopes["loglog_slope"],
        c=slopes["seed"],
        cmap="viridis",
        s=55,
    )
    axes[1].axhline(2.0, color="black", linestyle="--", label="theory: slope 2")
    axes[1].set(
        xlabel="chain size",
        ylabel="fitted log-log slope",
        title="Observed truncation order",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_scaling_audit.png", dpi=190)
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_v3_gradient_scaling")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(run(arguments.profile, arguments.output_dir))
