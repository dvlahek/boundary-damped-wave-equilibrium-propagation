#!/usr/bin/env python3
"""Weak-damping modal-relaxation audit: weak-damping modal relaxation law.

For M=I and a simple conservative mode K v_j = omega_j^2 v_j, scaling the
damping matrix as eta D gives the first-order pole perturbation

    lambda_j^+/- = +/- i omega_j - eta (v_j^T D v_j)/(2 v_j^T v_j) + O(eta^2).

This audit compares the slowest first-order modal prediction with the exact
spectral abscissa of the first-order state matrix as eta -> 0.  It provides a
direct bridge between modal boundary participation and relaxation time without
claiming a universal chain-size power law.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib-bdw-modal-relaxation"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_rep_experiments import damping_vector
from boundary_rep_learning import solve_equilibrium, state_hessian
from theory_audit_core import (
    first_order_system_matrix,
    fit_loglog_slope,
    make_audit_problem,
)


PROFILES = {
    "quick": {
        "sizes": (5, 9, 17),
        "seeds": (17,),
        "etas": (1e-3, 3e-3, 1e-2, 3e-2),
    },
    "paper": {
        "sizes": (3, 5, 9, 17, 33),
        "seeds": (17, 29),
        "etas": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    },
}


def slowest_first_order_coefficient(
    hessian: np.ndarray, damping: np.ndarray
) -> tuple[float, float, int]:
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    # np.linalg.eigh returns Euclidean-normalized eigenvectors, so v^T v = 1.
    coefficients = 0.5 * np.sum(
        damping[:, None] * eigenvectors**2,
        axis=0,
    )
    index = int(np.argmin(coefficients))
    return (
        float(coefficients[index]),
        float(np.sqrt(eigenvalues[index])),
        index,
    )


def exact_decay_rate(hessian: np.ndarray, damping: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(
        first_order_system_matrix(hessian, damping)
    )
    spectral_abscissa = float(np.max(eigenvalues.real))
    return max(0.0, -spectral_abscissa)


def run(profile: str, output_dir: Path) -> dict[str, object]:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in settings["seeds"]:
        for size in settings["sizes"]:
            paper_config, paper_params, features, _ = make_audit_problem(
                int(size),
                int(seed),
                n_samples=4,
                boundary_variant="paper_boundary",
            )
            terminal_config, terminal_params, _, _ = make_audit_problem(
                int(size),
                int(seed),
                n_samples=4,
                boundary_variant="single_terminal",
            )
            q_free, _, residual = solve_equilibrium(
                paper_params, features, paper_config
            )
            hessian = state_hessian(
                q_free[:1], paper_params, paper_config
            )[0]

            variants = (
                (
                    "single_terminal",
                    terminal_config,
                    terminal_params,
                    "boundary",
                ),
                (
                    "paper_boundary",
                    paper_config,
                    paper_params,
                    "boundary",
                ),
                (
                    "uniform_trace",
                    paper_config,
                    paper_params,
                    "uniform_trace",
                ),
            )
            for variant, config, params, damping_mode in variants:
                base_damping = damping_vector(
                    config, damping_mode, params
                )
                coefficient, frequency, mode_index = (
                    slowest_first_order_coefficient(
                        hessian, base_damping
                    )
                )
                for eta in settings["etas"]:
                    eta = float(eta)
                    exact = exact_decay_rate(
                        hessian, eta * base_damping
                    )
                    predicted = eta * coefficient
                    relative = abs(exact - predicted) / max(
                        predicted, 1e-300
                    )
                    rows.append(
                        {
                            "profile": profile,
                            "seed": int(seed),
                            "chain_size": int(size),
                            "variant": variant,
                            "eta": eta,
                            "n_damped": int(
                                np.count_nonzero(base_damping)
                            ),
                            "base_damping_trace": float(
                                np.sum(base_damping)
                            ),
                            "slow_mode_index": mode_index,
                            "slow_mode_frequency": frequency,
                            "first_order_decay_coefficient": coefficient,
                            "first_order_predicted_decay_rate": predicted,
                            "exact_decay_rate": exact,
                            "relative_prediction_error": relative,
                            "exact_free_residual": float(residual),
                        }
                    )

    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "modal_relaxation_audit.csv",
        index=False,
    )

    fit_rows: list[dict[str, object]] = []
    for (seed, size, variant), part in table.groupby(
        ["seed", "chain_size", "variant"], sort=True
    ):
        slope, r2 = fit_loglog_slope(
            part["eta"].to_numpy(float),
            part["exact_decay_rate"].to_numpy(float),
        )
        smallest_eta = part.loc[part["eta"].idxmin()]
        fit_rows.append(
            {
                "seed": int(seed),
                "chain_size": int(size),
                "variant": variant,
                "exact_decay_vs_eta_slope": float(slope),
                "exact_decay_vs_eta_r_squared": float(r2),
                "smallest_eta": float(smallest_eta["eta"]),
                "smallest_eta_relative_prediction_error": float(
                    smallest_eta["relative_prediction_error"]
                ),
                "first_order_decay_coefficient": float(
                    smallest_eta["first_order_decay_coefficient"]
                ),
            }
        )
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(
        output_dir / "modal_relaxation_fits.csv",
        index=False,
    )

    smallest = table.loc[
        table.groupby(
            ["seed", "chain_size", "variant"]
        )["eta"].idxmin()
    ].copy()
    summary = {
        "version": "1.0",
        "profile": profile,
        "sizes": list(settings["sizes"]),
        "seeds": list(settings["seeds"]),
        "etas": list(settings["etas"]),
        "n_cases": int(len(table)),
        "median_exact_decay_vs_eta_slope": float(
            fits["exact_decay_vs_eta_slope"].median()
        ),
        "minimum_exact_decay_vs_eta_r_squared": float(
            fits["exact_decay_vs_eta_r_squared"].min()
        ),
        "median_smallest_eta_relative_prediction_error": float(
            smallest["relative_prediction_error"].median()
        ),
        "maximum_smallest_eta_relative_prediction_error": float(
            smallest["relative_prediction_error"].max()
        ),
    }
    (output_dir / "modal_relaxation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    for variant, part in table.groupby("variant"):
        grouped = part.groupby("eta", as_index=False).agg(
            exact=("exact_decay_rate", "median"),
            predicted=("first_order_predicted_decay_rate", "median"),
        )
        axes[0].loglog(
            grouped["eta"],
            grouped["exact"],
            marker="o",
            label=f"{variant}: exact",
        )
        axes[0].loglog(
            grouped["eta"],
            grouped["predicted"],
            linestyle="--",
            label=f"{variant}: first order",
        )

    for variant, part in smallest.groupby("variant"):
        grouped = part.groupby("chain_size", as_index=False).agg(
            coefficient=("first_order_decay_coefficient", "median")
        )
        axes[1].loglog(
            grouped["chain_size"],
            grouped["coefficient"],
            marker="o",
            label=variant,
        )

    axes[0].set(
        xlabel=r"damping scale $\eta$",
        ylabel="slowest decay rate",
        title="Weak-damping modal law",
    )
    axes[1].set(
        xlabel="chain size",
        ylabel="slowest first-order modal coefficient",
        title="Boundary participation bottleneck",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(
        output_dir / "modal_relaxation_audit.png",
        dpi=220,
    )
    plt.close(figure)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="quick"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/modal_relaxation_audit"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.profile, args.output_dir), indent=2))
