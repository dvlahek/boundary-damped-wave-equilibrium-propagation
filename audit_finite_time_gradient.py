#!/usr/bin/env python3
"""Audit REP v3 Theorem 3: finite-time centered-gradient error.

The tolerance sweep and the fixed-horizon sweep are kept in separate CSV
files.  The script evaluates the proved decomposition into centered EqProp
truncation error plus endpoint-relaxation error and reports, but does not force,
the predicted one-third scaling of the empirically optimal beta.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v3-finite"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boundary_rep_learning import state_hessian
from theory_audit_core import (
    dynamic_centered_gradient,
    fit_loglog_slope,
    make_audit_problem,
    parameter_map_segment_lipschitz_bound,
    write_json,
)


PROFILES = {
    "quick": {
        "sizes": (5,),
        "seeds": (17,),
        "betas": (0.08, 0.04, 0.02, 0.01),
        "tolerances": (2e-3, 5e-4, 1e-4),
        "max_steps": 20_000,
        "time_steps": (100, 300, 900, 2700),
        "n_samples": 8,
    },
    "paper": {
        "sizes": (5, 9),
        "seeds": (17, 29),
        "betas": (0.12, 0.08, 0.05, 0.035, 0.02, 0.01, 0.005),
        "tolerances": (3e-3, 1e-3, 3e-4, 1e-4, 3e-5),
        "max_steps": 120_000,
        "time_steps": (100, 300, 900, 2700, 8100),
        "n_samples": 16,
    },
}


def theorem_row(
    result: dict[str, object],
    *,
    params,
    features: np.ndarray,
    config,
    beta: float,
) -> dict[str, object]:
    q_plus_dynamic = np.asarray(result["q_plus_dynamic"])
    q_minus_dynamic = np.asarray(result["q_minus_dynamic"])
    q_plus_exact = np.asarray(result["q_plus_exact"])
    q_minus_exact = np.asarray(result["q_minus_exact"])
    implicit = np.asarray(result["implicit_gradient"])
    centered = np.asarray(result["exact_centered_gradient"])
    dynamic = np.asarray(result["dynamic_gradient"])
    implicit_norm = max(float(np.linalg.norm(implicit)), 1e-14)

    plus_state_frobenius = float(np.linalg.norm(q_plus_dynamic - q_plus_exact))
    minus_state_frobenius = float(np.linalg.norm(q_minus_dynamic - q_minus_exact))
    plus_lipschitz = parameter_map_segment_lipschitz_bound(
        params, features, q_plus_dynamic, q_plus_exact
    )
    minus_lipschitz = parameter_map_segment_lipschitz_bound(
        params, features, q_minus_dynamic, q_minus_exact
    )
    truncation_absolute = float(np.linalg.norm(centered - implicit))
    dynamic_absolute = float(np.linalg.norm(dynamic - centered))
    total_absolute = float(np.linalg.norm(dynamic - implicit))
    state_bound_absolute = (
        plus_lipschitz * plus_state_frobenius
        + minus_lipschitz * minus_state_frobenius
    ) / (2.0 * abs(beta))
    total_state_bound_absolute = truncation_absolute + state_bound_absolute

    plus_mu = float(
        np.min(np.linalg.eigvalsh(state_hessian(q_plus_exact, params, config, beta)))
    )
    minus_mu = float(
        np.min(np.linalg.eigvalsh(state_hessian(q_minus_exact, params, config, -beta)))
    )
    mu = min(plus_mu, minus_mu)
    batch_factor = np.sqrt(features.shape[0])
    residual_bound_absolute = truncation_absolute + (
        plus_lipschitz * batch_factor * float(result["plus_residual"]) / mu
        + minus_lipschitz * batch_factor * float(result["minus_residual"]) / mu
    ) / (2.0 * abs(beta))
    return {
        "dynamic_vs_implicit_relative_error": total_absolute / implicit_norm,
        "dynamic_vs_centered_relative_error": dynamic_absolute / implicit_norm,
        "exact_truncation_relative_error": truncation_absolute / implicit_norm,
        "state_error_bound_relative": total_state_bound_absolute / implicit_norm,
        "residual_bound_relative": residual_bound_absolute / implicit_norm,
        "state_error_bound_holds": bool(
            total_absolute <= total_state_bound_absolute + 1e-10
        ),
        "residual_bound_holds": bool(
            total_absolute <= residual_bound_absolute + 1e-10
        ),
        "plus_state_frobenius_error": plus_state_frobenius,
        "minus_state_frobenius_error": minus_state_frobenius,
        "plus_parameter_map_lipschitz": plus_lipschitz,
        "minus_parameter_map_lipschitz": minus_lipschitz,
        "minimum_endpoint_curvature": mu,
        "achieved_max_residual": max(
            float(result["plus_residual"]), float(result["minus_residual"])
        ),
        "achieved_max_velocity": max(
            float(result["plus_velocity_norm"]),
            float(result["minus_velocity_norm"]),
        ),
    }


def run(profile: str, output_dir: Path) -> dict[str, object]:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerance_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []

    for seed in settings["seeds"]:
        for chain_size in settings["sizes"]:
            config, params, features, targets = make_audit_problem(
                chain_size,
                seed,
                n_samples=settings["n_samples"],
                boundary_variant="paper_boundary",
            )
            for tolerance in settings["tolerances"]:
                for beta in settings["betas"]:
                    result = dynamic_centered_gradient(
                        params,
                        features,
                        targets,
                        config,
                        beta,
                        tolerance=tolerance,
                        max_steps=settings["max_steps"],
                        dt=0.2,
                    )
                    tolerance_rows.append(
                        {
                            "profile": profile,
                            "seed": seed,
                            "chain_size": chain_size,
                            "beta": beta,
                            "requested_tolerance": tolerance,
                            "plus_converged": result["plus_converged"],
                            "minus_converged": result["minus_converged"],
                            "both_converged": bool(
                                result["plus_converged"]
                                and result["minus_converged"]
                            ),
                            "plus_steps": result["plus_steps"],
                            "minus_steps": result["minus_steps"],
                            "gradient_cosine": result[
                                "dynamic_vs_implicit_cosine"
                            ],
                            **theorem_row(
                                result,
                                params=params,
                                features=features,
                                config=config,
                                beta=beta,
                            ),
                        }
                    )

            fixed_beta = 0.035
            for steps in settings["time_steps"]:
                result = dynamic_centered_gradient(
                    params,
                    features,
                    targets,
                    config,
                    fixed_beta,
                    tolerance=0.0,
                    max_steps=steps,
                    dt=0.2,
                    force_full_steps=True,
                )
                time_rows.append(
                    {
                        "profile": profile,
                        "seed": seed,
                        "chain_size": chain_size,
                        "beta": fixed_beta,
                        "steps": steps,
                        "physical_time": 0.2 * steps,
                        **theorem_row(
                            result,
                            params=params,
                            features=features,
                            config=config,
                            beta=fixed_beta,
                        ),
                    }
                )

    tolerance_table = pd.DataFrame(tolerance_rows)
    time_table = pd.DataFrame(time_rows)
    tolerance_table.to_csv(
        output_dir / "finite_time_tolerance_sweep.csv", index=False
    )
    time_table.to_csv(output_dir / "finite_time_horizon_sweep.csv", index=False)

    optimum_rows = []
    for (seed, chain_size, tolerance), part in tolerance_table.groupby(
        ["seed", "chain_size", "requested_tolerance"]
    ):
        valid = part[part["both_converged"]]
        if valid.empty:
            continue
        best = valid.loc[valid["dynamic_vs_implicit_relative_error"].idxmin()]
        optimum_rows.append(
            {
                "seed": seed,
                "chain_size": chain_size,
                "requested_tolerance": tolerance,
                "best_beta": float(best["beta"]),
                "best_relative_gradient_error": float(
                    best["dynamic_vs_implicit_relative_error"]
                ),
                "achieved_residual_at_best": float(
                    best["achieved_max_residual"]
                ),
            }
        )
    optima = pd.DataFrame(optimum_rows)
    optima.to_csv(output_dir / "finite_time_beta_optima.csv", index=False)
    if len(optima) >= 2:
        beta_tolerance_slope, beta_tolerance_r2 = fit_loglog_slope(
            optima["requested_tolerance"].to_numpy(float),
            optima["best_beta"].to_numpy(float),
        )
        beta_residual_slope, beta_residual_r2 = fit_loglog_slope(
            optima["achieved_residual_at_best"].to_numpy(float),
            optima["best_beta"].to_numpy(float),
        )
    else:
        beta_tolerance_slope = beta_tolerance_r2 = float("nan")
        beta_residual_slope = beta_residual_r2 = float("nan")

    convergence_slopes = []
    for (seed, chain_size), part in time_table.groupby(["seed", "chain_size"]):
        time_values = part["physical_time"].to_numpy(float)
        error_values = (
            part["plus_state_frobenius_error"].to_numpy(float)
            + part["minus_state_frobenius_error"].to_numpy(float)
        )
        mask = np.isfinite(error_values) & (error_values > 1e-14)
        if np.count_nonzero(mask) >= 2:
            slope, intercept = np.polyfit(time_values[mask], np.log(error_values[mask]), 1)
            prediction = slope * time_values[mask] + intercept
            observed = np.log(error_values[mask])
            denominator = float(np.sum((observed - np.mean(observed)) ** 2))
            r_squared = 1.0 - float(np.sum((observed - prediction) ** 2)) / max(denominator, 1e-30)
        else:
            slope = r_squared = float("nan")
        convergence_slopes.append(
            {
                "seed": seed,
                "chain_size": chain_size,
                "fitted_exponential_rate": -slope,
                "fit_r_squared": r_squared,
            }
        )
    convergence_table = pd.DataFrame(convergence_slopes)
    convergence_table.to_csv(
        output_dir / "finite_time_convergence_rates.csv", index=False
    )

    converged_fraction = float(tolerance_table["both_converged"].mean())
    summary = {
        "version": "3.0",
        "audit": "finite-time centered-gradient error",
        "profile": profile,
        "n_tolerance_cases": int(len(tolerance_table)),
        "n_fixed_horizon_cases": int(len(time_table)),
        "converged_fraction": converged_fraction,
        "all_state_error_bounds_hold": bool(
            tolerance_table["state_error_bound_holds"].all()
            and time_table["state_error_bound_holds"].all()
        ),
        "all_residual_bounds_hold_for_converged_cases": bool(
            tolerance_table.loc[
                tolerance_table["both_converged"], "residual_bound_holds"
            ].all()
        ),
        "maximum_converged_relative_gradient_error": float(
            tolerance_table.loc[
                tolerance_table["both_converged"],
                "dynamic_vs_implicit_relative_error",
            ].max()
        )
        if tolerance_table["both_converged"].any()
        else float("nan"),
        "beta_optimum_vs_tolerance_loglog_slope": beta_tolerance_slope,
        "beta_optimum_vs_tolerance_r_squared": beta_tolerance_r2,
        "beta_optimum_vs_achieved_residual_loglog_slope": beta_residual_slope,
        "beta_optimum_vs_achieved_residual_r_squared": beta_residual_r2,
        "theoretical_optimum_slope": 1.0 / 3.0,
        "median_fitted_exponential_rate": float(
            convergence_table["fitted_exponential_rate"].median()
        ),
    }
    # The one-third law is reported as a scientific result, not used as a
    # software-completion gate.  A coarse beta grid can quantize the optimum.
    summary["passes_core_gate"] = bool(
        converged_fraction >= 0.95
        and summary["all_state_error_bounds_hold"]
        and summary["all_residual_bounds_hold_for_converged_cases"]
    )
    write_json(output_dir / "finite_time_summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    first_seed = settings["seeds"][0]
    first_size = settings["sizes"][0]
    heat = tolerance_table[
        (tolerance_table["seed"] == first_seed)
        & (tolerance_table["chain_size"] == first_size)
    ].pivot(
        index="requested_tolerance",
        columns="beta",
        values="dynamic_vs_implicit_relative_error",
    )
    image = axes[0].imshow(
        np.log10(np.maximum(heat.to_numpy(), 1e-16)),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[0].set_xticks(range(len(heat.columns)), [f"{value:g}" for value in heat.columns], rotation=45)
    axes[0].set_yticks(range(len(heat.index)), [f"{value:.0e}" for value in heat.index])
    axes[0].set(xlabel=r"$\beta$", ylabel="endpoint tolerance", title="log10 gradient error")
    fig.colorbar(image, ax=axes[0])

    if not optima.empty:
        axes[1].loglog(
            optima["requested_tolerance"],
            optima["best_beta"],
            "o",
            label="empirical optimum",
        )
        reference_x = np.asarray(sorted(optima["requested_tolerance"].unique()))
        anchor = float(np.median(optima["best_beta"] / optima["requested_tolerance"] ** (1.0 / 3.0)))
        axes[1].loglog(reference_x, anchor * reference_x ** (1.0 / 3.0), "k--", label=r"reference $\tau^{1/3}$")
    axes[1].set(xlabel="endpoint tolerance", ylabel=r"best $\beta$", title="Finite-time optimum")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)

    for (seed, chain_size), part in time_table.groupby(["seed", "chain_size"]):
        axes[2].semilogy(
            part["physical_time"],
            part["plus_state_frobenius_error"] + part["minus_state_frobenius_error"],
            marker="o",
            label=f"n={chain_size}, seed={seed}",
        )
    axes[2].set(xlabel="physical relaxation time", ylabel="summed endpoint state error", title="Finite-time relaxation")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "finite_time_gradient_audit.png", dpi=190)
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_v3_finite_time")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(run(arguments.profile, arguments.output_dir))
