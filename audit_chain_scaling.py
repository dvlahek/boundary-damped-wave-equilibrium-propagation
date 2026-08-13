#!/usr/bin/env python3
"""Audit boundary-damping resource and convergence scaling with chain size."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v3-scaling"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_rep_experiments import DynamicsConfig, damping_vector, relax_dynamics
from boundary_rep_learning import solve_equilibrium, state_hessian
from theory_audit_core import (
    fit_loglog_slope,
    make_audit_problem,
    modal_observability,
    write_json,
)


PROFILES = {
    "quick": {
        "sizes": (3, 5, 9, 17),
        "seeds": (17,),
        "n_samples": 4,
        "tolerance": 5e-4,
        "max_steps": 30_000,
        "dynamic_max_size": 17,
    },
    "paper": {
        "sizes": (3, 5, 9, 17, 33, 65, 129),
        "seeds": (17, 29),
        "n_samples": 8,
        "tolerance": 1e-5,
        "max_steps": 120_000,
        "dynamic_max_size": 65,
    },
}


def run(profile: str, output_dir: Path, skip_dynamics: bool = False) -> dict[str, object]:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in settings["seeds"]:
        for chain_size in settings["sizes"]:
            paper_config, paper_params, features, _ = make_audit_problem(
                chain_size,
                seed,
                n_samples=settings["n_samples"],
                boundary_variant="paper_boundary",
            )
            terminal_config, terminal_params, _, _ = make_audit_problem(
                chain_size,
                seed,
                n_samples=settings["n_samples"],
                boundary_variant="single_terminal",
            )
            q_equilibrium, _, exact_residual = solve_equilibrium(
                paper_params, features, paper_config
            )
            hessian = state_hessian(
                q_equilibrium[:1], paper_params, paper_config
            )[0]
            minimum_curvature = float(
                np.min(np.linalg.eigvalsh(state_hessian(q_equilibrium, paper_params, paper_config)))
            )
            variants = (
                ("single_terminal", terminal_config, terminal_params, "boundary"),
                ("paper_boundary", paper_config, paper_params, "boundary"),
                ("uniform_trace", paper_config, paper_params, "uniform_trace"),
                ("uniform_local", paper_config, paper_params, "uniform_local"),
            )
            for variant, config, params, damping_mode in variants:
                damping = damping_vector(config, damping_mode, params)
                modal = modal_observability(hessian, damping)
                dynamic_requested = bool(
                    not skip_dynamics and chain_size <= settings["dynamic_max_size"]
                )
                dynamic_converged = False
                steps = np.nan
                active_sample_steps = np.nan
                residual = np.nan
                velocity_norm = np.nan
                state_error = np.nan
                if dynamic_requested:
                    dynamic = relax_dynamics(
                        params,
                        features,
                        config,
                        DynamicsConfig(
                            dt=0.2,
                            max_steps=settings["max_steps"],
                            gradient_tolerance=settings["tolerance"],
                            velocity_tolerance=settings["tolerance"],
                        ),
                        damping_mode=damping_mode,
                    )
                    dynamic_converged = bool(dynamic["converged"])
                    steps = int(dynamic["steps"])
                    active_sample_steps = int(dynamic["active_sample_steps"])
                    residual = float(dynamic["residual"])
                    velocity_norm = float(dynamic["velocity_norm"])
                    state_error = float(
                        np.max(
                            np.linalg.norm(
                                np.asarray(dynamic["q"]) - q_equilibrium,
                                axis=1,
                            )
                        )
                    )
                rows.append(
                    {
                        "profile": profile,
                        "seed": seed,
                        "chain_size": chain_size,
                        "variant": variant,
                        "n_damped": modal["n_damped"],
                        "damped_fraction": modal["n_damped"] / chain_size,
                        "damping_trace": modal["damping_trace"],
                        "minimum_eigenspace_visibility": modal[
                            "minimum_eigenspace_visibility"
                        ],
                        "observable": modal["observable"],
                        "linearly_exponentially_stable": modal[
                            "linearly_exponentially_stable"
                        ],
                        "spectral_abscissa": modal["spectral_abscissa"],
                        "spectral_decay_rate": modal["spectral_decay_rate"],
                        "predicted_e_folding_time": 1.0
                        / max(float(modal["spectral_decay_rate"]), 1e-300),
                        "minimum_hessian_eigenvalue": minimum_curvature,
                        "exact_equilibrium_residual": exact_residual,
                        "dynamic_requested": dynamic_requested,
                        "dynamic_converged": dynamic_converged,
                        "dynamic_steps": steps,
                        "active_sample_steps": active_sample_steps,
                        "dynamic_residual": residual,
                        "dynamic_velocity_norm": velocity_norm,
                        "dynamic_state_error": state_error,
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "chain_scaling.csv", index=False)

    slope_rows = []
    for variant, part in table.groupby("variant"):
        grouped = part.groupby("chain_size", as_index=False).agg(
            median_decay_rate=("spectral_decay_rate", "median"),
            median_e_folding_time=("predicted_e_folding_time", "median"),
            median_damped_nodes=("n_damped", "median"),
        )
        decay_slope, decay_r2 = fit_loglog_slope(
            grouped["chain_size"].to_numpy(float),
            grouped["median_decay_rate"].to_numpy(float),
        )
        resource_slope, resource_r2 = fit_loglog_slope(
            grouped["chain_size"].to_numpy(float),
            grouped["median_damped_nodes"].to_numpy(float),
        )
        slope_rows.append(
            {
                "variant": variant,
                "decay_rate_vs_size_slope": decay_slope,
                "decay_rate_fit_r_squared": decay_r2,
                "damped_nodes_vs_size_slope": resource_slope,
                "damped_nodes_fit_r_squared": resource_r2,
            }
        )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(output_dir / "chain_scaling_slopes.csv", index=False)

    dynamic_cases = table[table["dynamic_requested"]]
    summary = {
        "version": "3.0",
        "audit": "chain-size and damping-resource scaling",
        "profile": profile,
        "n_spectral_cases": int(len(table)),
        "n_dynamic_cases": int(len(dynamic_cases)),
        "all_hessians_positive": bool(
            (table["minimum_hessian_eigenvalue"] > 0.0).all()
        ),
        "all_spectral_cases_observable": bool(table["observable"].all()),
        "all_spectral_cases_exponentially_stable": bool(
            table["linearly_exponentially_stable"].all()
        ),
        "dynamic_convergence_fraction": float(
            dynamic_cases["dynamic_converged"].mean()
        )
        if not dynamic_cases.empty
        else None,
        "single_terminal_uses_one_damper_for_every_size": bool(
            (
                table.loc[table["variant"] == "single_terminal", "n_damped"]
                == 1
            ).all()
        ),
        "uniform_controls_use_n_dampers": bool(
            (
                table.loc[table["variant"] == "uniform_trace", "n_damped"].to_numpy()
                == table.loc[table["variant"] == "uniform_trace", "chain_size"].to_numpy()
            ).all()
        ),
        "largest_spectrally_audited_chain": int(table["chain_size"].max()),
        "largest_dynamically_audited_chain": int(
            dynamic_cases["chain_size"].max()
        )
        if not dynamic_cases.empty
        else None,
        "slopes": slopes.to_dict(orient="records"),
    }
    summary["passes_core_gate"] = bool(
        summary["all_hessians_positive"]
        and summary["all_spectral_cases_observable"]
        and summary["all_spectral_cases_exponentially_stable"]
        and summary["single_terminal_uses_one_damper_for_every_size"]
        and summary["uniform_controls_use_n_dampers"]
        and (
            skip_dynamics
            or summary["dynamic_convergence_fraction"] is not None
            and summary["dynamic_convergence_fraction"] >= 0.75
        )
    )
    write_json(output_dir / "chain_scaling_summary.json", summary)

    grouped = table.groupby(["chain_size", "variant"], as_index=False).agg(
        decay_rate=("spectral_decay_rate", "median"),
        damped_nodes=("n_damped", "median"),
        dynamic_steps=("dynamic_steps", "median"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    for variant, part in grouped.groupby("variant"):
        axes[0].loglog(
            part["chain_size"], part["decay_rate"], marker="o", label=variant
        )
        axes[1].loglog(
            part["chain_size"], part["damped_nodes"], marker="o", label=variant
        )
        dynamic_part = part.dropna(subset=["dynamic_steps"])
        if not dynamic_part.empty:
            axes[2].loglog(
                dynamic_part["chain_size"],
                dynamic_part["dynamic_steps"],
                marker="o",
                label=variant,
            )
    axes[0].set(xlabel="chain size", ylabel="spectral decay rate", title="Convergence-rate scaling")
    axes[1].set(xlabel="chain size", ylabel="number of damped nodes", title="Damping-resource scaling")
    axes[2].set(xlabel="chain size", ylabel="dynamic steps", title="Measured relaxation work")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "chain_scaling_audit.png", dpi=190)
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--skip-dynamics", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_v3_chain_scaling")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(run(arguments.profile, arguments.output_dir, arguments.skip_dynamics))
