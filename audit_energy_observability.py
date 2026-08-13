#!/usr/bin/env python3
"""Audit REP v3 Theorem 1: energy balance and boundary observability.

Outputs are isolated in their own directory.  The script checks both the
calibrated v2.3 boundary region and the theorem-focused single-terminal
damper.  A deliberately blind center-damped chain is included as a negative
control rather than being mixed with learning failures.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v3-energy"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from advanced_rep_experiments import damping_vector
from boundary_rep_learning import solve_equilibrium, state_hessian
from theory_audit_core import (
    hessian_summary,
    integrate_energy_trajectory,
    make_audit_problem,
    modal_observability,
    write_json,
)


PROFILES = {
    "quick": {
        "sizes": (3, 5, 9),
        "seeds": (17,),
        "betas": (0.0, 0.035, -0.035),
        "t_end": 45.0,
        "n_times": 241,
        "n_samples": 10,
    },
    "paper": {
        "sizes": (3, 5, 9, 17, 33, 65),
        "seeds": (17, 29),
        "betas": (0.0, 0.035, -0.035),
        "t_end": 100.0,
        "n_times": 501,
        "n_samples": 16,
    },
}


def blind_mode_control(t_end: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three-node symmetric chain with an antisymmetric center-blind mode."""
    laplacian = np.asarray(
        [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]
    )
    stiffness = 0.4 * np.eye(3) + 1.2 * laplacian
    initial = np.asarray([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    times = np.linspace(0.0, t_end, 501)
    rows = []
    trajectories = []
    for name, damping in (
        ("blind_center", np.asarray([0.0, 0.7, 0.0])),
        ("visible_endpoint", np.asarray([0.7, 0.0, 0.0])),
    ):
        modal = modal_observability(stiffness, damping)

        def rhs(_: float, state: np.ndarray) -> np.ndarray:
            q = state[:3]
            velocity = state[3:]
            return np.concatenate((velocity, -stiffness @ q - damping * velocity))

        solution = solve_ivp(
            rhs,
            (0.0, t_end),
            np.concatenate((initial, np.zeros(3))),
            t_eval=times,
            method="DOP853",
            rtol=2e-10,
            atol=2e-12,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        q = solution.y[:3].T
        velocity = solution.y[3:].T
        energy = 0.5 * np.sum(velocity**2, axis=1) + 0.5 * np.einsum(
            "bi,ij,bj->b", q, stiffness, q
        )
        fraction = energy / energy[0]
        rows.append(
            {
                "control": name,
                "observable": modal["observable"],
                "minimum_eigenspace_visibility": modal[
                    "minimum_eigenspace_visibility"
                ],
                "spectral_abscissa": modal["spectral_abscissa"],
                "spectral_decay_rate": modal["spectral_decay_rate"],
                "final_energy_fraction": float(fraction[-1]),
            }
        )
        trajectories.extend(
            {
                "control": name,
                "time": float(time),
                "energy_fraction": float(value),
            }
            for time, value in zip(times, fraction, strict=True)
        )
    return pd.DataFrame(rows), pd.DataFrame(trajectories)


def run(profile: str, output_dir: Path) -> dict[str, object]:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)
    modal_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    for seed in settings["seeds"]:
        for chain_size in settings["sizes"]:
            for variant in ("single_terminal", "paper_boundary"):
                config, params, features, targets = make_audit_problem(
                    chain_size,
                    seed,
                    n_samples=settings["n_samples"],
                    beta=0.035,
                    boundary_variant=variant,
                )
                for beta in settings["betas"]:
                    target_arg = None if beta == 0.0 else targets[:1]
                    q_equilibrium, _, residual = solve_equilibrium(
                        params,
                        features[:1],
                        config,
                        target_arg,
                        beta,
                    )
                    hessian = state_hessian(
                        q_equilibrium, params, config, beta
                    )[0]
                    damping = damping_vector(config, "boundary", params)
                    modal = modal_observability(hessian, damping)
                    curvature = hessian_summary(
                        q_equilibrium, params, config, beta
                    )
                    modal_rows.append(
                        {
                            "profile": profile,
                            "seed": seed,
                            "chain_size": chain_size,
                            "boundary_variant": variant,
                            "beta": beta,
                            "equilibrium_residual": residual,
                            **curvature,
                            "n_damped": modal["n_damped"],
                            "damping_trace": modal["damping_trace"],
                            "minimum_visibility": modal["minimum_visibility"],
                            "minimum_eigenspace_visibility": modal[
                                "minimum_eigenspace_visibility"
                            ],
                            "minimum_damping_visibility": modal[
                                "minimum_damping_visibility"
                            ],
                            "observable": modal["observable"],
                            "spectral_abscissa": modal["spectral_abscissa"],
                            "spectral_decay_rate": modal[
                                "spectral_decay_rate"
                            ],
                            "linearly_exponentially_stable": modal[
                                "linearly_exponentially_stable"
                            ],
                        }
                    )

                    # Energy integration is repeated for both boundary choices
                    # so locality and numerical balance can be inspected apart.
                    trajectory = integrate_energy_trajectory(
                        params,
                        features[:1],
                        config,
                        targets=target_arg,
                        beta=beta,
                        t_end=settings["t_end"],
                        n_times=settings["n_times"],
                    )
                    energy_rows.append(
                        {
                            "profile": profile,
                            "seed": seed,
                            "chain_size": chain_size,
                            "boundary_variant": variant,
                            "beta": beta,
                            "n_damped": int(np.count_nonzero(trajectory["damping"])),
                            "relative_balance_error": trajectory[
                                "relative_balance_error"
                            ],
                            "monotonicity_violations": trajectory[
                                "monotonicity_violations"
                            ],
                            "final_energy_gap_fraction": trajectory[
                                "final_energy_gap_fraction"
                            ],
                            "final_distance_to_equilibrium": trajectory[
                                "final_distance_to_equilibrium"
                            ],
                            "final_velocity_norm": trajectory[
                                "final_velocity_norm"
                            ],
                        }
                    )
                    trajectory_rows.extend(
                        {
                            "seed": seed,
                            "chain_size": chain_size,
                            "boundary_variant": variant,
                            "beta": beta,
                            "time": float(time),
                            "total_energy": float(total),
                            "dissipated_energy": float(dissipated),
                            "balance_residual": float(balance),
                            "energy_gap": float(gap),
                            "distance_to_equilibrium": float(distance),
                        }
                        for time, total, dissipated, balance, gap, distance in zip(
                            trajectory["time"],
                            trajectory["total_energy"],
                            trajectory["dissipated_energy"],
                            trajectory["balance_residual"],
                            trajectory["energy_gap"],
                            trajectory["distance_to_equilibrium"],
                            strict=True,
                        )
                    )

    modal_table = pd.DataFrame(modal_rows)
    energy_table = pd.DataFrame(energy_rows)
    trajectory_table = pd.DataFrame(trajectory_rows)
    controls, control_trajectories = blind_mode_control(settings["t_end"])
    modal_table.to_csv(output_dir / "modal_observability.csv", index=False)
    energy_table.to_csv(output_dir / "energy_balance_summary.csv", index=False)
    trajectory_table.to_csv(output_dir / "energy_trajectories.csv", index=False)
    controls.to_csv(output_dir / "dark_mode_control.csv", index=False)
    control_trajectories.to_csv(
        output_dir / "dark_mode_trajectories.csv", index=False
    )

    summary = {
        "version": "3.0",
        "audit": "energy and modal observability",
        "profile": profile,
        "n_modal_cases": int(len(modal_table)),
        "n_energy_cases": int(len(energy_table)),
        "all_audited_hessians_positive": bool(
            modal_table["strongly_convex_on_audited_states"].all()
        ),
        "minimum_hessian_eigenvalue": float(
            modal_table["minimum_hessian_eigenvalue"].min()
        ),
        "all_chain_cases_observable": bool(modal_table["observable"].all()),
        "all_chain_cases_exponentially_stable": bool(
            modal_table["linearly_exponentially_stable"].all()
        ),
        "minimum_chain_eigenspace_visibility": float(
            modal_table["minimum_eigenspace_visibility"].min()
        ),
        "maximum_relative_energy_balance_error": float(
            energy_table["relative_balance_error"].max()
        ),
        "total_monotonicity_violations": int(
            energy_table["monotonicity_violations"].sum()
        ),
        "single_terminal_maximum_damped_nodes": int(
            modal_table.loc[
                modal_table["boundary_variant"] == "single_terminal", "n_damped"
            ].max()
        ),
        "dark_control_is_unobservable": bool(
            not controls.loc[
                controls["control"] == "blind_center", "observable"
            ].iloc[0]
        ),
        "dark_control_final_energy_fraction": float(
            controls.loc[
                controls["control"] == "blind_center", "final_energy_fraction"
            ].iloc[0]
        ),
        "visible_control_final_energy_fraction": float(
            controls.loc[
                controls["control"] == "visible_endpoint", "final_energy_fraction"
            ].iloc[0]
        ),
    }
    summary["passes_core_gate"] = bool(
        summary["all_audited_hessians_positive"]
        and summary["all_chain_cases_observable"]
        and summary["all_chain_cases_exponentially_stable"]
        and summary["maximum_relative_energy_balance_error"] <= 1e-5
        and summary["total_monotonicity_violations"] == 0
        and summary["single_terminal_maximum_damped_nodes"] == 1
        and summary["dark_control_is_unobservable"]
        and summary["dark_control_final_energy_fraction"] >= 0.95
    )
    write_json(output_dir / "energy_observability_summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    representative = trajectory_table[
        (trajectory_table["seed"] == settings["seeds"][0])
        & (trajectory_table["chain_size"] == settings["sizes"][-1])
        & (trajectory_table["boundary_variant"] == "single_terminal")
        & (trajectory_table["beta"] == 0.0)
    ]
    axes[0].semilogy(
        representative["time"],
        np.maximum(representative["energy_gap"], 1e-16),
        label="energy above equilibrium",
    )
    axes[0].semilogy(
        representative["time"],
        np.maximum(np.abs(representative["balance_residual"]), 1e-16),
        label="balance residual",
    )
    axes[0].set(xlabel="time", ylabel="energy/error", title="Single-terminal energy balance")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)

    grouped = modal_table[modal_table["beta"] == 0.0].groupby(
        ["chain_size", "boundary_variant"], as_index=False
    ).agg(
        minimum_visibility=("minimum_eigenspace_visibility", "min"),
        decay_rate=("spectral_decay_rate", "median"),
    )
    for variant, part in grouped.groupby("boundary_variant"):
        axes[1].loglog(
            part["chain_size"],
            part["minimum_visibility"],
            marker="o",
            label=variant,
        )
        axes[2].loglog(
            part["chain_size"],
            part["decay_rate"],
            marker="o",
            label=variant,
        )
    axes[1].set(xlabel="chain size", ylabel="minimum modal visibility", title="Boundary observability")
    axes[2].set(xlabel="chain size", ylabel="spectral decay rate", title="Predicted exponential rate")
    for axis in axes[1:]:
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "energy_observability_audit.png", dpi=190)
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_v3_energy_observability")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.profile, arguments.output_dir)
    print(result)
