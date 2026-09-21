#!/usr/bin/env python3
"""Timestep-refinement audit for localized boundary relaxation.

This audit checks representative trajectories with smaller time steps while holding physical time,
conservative parameters, nudging strength, initial free state, damping support,
and total damping trace fixed while reducing the velocity-Verlet split step.

We test a slow endpoint-only configuration together with the 6/17 and 8/17
boundary-layer configurations used as the principal localized-damping reference points.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib-bdw-timestep"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_rep_experiments import damping_vector
from benchmark_suite import chain_configuration, make_dataset
from boundary_rep_learning import (
    energy,
    flatten_gradients,
    initialize_parameters,
    make_features,
    solve_equilibrium,
    state_gradient,
    symmetric_eqprop_gradients,
)


PROFILES = {
    "quick": {
        "datasets": ("moons",),
        "seeds": (17,),
        "n_samples": 4,
        "n_rbf": 4,
        "physical_time": 400.0,
    },
    "paper": {
        "datasets": ("moons", "spirals"),
        "seeds": (17, 29, 43),
        "n_samples": 6,
        "n_rbf": 6,
        "physical_time": 1200.0,
    },
}

CHAIN_SIZE = 17
SUPPORT_WIDTHS = (1, 6, 8)
DTS = (0.20, 0.10, 0.05)
BETA = 0.035
DAMPING_TRACE = 5.0


def make_problem(dataset: str, seed: int, n_samples: int, n_rbf: int):
    rng = np.random.default_rng(seed)
    x, targets = make_dataset(dataset, n_samples, rng)
    x = (x - np.mean(x, axis=0)) / np.maximum(np.std(x, axis=0), 1e-12)
    centers = x[rng.choice(n_samples, min(n_rbf, n_samples), replace=False)]
    features = make_features(x, centers, 0.72)
    config = chain_configuration(
        CHAIN_SIZE,
        seed,
        BETA,
        n_input_nodes=2,
        damping_schedule="fixed_trace",
        damping_trace=DAMPING_TRACE,
        chain_regime="propagating",
    )
    params = initialize_parameters(config, features.shape[1])
    return config, params, features, targets


def integrate_phase(
    params,
    features: np.ndarray,
    config,
    *,
    targets: np.ndarray,
    beta: float,
    initial_q: np.ndarray,
    exact_q: np.ndarray,
    dt: float,
    physical_time: float,
) -> dict[str, object]:
    steps_float = physical_time / dt
    steps = int(round(steps_float))
    if not np.isclose(steps * dt, physical_time, rtol=0.0, atol=1e-12):
        raise ValueError("physical_time must be an integer multiple of dt")

    q = np.asarray(initial_q, dtype=float).copy()
    velocity = np.zeros_like(q)
    damping = damping_vector(config, "boundary", params)
    half_decay = np.exp(-0.5 * dt * damping)[None, :]
    dissipated = np.zeros(features.shape[0], dtype=float)

    h0 = energy(q, params, features, config, targets, beta)
    h_eq = energy(exact_q, params, features, config, targets, beta)

    for _ in range(steps):
        kinetic_before = 0.5 * np.sum(velocity**2, axis=1)
        velocity *= half_decay
        kinetic_after = 0.5 * np.sum(velocity**2, axis=1)
        dissipated += np.maximum(kinetic_before - kinetic_after, 0.0)

        gradient = state_gradient(
            q, params, features, config, targets, beta
        )
        velocity -= 0.5 * dt * gradient
        q += dt * velocity
        gradient_new = state_gradient(
            q, params, features, config, targets, beta
        )
        velocity -= 0.5 * dt * gradient_new

        kinetic_before = 0.5 * np.sum(velocity**2, axis=1)
        velocity *= half_decay
        kinetic_after = 0.5 * np.sum(velocity**2, axis=1)
        dissipated += np.maximum(kinetic_before - kinetic_after, 0.0)

        if not np.isfinite(q).all() or not np.isfinite(velocity).all():
            raise FloatingPointError(
                f"non-finite split trajectory dt={dt}, beta={beta}"
            )

    final_gradient = state_gradient(
        q, params, features, config, targets, beta
    )
    h_final = (
        0.5 * np.sum(velocity**2, axis=1)
        + energy(q, params, features, config, targets, beta)
    )
    energy_gap_scale = np.maximum(np.abs(h0 - h_eq), 1e-14)
    balance = np.abs(h_final + dissipated - h0) / energy_gap_scale

    return {
        "q": q,
        "velocity": velocity,
        "steps": steps,
        "residual": float(np.max(np.linalg.norm(final_gradient, axis=1))),
        "velocity_norm": float(np.max(np.linalg.norm(velocity, axis=1))),
        "mean_dissipated_energy": float(np.mean(dissipated)),
        "maximum_relative_split_energy_balance_error": float(
            np.max(balance)
        ),
        "median_relative_split_energy_balance_error": float(
            np.median(balance)
        ),
    }


def rel_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - reference)
        / max(float(np.linalg.norm(reference)), 1e-14)
    )


def run(profile: str, output_dir: Path) -> dict[str, object]:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for dataset in settings["datasets"]:
        for seed in settings["seeds"]:
            base_config, params, features, targets = make_problem(
                dataset,
                int(seed),
                int(settings["n_samples"]),
                int(settings["n_rbf"]),
            )
            q_free, _, free_residual = solve_equilibrium(
                params, features, base_config
            )
            q_plus_exact, _, plus_residual = solve_equilibrium(
                params,
                features,
                base_config,
                targets,
                BETA,
                initial=q_free,
            )
            q_minus_exact, _, minus_residual = solve_equilibrium(
                params,
                features,
                base_config,
                targets,
                -BETA,
                initial=q_free,
            )
            exact_centered = flatten_gradients(
                symmetric_eqprop_gradients(
                    q_minus_exact,
                    q_plus_exact,
                    params,
                    features,
                    base_config,
                    BETA,
                )
            )

            for width in SUPPORT_WIDTHS:
                config = replace(
                    base_config,
                    boundary_width=int(width),
                    boundary_profile="fixed_trace",
                    boundary_trace=DAMPING_TRACE,
                    boundary_damping=DAMPING_TRACE / float(width),
                )
                damping = damping_vector(config, "boundary", params)
                if not np.isclose(
                    np.sum(damping), DAMPING_TRACE, atol=1e-12, rtol=0.0
                ):
                    raise AssertionError("damping trace changed during refinement")

                temporary: list[dict[str, object]] = []
                gradient_by_dt: dict[float, np.ndarray] = {}
                plus_q_by_dt: dict[float, np.ndarray] = {}
                minus_q_by_dt: dict[float, np.ndarray] = {}

                for dt in DTS:
                    plus = integrate_phase(
                        params,
                        features,
                        config,
                        targets=targets,
                        beta=BETA,
                        initial_q=q_free,
                        exact_q=q_plus_exact,
                        dt=float(dt),
                        physical_time=float(settings["physical_time"]),
                    )
                    minus = integrate_phase(
                        params,
                        features,
                        config,
                        targets=targets,
                        beta=-BETA,
                        initial_q=q_free,
                        exact_q=q_minus_exact,
                        dt=float(dt),
                        physical_time=float(settings["physical_time"]),
                    )
                    dynamic = flatten_gradients(
                        symmetric_eqprop_gradients(
                            np.asarray(minus["q"]),
                            np.asarray(plus["q"]),
                            params,
                            features,
                            config,
                            BETA,
                        )
                    )
                    gradient_by_dt[float(dt)] = dynamic
                    plus_q_by_dt[float(dt)] = np.asarray(plus["q"])
                    minus_q_by_dt[float(dt)] = np.asarray(minus["q"])
                    temporary.append(
                        {
                            "profile": profile,
                            "dataset": dataset,
                            "seed": int(seed),
                            "chain_size": CHAIN_SIZE,
                            "support_width": int(width),
                            "damping_trace": float(np.sum(damping)),
                            "dt": float(dt),
                            "physical_time": float(settings["physical_time"]),
                            "steps_per_phase": int(plus["steps"]),
                            "dynamic_vs_exact_centered_gradient_error": rel_error(
                                dynamic, exact_centered
                            ),
                            "plus_state_vs_exact_error": rel_error(
                                np.asarray(plus["q"]), q_plus_exact
                            ),
                            "minus_state_vs_exact_error": rel_error(
                                np.asarray(minus["q"]), q_minus_exact
                            ),
                            "plus_residual": float(plus["residual"]),
                            "minus_residual": float(minus["residual"]),
                            "plus_velocity_norm": float(
                                plus["velocity_norm"]
                            ),
                            "minus_velocity_norm": float(
                                minus["velocity_norm"]
                            ),
                            "maximum_relative_split_energy_balance_error": float(
                                max(
                                    plus[
                                        "maximum_relative_split_energy_balance_error"
                                    ],
                                    minus[
                                        "maximum_relative_split_energy_balance_error"
                                    ],
                                )
                            ),
                            "median_relative_split_energy_balance_error": float(
                                np.median(
                                    [
                                        plus[
                                            "median_relative_split_energy_balance_error"
                                        ],
                                        minus[
                                            "median_relative_split_energy_balance_error"
                                        ],
                                    ]
                                )
                            ),
                            "mean_dissipated_energy_two_phases": float(
                                plus["mean_dissipated_energy"]
                                + minus["mean_dissipated_energy"]
                            ),
                            "free_exact_residual": float(free_residual),
                            "plus_exact_residual": float(plus_residual),
                            "minus_exact_residual": float(minus_residual),
                        }
                    )

                finest = float(min(DTS))
                finest_gradient = gradient_by_dt[finest]
                finest_plus = plus_q_by_dt[finest]
                finest_minus = minus_q_by_dt[finest]
                for row in temporary:
                    dt = float(row["dt"])
                    row["gradient_difference_vs_finest_dt"] = rel_error(
                        gradient_by_dt[dt], finest_gradient
                    )
                    row["plus_state_difference_vs_finest_dt"] = rel_error(
                        plus_q_by_dt[dt], finest_plus
                    )
                    row["minus_state_difference_vs_finest_dt"] = rel_error(
                        minus_q_by_dt[dt], finest_minus
                    )
                    rows.append(row)
                    print(
                        f"{dataset} seed={seed} width={width} dt={dt:.3f}: "
                        f"relax_err={row['dynamic_vs_exact_centered_gradient_error']:.3g} "
                        f"vs_finest={row['gradient_difference_vs_finest_dt']:.3g} "
                        f"balance={row['maximum_relative_split_energy_balance_error']:.3g}",
                        flush=True,
                    )

    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "timestep_refinement.csv",
        index=False,
    )

    grouped = (
        table.groupby(["support_width", "dt"], as_index=False)
        .agg(
            median_relaxation_gradient_error=(
                "dynamic_vs_exact_centered_gradient_error",
                "median",
            ),
            maximum_gradient_difference_vs_finest=(
                "gradient_difference_vs_finest_dt",
                "max",
            ),
            maximum_state_difference_vs_finest=(
                "plus_state_difference_vs_finest_dt",
                "max",
            ),
            maximum_split_energy_balance_error=(
                "maximum_relative_split_energy_balance_error",
                "max",
            ),
        )
        .sort_values(["support_width", "dt"])
    )
    grouped.to_csv(
        output_dir / "timestep_refinement_summary.csv",
        index=False,
    )

    # Estimate observed refinement order using the two non-finest step sizes
    # against the dt=0.05 reference.  This is a consistency diagnostic, not a
    # formal asymptotic-order claim.
    orders: list[dict[str, object]] = []
    for (dataset, seed, width), part in table.groupby(
        ["dataset", "seed", "support_width"], sort=True
    ):
        values = {
            float(row.dt): float(row.gradient_difference_vs_finest_dt)
            for row in part.itertuples()
        }
        e20 = values[0.20]
        e10 = values[0.10]
        if e20 > 0.0 and e10 > 0.0:
            order = float(np.log(e20 / e10) / np.log(2.0))
        else:
            order = float("nan")
        orders.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "support_width": int(width),
                "observed_gradient_refinement_order": order,
            }
        )
    order_table = pd.DataFrame(orders)
    order_table.to_csv(
        output_dir / "timestep_refinement_orders.csv",
        index=False,
    )

    summary = {
        "version": "1.0",
        "profile": profile,
        "physical_time": float(settings["physical_time"]),
        "dt_values": list(DTS),
        "support_widths": list(SUPPORT_WIDTHS),
        "fixed_damping_trace": DAMPING_TRACE,
        "n_cases": int(len(table)),
        "maximum_dt_0p20_gradient_difference_vs_dt_0p05": float(
            table.loc[
                np.isclose(table["dt"], 0.20),
                "gradient_difference_vs_finest_dt",
            ].max()
        ),
        "maximum_dt_0p10_gradient_difference_vs_dt_0p05": float(
            table.loc[
                np.isclose(table["dt"], 0.10),
                "gradient_difference_vs_finest_dt",
            ].max()
        ),
        "maximum_split_energy_balance_error": float(
            table["maximum_relative_split_energy_balance_error"].max()
        ),
        "median_observed_gradient_refinement_order": float(
            order_table["observed_gradient_refinement_order"].median()
        ),
    }
    (output_dir / "timestep_refinement_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    for width, part in grouped.groupby("support_width"):
        axes[0].loglog(
            part["dt"],
            part["median_relaxation_gradient_error"],
            marker="o",
            label=f"m={int(width)}",
        )
        axes[1].loglog(
            part["dt"],
            np.maximum(
                part["maximum_gradient_difference_vs_finest"], 1e-16
            ),
            marker="o",
            label=f"m={int(width)}",
        )
        axes[2].loglog(
            part["dt"],
            np.maximum(
                part["maximum_split_energy_balance_error"], 1e-16
            ),
            marker="o",
            label=f"m={int(width)}",
        )

    axes[0].set(
        xlabel="time step",
        ylabel="median relaxation-gradient error",
        title="Fixed physical time",
    )
    axes[1].set(
        xlabel="time step",
        ylabel="max gradient difference vs dt=0.05",
        title="Timestep refinement",
    )
    axes[2].set(
        xlabel="time step",
        ylabel="max split energy-balance residual",
        title="Integration consistency",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
        axis.invert_xaxis()
    figure.tight_layout()
    figure.savefig(
        output_dir / "timestep_refinement.png",
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
        default=Path("results/timestep_refinement"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.profile, args.output_dir), indent=2))
