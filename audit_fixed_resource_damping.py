#!/usr/bin/env python3
"""Fixed-resource damping-support audit for localized wave relaxation.

This experiment isolates the physical cost of localizing dissipation. The chain
size, conservative model, nudging strength, and total damping trace are held
fixed while the support of the damping operator is reduced from all nodes to a
single terminal node. Positive and negative nudged trajectories start from the
same exact free equilibrium so that the comparison measures the resource needed
to generate the centered EqProp gradient rather than differences in free-phase
initialization.

Outputs include gradient error against implicit differentiation at fixed physical
budgets, error against exact centered EqProp, modal visibility, exact spectral
decay rate, and dissipated energy of the split dynamical integrator.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-bdw-fixed-resource")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_rep_experiments import damping_vector
from benchmark_suite import chain_configuration, make_dataset
from boundary_rep_learning import (
    flatten_gradients,
    implicit_gradients,
    initialize_parameters,
    make_features,
    solve_equilibrium,
    state_gradient,
    state_hessian,
    symmetric_eqprop_gradients,
)
from theory_audit_core import cosine_similarity, modal_observability, relative_error


PROFILES = {
    "quick": {
        "datasets": ("moons",),
        "seeds": (17,),
        "n_samples": 4,
        "n_rbf": 4,
        "dt": 0.20,
        "budgets": (250, 1_000, 4_000, 12_000),
    },
    "paper": {
        "datasets": ("moons", "spirals"),
        "seeds": (17, 29, 43),
        "n_samples": 8,
        "n_rbf": 8,
        "dt": 0.20,
        "budgets": (500, 2_000, 8_000, 20_000, 40_000, 80_000),
    },
    "paper_refined": {
        "datasets": ("moons", "spirals"),
        "seeds": (17, 29, 43),
        "n_samples": 8,
        "n_rbf": 8,
        "dt": 0.10,
        # Same physical checkpoints as the paper profile above.
        "budgets": (1_000, 4_000, 16_000, 40_000, 80_000, 160_000),
    },
}

CHAIN_SIZE = 17
SUPPORT_WIDTHS = (1, 2, 4, 6, 8, 17)
BETA = 0.035
DAMPING_TRACE = 5.0
TARGET_ERRORS = (0.01, 0.003)
TARGET_METRIC = "dynamic_vs_exact_centered_relative_error"


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


def relax_with_snapshots(
    params,
    features: np.ndarray,
    config,
    budgets: tuple[int, ...],
    *,
    targets: np.ndarray,
    beta: float,
    initial_q: np.ndarray,
    dt: float,
) -> dict[int, dict[str, object]]:
    """Integrate one phase without early freezing and retain fixed-budget snapshots."""
    budgets = tuple(sorted(set(int(value) for value in budgets)))
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must be positive")

    q = np.asarray(initial_q, dtype=float).copy()
    velocity = np.zeros_like(q)
    damping = damping_vector(config, "boundary", params)
    half_decay = np.exp(-0.5 * dt * damping)[None, :]
    dissipated_per_sample = np.zeros(features.shape[0], dtype=float)
    snapshots: dict[int, dict[str, object]] = {}

    for step in range(1, budgets[-1] + 1):
        kinetic_before = 0.5 * np.sum(velocity**2, axis=1)
        velocity *= half_decay
        kinetic_after = 0.5 * np.sum(velocity**2, axis=1)
        dissipated_per_sample += np.maximum(kinetic_before - kinetic_after, 0.0)

        gradient = state_gradient(q, params, features, config, targets, beta)
        velocity -= 0.5 * dt * gradient
        q += dt * velocity
        gradient_new = state_gradient(q, params, features, config, targets, beta)
        velocity -= 0.5 * dt * gradient_new

        kinetic_before = 0.5 * np.sum(velocity**2, axis=1)
        velocity *= half_decay
        kinetic_after = 0.5 * np.sum(velocity**2, axis=1)
        dissipated_per_sample += np.maximum(kinetic_before - kinetic_after, 0.0)

        if not np.isfinite(q).all() or not np.isfinite(velocity).all():
            raise FloatingPointError(
                f"non-finite state for width={config.boundary_width}, beta={beta}, step={step}"
            )

        if step in budgets:
            final_gradient = state_gradient(q, params, features, config, targets, beta)
            snapshots[step] = {
                "q": q.copy(),
                "velocity": velocity.copy(),
                "residual": float(np.max(np.linalg.norm(final_gradient, axis=1))),
                "velocity_norm": float(np.max(np.linalg.norm(velocity, axis=1))),
                "mean_dissipated_energy": float(np.mean(dissipated_per_sample)),
                "total_dissipated_energy": float(np.sum(dissipated_per_sample)),
            }

    return snapshots


def spectral_summary(q_states: np.ndarray, params, config) -> dict[str, float | bool]:
    damping = damping_vector(config, "boundary", params)
    hessians = state_hessian(q_states, params, config)
    rows = [modal_observability(hessian, damping) for hessian in hessians]
    decay = np.asarray([float(row["spectral_decay_rate"]) for row in rows])
    visibility = np.asarray(
        [float(row["minimum_eigenspace_visibility"]) for row in rows]
    )
    return {
        "minimum_eigenspace_visibility": float(np.min(visibility)),
        "median_eigenspace_visibility": float(np.median(visibility)),
        "minimum_spectral_decay_rate": float(np.min(decay)),
        "median_spectral_decay_rate": float(np.median(decay)),
        "all_observable": bool(all(bool(row["observable"]) for row in rows)),
    }


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
            q_free, _, free_residual = solve_equilibrium(params, features, base_config)
            q_plus_exact, _, plus_exact_residual = solve_equilibrium(
                params, features, base_config, targets, BETA, initial=q_free
            )
            q_minus_exact, _, minus_exact_residual = solve_equilibrium(
                params, features, base_config, targets, -BETA, initial=q_free
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
            implicit = flatten_gradients(
                implicit_gradients(q_free, params, features, targets, base_config)
            )
            implicit_norm = max(float(np.linalg.norm(implicit)), 1e-14)
            finite_beta_bias = float(
                np.linalg.norm(exact_centered - implicit) / implicit_norm
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
                    float(np.sum(damping)), DAMPING_TRACE, rtol=0.0, atol=1e-12
                ):
                    raise AssertionError("damping trace changed across support sweep")

                spectral = spectral_summary(q_free, params, config)
                plus = relax_with_snapshots(
                    params,
                    features,
                    config,
                    settings["budgets"],
                    targets=targets,
                    beta=BETA,
                    initial_q=q_free,
                    dt=float(settings["dt"]),
                )
                minus = relax_with_snapshots(
                    params,
                    features,
                    config,
                    settings["budgets"],
                    targets=targets,
                    beta=-BETA,
                    initial_q=q_free,
                    dt=float(settings["dt"]),
                )

                for budget in settings["budgets"]:
                    q_plus = np.asarray(plus[int(budget)]["q"])
                    q_minus = np.asarray(minus[int(budget)]["q"])
                    dynamic = flatten_gradients(
                        symmetric_eqprop_gradients(
                            q_minus, q_plus, params, features, config, BETA
                        )
                    )
                    total_error = float(
                        np.linalg.norm(dynamic - implicit) / implicit_norm
                    )
                    relaxation_error = float(
                        np.linalg.norm(dynamic - exact_centered) / implicit_norm
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "chain_size": CHAIN_SIZE,
                            "support_width": int(width),
                            "damped_fraction": float(width / CHAIN_SIZE),
                            "damping_trace": float(np.sum(damping)),
                            "damping_per_node": float(DAMPING_TRACE / width),
                            "budget_steps": int(budget),
                            "physical_time": float(settings["dt"] * budget),
                            "dt": float(settings["dt"]),
                            "dynamic_vs_implicit_relative_error": total_error,
                            "dynamic_vs_exact_centered_relative_error": relaxation_error,
                            "exact_centered_vs_implicit_relative_error": finite_beta_bias,
                            "dynamic_vs_implicit_cosine": cosine_similarity(
                                dynamic, implicit
                            ),
                            "plus_state_relative_error": relative_error(
                                q_plus, q_plus_exact
                            ),
                            "minus_state_relative_error": relative_error(
                                q_minus, q_minus_exact
                            ),
                            "plus_residual": float(plus[int(budget)]["residual"]),
                            "minus_residual": float(minus[int(budget)]["residual"]),
                            "plus_velocity_norm": float(
                                plus[int(budget)]["velocity_norm"]
                            ),
                            "minus_velocity_norm": float(
                                minus[int(budget)]["velocity_norm"]
                            ),
                            "mean_dissipated_energy_two_phases": float(
                                plus[int(budget)]["mean_dissipated_energy"]
                                + minus[int(budget)]["mean_dissipated_energy"]
                            ),
                            "minimum_eigenspace_visibility": spectral[
                                "minimum_eigenspace_visibility"
                            ],
                            "median_eigenspace_visibility": spectral[
                                "median_eigenspace_visibility"
                            ],
                            "minimum_spectral_decay_rate": spectral[
                                "minimum_spectral_decay_rate"
                            ],
                            "median_spectral_decay_rate": spectral[
                                "median_spectral_decay_rate"
                            ],
                            "all_modes_observable": spectral["all_observable"],
                            "free_exact_residual": float(free_residual),
                            "plus_exact_residual": float(plus_exact_residual),
                            "minus_exact_residual": float(minus_exact_residual),
                        }
                    )
                print(
                    f"dataset={dataset} seed={seed} width={width} "
                    f"final_relax_error={rows[-1]['dynamic_vs_exact_centered_relative_error']:.4g} "
                    f"final_total_error={rows[-1]['dynamic_vs_implicit_relative_error']:.4g}"
                )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "fixed_resource_damping_runs.csv", index=False)

    target_rows: list[dict[str, object]] = []
    for (dataset, seed, width), part in table.groupby(
        ["dataset", "seed", "support_width"], sort=True
    ):
        part = part.sort_values("budget_steps")
        for target in TARGET_ERRORS:
            reached = part[part[TARGET_METRIC] <= target]
            if reached.empty:
                target_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "support_width": int(width),
                        "target_relative_error": float(target),
                        "target_metric": TARGET_METRIC,
                        "target_reached": False,
                        "first_budget_steps": np.nan,
                        "first_physical_time": np.nan,
                        "dissipated_energy_at_first_target": np.nan,
                    }
                )
            else:
                first = reached.iloc[0]
                target_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "support_width": int(width),
                        "target_relative_error": float(target),
                        "target_metric": TARGET_METRIC,
                        "target_reached": True,
                        "first_budget_steps": int(first["budget_steps"]),
                        "first_physical_time": float(first["physical_time"]),
                        "dissipated_energy_at_first_target": float(
                            first["mean_dissipated_energy_two_phases"]
                        ),
                    }
                )
    target_table = pd.DataFrame(target_rows)
    target_table.to_csv(
        output_dir / "fixed_resource_damping_targets.csv", index=False
    )

    grouped = (
        table.groupby(["support_width", "budget_steps"], as_index=False)
        .agg(
            median_gradient_error=(
                "dynamic_vs_implicit_relative_error",
                "median",
            ),
            maximum_gradient_error=(
                "dynamic_vs_implicit_relative_error",
                "max",
            ),
            median_relaxation_error=(
                "dynamic_vs_exact_centered_relative_error",
                "median",
            ),
            median_dissipated_energy=(
                "mean_dissipated_energy_two_phases",
                "median",
            ),
            minimum_visibility=("minimum_eigenspace_visibility", "min"),
            minimum_decay_rate=("minimum_spectral_decay_rate", "min"),
        )
        .sort_values(["support_width", "budget_steps"])
    )
    grouped.to_csv(
        output_dir / "fixed_resource_damping_summary.csv", index=False
    )

    one_percent = target_table[target_table["target_relative_error"] == 0.01]
    summary = {
        "version": "1.0",
        "profile": profile,
        "chain_size": CHAIN_SIZE,
        "support_widths": list(SUPPORT_WIDTHS),
        "beta": BETA,
        "fixed_total_damping_trace": DAMPING_TRACE,
        "dt": float(settings["dt"]),
        "datasets": list(settings["datasets"]),
        "seeds": list(settings["seeds"]),
        "budgets": list(settings["budgets"]),
        "n_run_rows": int(len(table)),
        "all_damping_traces_fixed": bool(
            np.allclose(
                table["damping_trace"].to_numpy(float),
                DAMPING_TRACE,
            )
        ),
        "all_modes_observable": bool(table["all_modes_observable"].all()),
        "maximum_exact_finite_beta_bias": float(
            table["exact_centered_vs_implicit_relative_error"].max()
        ),
        "target_metric": TARGET_METRIC,
        "one_percent_relaxation_target_reached_fraction_by_support": {
            str(int(width)): float(block["target_reached"].mean())
            for width, block in one_percent.groupby("support_width")
        },
    }
    (output_dir / "fixed_resource_damping_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    for width, part in grouped.groupby("support_width"):
        axes[0].loglog(
            part["budget_steps"],
            part["median_relaxation_error"],
            marker="o",
            label=f"m={int(width)}",
        )
    axes[0].axhline(
        0.01, linestyle="--", linewidth=1.0, label="1% target"
    )
    axes[0].set(
        xlabel="integration steps",
        ylabel="median relative relaxation-gradient error",
        title="Fixed-time relaxation-gradient fidelity",
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8, ncol=2)

    final = grouped.loc[
        grouped.groupby("support_width")["budget_steps"].idxmax()
    ]
    axes[1].semilogy(
        final["support_width"],
        final["minimum_decay_rate"],
        marker="o",
    )
    axes[1].set(
        xlabel="number of damped nodes",
        ylabel="minimum spectral decay rate",
        title="Modal relaxation bottleneck",
    )
    axes[1].grid(alpha=0.25, which="both")

    reached = one_percent[one_percent["target_reached"]]
    target_group = reached.groupby("support_width", as_index=False).agg(
        median_first_steps=("first_budget_steps", "median")
    )
    axes[2].plot(
        target_group["support_width"],
        target_group["median_first_steps"],
        marker="o",
    )
    axes[2].set(
        xlabel="number of damped nodes",
        ylabel="median first checkpoint to <1% relaxation error",
        title="Resource to relaxation-gradient target",
    )
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "fixed_resource_damping_audit.png", dpi=220
    )
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="quick"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fixed_resource_damping"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.profile, args.output_dir), indent=2))
