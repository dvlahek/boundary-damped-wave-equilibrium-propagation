#!/usr/bin/env python3
"""Audit boundary-damped EqProp on grids and sparse non-chain graphs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v5-graphs")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from graph_eqprop_core import (
    GraphConfig,
    GraphParameters,
    centered_gradients,
    cosine_similarity,
    damping_vector,
    flatten,
    implicit_gradients,
    initialize_parameters,
    make_config,
    modal_observability,
    physical_coefficients,
    relative_error,
    relax_dynamics,
    select_modal_boundary,
    solve_equilibrium,
    state_hessian,
)


PROFILES = {
    "quick": {
        "topologies": ("grid_3", "sparse_9"),
        "seeds": (17,),
        "n_samples": 6,
        "max_steps": 120_000,
        "tolerance": 3e-5,
    },
    "paper": {
        "topologies": ("grid_4", "grid_5", "sparse_16", "sparse_25"),
        "seeds": (17, 29, 43, 71, 97),
        "n_samples": 8,
        "max_steps": 360_000,
        "tolerance": 1e-5,
    },
}


def make_problem(
    topology: str, seed: int, n_samples: int
) -> tuple[GraphConfig, GraphParameters, np.ndarray, np.ndarray]:
    config = make_config(topology, seed)
    rng = np.random.default_rng(seed + 10_003)
    features = rng.normal(size=(n_samples, 7))
    features[:, 0] = 1.0
    targets = np.where(
        features[:, 1] * features[:, 2] + 0.35 * features[:, 3] >= 0.0,
        1.0,
        -1.0,
    )
    params = initialize_parameters(config, features.shape[1], seed + 20_011)
    if topology.startswith("sparse_"):
        preliminary_free, _, _ = solve_equilibrium(params, features, config)
        config = select_modal_boundary(config, params, q_states=preliminary_free)
    return config, params, features, targets


def run_dark_mode_controls(output_dir: Path) -> pd.DataFrame:
    """Symmetric star has a leaf-antisymmetric mode invisible at its center."""
    n_nodes = 7
    edges = np.asarray([(0, node) for node in range(1, n_nodes)], dtype=int)
    config = GraphConfig(
        n_nodes=n_nodes,
        edges=edges,
        input_nodes=np.asarray([1, 2]),
        output_nodes=np.asarray([0]),
        alpha=0.0,
        damping_trace=1.0,
    )
    base_log_a = np.full(n_nodes, np.log(np.expm1(0.90 - config.stiffness_floor)))
    base_log_w = np.full(len(edges), np.log(np.expm1(0.70 - config.edge_floor)))
    features = np.zeros((1, 2))
    rows: list[dict[str, object]] = []
    for variant in ("symmetric_dark", "perturbed_visible"):
        log_w = base_log_w.copy()
        if variant == "perturbed_visible":
            log_w += np.linspace(-1.20, 1.20, len(log_w))
        params = GraphParameters(base_log_a.copy(), log_w, np.zeros((2, 2)))
        equilibrium = np.zeros((1, n_nodes))
        hessian = state_hessian(equilibrium, params, config)[0]
        damping = damping_vector(config, "boundary")
        modal = modal_observability(hessian, damping)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        boundary_amplitude = np.abs(eigenvectors[0, :])
        mode_index = int(np.argmin(boundary_amplitude))
        initial_q = eigenvectors[:, mode_index]
        initial_state = np.concatenate((initial_q, np.zeros(n_nodes)))

        system = np.block(
            [
                [np.zeros_like(hessian), np.eye(n_nodes)],
                [-hessian, -np.diag(damping)],
            ]
        )

        def rhs(_: float, state: np.ndarray) -> np.ndarray:
            return system @ state

        solution = solve_ivp(
            rhs,
            (0.0, 600.0),
            initial_state,
            method="DOP853",
            rtol=2e-10,
            atol=2e-12,
        )
        final_q = solution.y[:n_nodes, -1]
        final_velocity = solution.y[n_nodes:, -1]
        initial_energy = 0.5 * initial_q @ hessian @ initial_q
        final_energy = 0.5 * (
            final_velocity @ final_velocity + final_q @ hessian @ final_q
        )
        rows.append(
            {
                "variant": variant,
                "selected_mode_eigenvalue": float(eigenvalues[mode_index]),
                "selected_mode_boundary_amplitude": float(
                    boundary_amplitude[mode_index]
                ),
                **modal,
                "integration_time": 600.0,
                "retained_energy_fraction": float(final_energy / initial_energy),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "dark_mode_graph_control.csv", index=False)
    return table


def run(profile: str, output_dir: Path) -> pd.DataFrame:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "graph_topology_audit.csv"
    if result_path.exists():
        existing = pd.read_csv(result_path)
        rows: list[dict[str, object]] = existing.to_dict(orient="records")
        completed = {
            (str(row.topology), int(row.seed), str(row.damping_mode))
            for row in existing.itertuples()
        }
        print(f"resuming from {len(existing)} completed graph cases", flush=True)
    else:
        rows = []
        completed: set[tuple[str, int, str]] = set()
    for topology in settings["topologies"]:
        for seed in settings["seeds"]:
            requested_keys = {
                (topology, int(seed), mode)
                for mode in ("boundary", "uniform_trace")
            }
            if requested_keys.issubset(completed):
                print(f"topology={topology} seed={seed}: already completed", flush=True)
                continue
            print(f"topology={topology} seed={seed}: preparing exact endpoints", flush=True)
            config, params, features, targets = make_problem(
                topology, seed, settings["n_samples"]
            )
            q_free, _, free_exact_residual = solve_equilibrium(
                params, features, config
            )
            q_plus, _, plus_exact_residual = solve_equilibrium(
                params,
                features,
                config,
                targets,
                config.beta,
                initial=q_free,
            )
            q_minus, _, minus_exact_residual = solve_equilibrium(
                params,
                features,
                config,
                targets,
                -config.beta,
                initial=q_free,
            )
            exact_centered = flatten(
                centered_gradients(
                    q_minus, q_plus, params, features, config, config.beta
                )
            )
            implicit = flatten(
                implicit_gradients(q_free, params, features, targets, config)
            )
            exact_bias = relative_error(exact_centered, implicit)
            exact_cosine = cosine_similarity(exact_centered, implicit)
            hessian = state_hessian(q_free, params, config)
            minimum_hessian = float(np.min(np.linalg.eigvalsh(hessian)))

            for damping_mode in ("boundary", "uniform_trace"):
                key = (topology, int(seed), damping_mode)
                if key in completed:
                    continue
                modal_rows = [
                    modal_observability(
                        hessian[sample_index], damping_vector(config, damping_mode)
                    )
                    for sample_index in range(len(features))
                ]
                print(
                    f"  damping={damping_mode}: free, plus, and minus dynamics",
                    flush=True,
                )
                free_dynamic = relax_dynamics(
                    params,
                    features,
                    config,
                    damping_mode=damping_mode,
                    max_steps=settings["max_steps"],
                    tolerance=settings["tolerance"],
                )
                plus_dynamic = relax_dynamics(
                    params,
                    features,
                    config,
                    targets=targets,
                    beta=config.beta,
                    damping_mode=damping_mode,
                    initial_q=np.asarray(free_dynamic["q"]),
                    max_steps=settings["max_steps"],
                    tolerance=settings["tolerance"],
                )
                minus_dynamic = relax_dynamics(
                    params,
                    features,
                    config,
                    targets=targets,
                    beta=-config.beta,
                    damping_mode=damping_mode,
                    initial_q=np.asarray(free_dynamic["q"]),
                    max_steps=settings["max_steps"],
                    tolerance=settings["tolerance"],
                )
                dynamic_centered = flatten(
                    centered_gradients(
                        np.asarray(minus_dynamic["q"]),
                        np.asarray(plus_dynamic["q"]),
                        params,
                        features,
                        config,
                        config.beta,
                    )
                )
                phase_results = (free_dynamic, plus_dynamic, minus_dynamic)
                rows.append(
                    {
                        "profile": profile,
                        "topology": topology,
                        "seed": seed,
                        "n_nodes": config.n_nodes,
                        "n_edges": len(config.edges),
                        "n_input_nodes": len(config.input_nodes),
                        "n_output_dampers": len(config.output_nodes),
                        "damping_mode": damping_mode,
                        "damping_trace": config.damping_trace,
                        "minimum_hessian_eigenvalue": minimum_hessian,
                        "minimum_eigenspace_visibility": float(
                            min(
                                float(item["minimum_eigenspace_visibility"])
                                for item in modal_rows
                            )
                        ),
                        "all_samples_observable": bool(
                            all(bool(item["observable"]) for item in modal_rows)
                        ),
                        "minimum_spectral_decay_rate": float(
                            min(float(item["spectral_decay_rate"]) for item in modal_rows)
                        ),
                        "all_phases_converged": bool(
                            all(bool(item["converged"]) for item in phase_results)
                        ),
                        "minimum_phase_convergence_fraction": float(
                            min(
                                float(item["convergence_fraction"])
                                for item in phase_results
                            )
                        ),
                        "free_state_relative_error": relative_error(
                            np.asarray(free_dynamic["q"]), q_free
                        ),
                        "plus_state_relative_error": relative_error(
                            np.asarray(plus_dynamic["q"]), q_plus
                        ),
                        "minus_state_relative_error": relative_error(
                            np.asarray(minus_dynamic["q"]), q_minus
                        ),
                        "dynamic_vs_exact_centered_gradient_error": relative_error(
                            dynamic_centered, exact_centered
                        ),
                        "dynamic_vs_implicit_gradient_error": relative_error(
                            dynamic_centered, implicit
                        ),
                        "dynamic_vs_implicit_cosine": cosine_similarity(
                            dynamic_centered, implicit
                        ),
                        "exact_centered_vs_implicit_error": exact_bias,
                        "exact_centered_vs_implicit_cosine": exact_cosine,
                        "free_exact_residual": free_exact_residual,
                        "plus_exact_residual": plus_exact_residual,
                        "minus_exact_residual": minus_exact_residual,
                        "total_active_sample_steps": int(
                            sum(int(item["active_sample_steps"]) for item in phase_results)
                        ),
                    }
                )
                completed.add(key)
                pd.DataFrame(rows).to_csv(result_path, index=False)

    table = pd.DataFrame(rows)
    table.to_csv(result_path, index=False)
    dark = run_dark_mode_controls(output_dir)
    summary = {
        "profile": profile,
        "topologies": list(settings["topologies"]),
        "seeds": list(settings["seeds"]),
        "n_regular_cases": int(len(table)),
        "all_regular_hessians_positive": bool(
            (table["minimum_hessian_eigenvalue"] > 0.0).all()
        ),
        "all_regular_cases_observable": bool(table["all_samples_observable"].all()),
        "all_regular_phases_converged": bool(table["all_phases_converged"].all()),
        "minimum_regular_convergence_fraction": float(
            table["minimum_phase_convergence_fraction"].min()
        ),
        "maximum_dynamic_vs_exact_centered_gradient_error": float(
            table["dynamic_vs_exact_centered_gradient_error"].max()
        ),
        "maximum_dynamic_vs_implicit_gradient_error": float(
            table["dynamic_vs_implicit_gradient_error"].max()
        ),
        "minimum_dynamic_vs_implicit_cosine": float(
            table["dynamic_vs_implicit_cosine"].min()
        ),
        "dark_control": dark.to_dict(orient="records"),
    }
    with (output_dir / "graph_topology_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    grouped = (
        table.groupby(["topology", "damping_mode"], as_index=False, observed=True)
        .agg(
            median_gradient_error=("dynamic_vs_implicit_gradient_error", "median"),
            minimum_visibility=("minimum_eigenspace_visibility", "min"),
            convergence_fraction=("minimum_phase_convergence_fraction", "mean"),
        )
    )
    labels = [f"{row.topology}\n{row.damping_mode}" for row in grouped.itertuples()]
    positions = np.arange(len(grouped))
    axes[0].bar(positions, grouped["median_gradient_error"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median relative gradient error")
    axes[1].bar(positions, grouped["minimum_visibility"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("minimum eigenspace visibility")
    axes[2].bar(positions, grouped["convergence_fraction"])
    axes[2].set_ylim(0.0, 1.03)
    axes[2].set_ylabel("mean minimum phase convergence")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Boundary damping beyond one-dimensional chains")
    figure.tight_layout()
    figure.savefig(output_dir / "graph_topology_audit.png", dpi=180)
    plt.close(figure)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path(f"results_graph_topology_{args.profile}")
    table = run(args.profile, output)
    print(f"saved {len(table)} regular graph audit rows to {output}")


if __name__ == "__main__":
    main()
