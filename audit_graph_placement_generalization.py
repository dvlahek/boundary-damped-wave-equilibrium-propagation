#!/usr/bin/env python3
"""Calibration-to-unseen graph-placement audit: calibration-locked graph damping placement.

The original sparse-graph audit selected the accessible set using modal
information from the same free-state linearizations that were later evaluated.
This control separates support selection from unseen-state evaluation.

For each sparse graph we:
1. keep a fixed supervised/readout set defined only from graph topology,
2. use a disjoint calibration batch to select a modal-optimized damping set,
3. freeze that damping set,
4. evaluate modal visibility, spectral decay, and finite-budget EqProp gradient
   generation on unseen states,
5. compare against the topology-only damping set and deterministic random
   accessible-node sets of the same size and total damping trace.

The supervised cost remains fixed across all damping-placement strategies.  Only
the support of the damping operator changes.
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
    os.path.join(tempfile.gettempdir(), "matplotlib-bdw-graph-placement"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graph_eqprop_core import (
    centered_gradients,
    cosine_similarity,
    flatten,
    implicit_gradients,
    initialize_parameters,
    make_config,
    modal_observability,
    relative_error,
    select_modal_boundary,
    solve_equilibrium,
    state_gradient,
    state_hessian,
)


PROFILES = {
    "quick": {
        "topologies": ("sparse_9",),
        "seeds": (17,),
        "n_calibration": 4,
        "n_evaluation": 4,
        "n_random_spectral": 16,
        "n_random_dynamic": 1,
        "max_steps": 45_000,
        "tolerance": 5e-5,
    },
    "paper": {
        "topologies": ("sparse_16", "sparse_25"),
        "seeds": (17, 29, 43, 71, 97),
        "n_calibration": 8,
        "n_evaluation": 8,
        "n_random_spectral": 64,
        "n_random_dynamic": 1,
        "max_steps": 180_000,
        "tolerance": 2e-5,
    },
}

DT = 0.12


def make_features_targets(seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 10_003)
    features = rng.normal(size=(n_samples, 7))
    features[:, 0] = 1.0
    targets = np.where(
        features[:, 1] * features[:, 2] + 0.35 * features[:, 3] >= 0.0,
        1.0,
        -1.0,
    )
    return features, targets


def custom_damping_vector(
    n_nodes: int, damping_nodes: np.ndarray, trace: float
) -> np.ndarray:
    damping = np.zeros(n_nodes, dtype=float)
    damping[np.asarray(damping_nodes, dtype=int)] = trace / len(damping_nodes)
    return damping


def relax_with_custom_damping(
    params,
    features: np.ndarray,
    config,
    damping: np.ndarray,
    *,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial_q: np.ndarray | None = None,
    max_steps: int,
    tolerance: float,
    dt: float = DT,
    check_every: int = 25,
    consecutive_checks: int = 3,
) -> dict[str, object]:
    """Velocity-Verlet split dynamics with a fixed custom diagonal damping set."""
    batch = features.shape[0]
    q = (
        np.zeros((batch, config.n_nodes))
        if initial_q is None
        else np.asarray(initial_q, dtype=float).copy()
    )
    velocity = np.zeros_like(q)
    damping = np.asarray(damping, dtype=float)
    half_decay = np.exp(-0.5 * dt * damping)[None, :]
    consecutive = np.zeros(batch, dtype=int)
    converged = np.zeros(batch, dtype=bool)
    sample_steps = np.zeros(batch, dtype=int)
    active_sample_steps = 0

    for step in range(1, max_steps + 1):
        active = np.flatnonzero(~converged)
        if active.size == 0:
            break
        q_active = q[active]
        v_active = velocity[active] * half_decay
        f_active = features[active]
        t_active = None if targets is None else targets[active]

        gradient = state_gradient(
            q_active, params, f_active, config, t_active, beta
        )
        v_active -= 0.5 * dt * gradient
        q_active += dt * v_active
        gradient_new = state_gradient(
            q_active, params, f_active, config, t_active, beta
        )
        v_active -= 0.5 * dt * gradient_new
        v_active *= half_decay

        q[active] = q_active
        velocity[active] = v_active
        sample_steps[active] += 1
        active_sample_steps += int(active.size)

        if step % check_every == 0:
            local = (
                (np.linalg.norm(gradient_new, axis=1) <= tolerance)
                & (np.linalg.norm(v_active, axis=1) <= tolerance)
            )
            consecutive[active] = np.where(
                local, consecutive[active] + 1, 0
            )
            newly = active[consecutive[active] >= consecutive_checks]
            converged[newly] = True

    final_gradient = state_gradient(
        q, params, features, config, targets, beta
    )
    return {
        "q": q,
        "velocity": velocity,
        "steps": int(step),
        "sample_steps": sample_steps,
        "active_sample_steps": int(active_sample_steps),
        "converged": bool(np.all(converged)),
        "convergence_fraction": float(np.mean(converged)),
        "residual": float(np.max(np.linalg.norm(final_gradient, axis=1))),
        "velocity_norm": float(np.max(np.linalg.norm(velocity, axis=1))),
    }


def modal_summary(
    hessians: np.ndarray,
    damping: np.ndarray,
) -> dict[str, float | bool]:
    rows = [
        modal_observability(hessian, damping)
        for hessian in np.asarray(hessians)
    ]
    return {
        "minimum_visibility": float(
            min(float(row["minimum_eigenspace_visibility"]) for row in rows)
        ),
        "median_visibility": float(
            np.median(
                [
                    float(row["minimum_eigenspace_visibility"])
                    for row in rows
                ]
            )
        ),
        "minimum_decay_rate": float(
            min(float(row["spectral_decay_rate"]) for row in rows)
        ),
        "median_decay_rate": float(
            np.median(
                [float(row["spectral_decay_rate"]) for row in rows]
            )
        ),
        "all_observable": bool(
            all(bool(row["observable"]) for row in rows)
        ),
    }


def random_damping_sets(
    config, width: int, seed: int, count: int
) -> list[np.ndarray]:
    candidates = np.asarray(
        [
            node
            for node in range(config.n_nodes)
            if node not in set(np.asarray(config.input_nodes, dtype=int))
        ],
        dtype=int,
    )
    rng = np.random.default_rng(seed + 80_003)
    sets: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(sets) < count:
        selected = tuple(
            sorted(
                int(value)
                for value in rng.choice(candidates, width, replace=False)
            )
        )
        if selected in seen:
            continue
        seen.add(selected)
        sets.append(np.asarray(selected, dtype=int))
    return sets


def run(profile: str, output_dir: Path) -> dict[str, object]:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []

    for topology in settings["topologies"]:
        for seed in settings["seeds"]:
            config = make_config(topology, int(seed))
            # The topology-only output set is fixed as the supervised/readout
            # set for every damping-placement strategy in this experiment.
            readout_nodes = np.asarray(config.output_nodes, dtype=int).copy()
            width = len(readout_nodes)
            n_total = int(settings["n_calibration"] + settings["n_evaluation"])
            features, targets = make_features_targets(int(seed), n_total)
            params = initialize_parameters(
                config, features.shape[1], int(seed) + 20_011
            )

            n_cal = int(settings["n_calibration"])
            f_cal, f_eval = features[:n_cal], features[n_cal:]
            t_cal, t_eval = targets[:n_cal], targets[n_cal:]

            q_cal, _, _ = solve_equilibrium(params, f_cal, config)
            q_eval, _, free_residual = solve_equilibrium(params, f_eval, config)
            h_cal = state_hessian(q_cal, params, config)
            h_eval = state_hessian(q_eval, params, config)

            # Modal optimization sees calibration states only.  We use the
            # returned node set as damping support; the supervised/readout set
            # in 'config' is not changed.
            selected_config = select_modal_boundary(
                config,
                params,
                n_boundary_nodes=width,
                q_states=q_cal,
            )
            modal_nodes = np.asarray(selected_config.output_nodes, dtype=int)
            topology_nodes = readout_nodes.copy()

            random_sets = random_damping_sets(
                config,
                width,
                int(seed),
                int(settings["n_random_spectral"]),
            )

            strategy_nodes: list[tuple[str, np.ndarray]] = [
                ("calibration_modal", modal_nodes),
                ("topology_only", topology_nodes),
            ]
            strategy_nodes.extend(
                (f"random_{index}", nodes)
                for index, nodes in enumerate(
                    random_sets[: int(settings["n_random_dynamic"])]
                )
            )

            # Evaluate a larger random pool spectrally on the unseen states.
            for index, nodes in enumerate(random_sets):
                damping = custom_damping_vector(
                    config.n_nodes, nodes, config.damping_trace
                )
                cal_modal = modal_summary(h_cal, damping)
                eval_modal = modal_summary(h_eval, damping)
                random_rows.append(
                    {
                        "profile": profile,
                        "topology": topology,
                        "seed": int(seed),
                        "random_index": int(index),
                        "n_nodes": config.n_nodes,
                        "n_dampers": width,
                        "damping_trace": config.damping_trace,
                        "calibration_minimum_visibility": cal_modal[
                            "minimum_visibility"
                        ],
                        "unseen_minimum_visibility": eval_modal[
                            "minimum_visibility"
                        ],
                        "unseen_minimum_decay_rate": eval_modal[
                            "minimum_decay_rate"
                        ],
                        "unseen_all_observable": eval_modal[
                            "all_observable"
                        ],
                        "damping_nodes": ",".join(map(str, nodes.tolist())),
                    }
                )

            q_plus_exact, _, plus_residual = solve_equilibrium(
                params, f_eval, config, t_eval, config.beta, initial=q_eval
            )
            q_minus_exact, _, minus_residual = solve_equilibrium(
                params, f_eval, config, t_eval, -config.beta, initial=q_eval
            )
            exact_centered = flatten(
                centered_gradients(
                    q_minus_exact,
                    q_plus_exact,
                    params,
                    f_eval,
                    config,
                    config.beta,
                )
            )
            implicit = flatten(
                implicit_gradients(q_eval, params, f_eval, t_eval, config)
            )
            exact_bias = relative_error(exact_centered, implicit)

            for strategy, nodes in strategy_nodes:
                damping = custom_damping_vector(
                    config.n_nodes, nodes, config.damping_trace
                )
                cal_modal = modal_summary(h_cal, damping)
                eval_modal = modal_summary(h_eval, damping)

                plus = relax_with_custom_damping(
                    params,
                    f_eval,
                    config,
                    damping,
                    targets=t_eval,
                    beta=config.beta,
                    initial_q=q_eval,
                    max_steps=int(settings["max_steps"]),
                    tolerance=float(settings["tolerance"]),
                )
                minus = relax_with_custom_damping(
                    params,
                    f_eval,
                    config,
                    damping,
                    targets=t_eval,
                    beta=-config.beta,
                    initial_q=q_eval,
                    max_steps=int(settings["max_steps"]),
                    tolerance=float(settings["tolerance"]),
                )
                dynamic = flatten(
                    centered_gradients(
                        np.asarray(minus["q"]),
                        np.asarray(plus["q"]),
                        params,
                        f_eval,
                        config,
                        config.beta,
                    )
                )

                evaluation_rows.append(
                    {
                        "profile": profile,
                        "topology": topology,
                        "seed": int(seed),
                        "strategy": strategy,
                        "n_nodes": config.n_nodes,
                        "n_input_nodes": len(config.input_nodes),
                        "n_readout_nodes": len(readout_nodes),
                        "n_dampers": width,
                        "damping_trace": float(np.sum(damping)),
                        "readout_nodes": ",".join(
                            map(str, readout_nodes.tolist())
                        ),
                        "damping_nodes": ",".join(map(str, nodes.tolist())),
                        "calibration_minimum_visibility": cal_modal[
                            "minimum_visibility"
                        ],
                        "unseen_minimum_visibility": eval_modal[
                            "minimum_visibility"
                        ],
                        "unseen_minimum_decay_rate": eval_modal[
                            "minimum_decay_rate"
                        ],
                        "unseen_all_observable": eval_modal[
                            "all_observable"
                        ],
                        "plus_converged": bool(plus["converged"]),
                        "minus_converged": bool(minus["converged"]),
                        "minimum_phase_convergence_fraction": float(
                            min(
                                plus["convergence_fraction"],
                                minus["convergence_fraction"],
                            )
                        ),
                        "total_active_sample_steps": int(
                            plus["active_sample_steps"]
                            + minus["active_sample_steps"]
                        ),
                        "dynamic_vs_exact_centered_gradient_error": relative_error(
                            dynamic, exact_centered
                        ),
                        "dynamic_vs_implicit_gradient_error": relative_error(
                            dynamic, implicit
                        ),
                        "dynamic_vs_implicit_cosine": cosine_similarity(
                            dynamic, implicit
                        ),
                        "exact_centered_vs_implicit_error": exact_bias,
                        "free_exact_residual": float(free_residual),
                        "plus_exact_residual": float(plus_residual),
                        "minus_exact_residual": float(minus_residual),
                    }
                )
                print(
                    f"{topology} seed={seed} {strategy}: "
                    f"unseen_vis={eval_modal['minimum_visibility']:.3g} "
                    f"relax_err={evaluation_rows[-1]['dynamic_vs_exact_centered_gradient_error']:.3g} "
                    f"conv={evaluation_rows[-1]['minimum_phase_convergence_fraction']:.3f}",
                    flush=True,
                )

    evaluation = pd.DataFrame(evaluation_rows)
    random_table = pd.DataFrame(random_rows)
    evaluation.to_csv(
        output_dir / "graph_placement_generalization.csv",
        index=False,
    )
    random_table.to_csv(
        output_dir / "graph_random_spectral_pool.csv",
        index=False,
    )

    # Quantify how the calibration-locked placement compares with the random
    # accessible-node distribution on unseen states.
    percentile_rows: list[dict[str, object]] = []
    for (topology, seed), block in evaluation.groupby(
        ["topology", "seed"], sort=True
    ):
        modal = block[block["strategy"] == "calibration_modal"].iloc[0]
        pool = random_table[
            (random_table["topology"] == topology)
            & (random_table["seed"] == seed)
        ]
        vis_percentile = float(
            np.mean(
                pool["unseen_minimum_visibility"].to_numpy(float)
                <= float(modal["unseen_minimum_visibility"])
            )
        )
        decay_percentile = float(
            np.mean(
                pool["unseen_minimum_decay_rate"].to_numpy(float)
                <= float(modal["unseen_minimum_decay_rate"])
            )
        )
        percentile_rows.append(
            {
                "topology": topology,
                "seed": int(seed),
                "modal_unseen_visibility_random_percentile": vis_percentile,
                "modal_unseen_decay_random_percentile": decay_percentile,
            }
        )
    percentile = pd.DataFrame(percentile_rows)
    percentile.to_csv(
        output_dir / "graph_modal_random_percentiles.csv",
        index=False,
    )

    summary_by_strategy = (
        evaluation.groupby("strategy", as_index=False)
        .agg(
            cases=("seed", "count"),
            observable_fraction=("unseen_all_observable", "mean"),
            median_unseen_visibility=("unseen_minimum_visibility", "median"),
            minimum_unseen_visibility=("unseen_minimum_visibility", "min"),
            median_unseen_decay_rate=("unseen_minimum_decay_rate", "median"),
            minimum_unseen_decay_rate=("unseen_minimum_decay_rate", "min"),
            median_relaxation_gradient_error=(
                "dynamic_vs_exact_centered_gradient_error",
                "median",
            ),
            maximum_relaxation_gradient_error=(
                "dynamic_vs_exact_centered_gradient_error",
                "max",
            ),
            median_minimum_phase_convergence=(
                "minimum_phase_convergence_fraction",
                "median",
            ),
            median_active_sample_steps=(
                "total_active_sample_steps",
                "median",
            ),
        )
        .sort_values("strategy")
    )
    summary_by_strategy.to_csv(
        output_dir / "graph_placement_strategy_summary.csv",
        index=False,
    )

    summary = {
        "version": "1.0",
        "profile": profile,
        "topologies": list(settings["topologies"]),
        "seeds": list(settings["seeds"]),
        "n_calibration_states": int(settings["n_calibration"]),
        "n_unseen_evaluation_states": int(settings["n_evaluation"]),
        "n_random_spectral_sets_per_graph": int(
            settings["n_random_spectral"]
        ),
        "n_dynamic_cases": int(len(evaluation)),
        "all_damping_traces_matched": bool(
            np.allclose(
                evaluation["damping_trace"].to_numpy(float),
                evaluation["damping_trace"].iloc[0],
            )
        ),
        "all_modal_calibration_sets_observable_on_unseen_states": bool(
            evaluation.loc[
                evaluation["strategy"] == "calibration_modal",
                "unseen_all_observable",
            ].all()
        ),
        "median_modal_unseen_visibility_random_percentile": float(
            percentile[
                "modal_unseen_visibility_random_percentile"
            ].median()
        ),
        "median_modal_unseen_decay_random_percentile": float(
            percentile[
                "modal_unseen_decay_random_percentile"
            ].median()
        ),
        "strategy_summary": summary_by_strategy.to_dict(orient="records"),
    }
    (output_dir / "graph_placement_generalization_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    order = list(summary_by_strategy["strategy"])
    positions = np.arange(len(order))
    axes[0].bar(
        positions,
        summary_by_strategy["median_unseen_visibility"],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median unseen minimum visibility")
    axes[0].set_title("Calibration-to-unseen modal visibility")

    axes[1].bar(
        positions,
        summary_by_strategy["median_unseen_decay_rate"],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median unseen spectral decay rate")
    axes[1].set_title("Unseen relaxation bottleneck")

    axes[2].bar(
        positions,
        summary_by_strategy["median_relaxation_gradient_error"],
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("median relaxation-gradient error")
    axes[2].set_title("Finite-budget gradient generation")

    labels = [value.replace("_", "\n") for value in order]
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "graph_placement_generalization.png",
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
        default=Path("results/graph_placement_generalization"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.profile, args.output_dir), indent=2))
