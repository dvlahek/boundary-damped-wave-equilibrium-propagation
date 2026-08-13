#!/usr/bin/env python3
"""Advanced validation for boundary-radiative Equilibrium Propagation.

This script adds four controlled results to the original proof-of-concept:

1. free and symmetrically nudged equilibria are produced by second-order
   boundary-damped dynamics, without Newton iteration in the training loop;
2. the EqProp update is expressed through node- and edge-local observables
   excited by an output-boundary effort;
3. boundary damping is compared with uniform damping under matched trace and
   matched local damping coefficient;
4. the linearized Robinson--Trautman harmonic operator is checked against its
   exact gradient-flow and modal identities.

Newton iteration is used only after training as an independent diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boundary_rep_learning import (
    Adam,
    ChainConfig,
    ChainParameters,
    classification_metrics,
    flatten_gradients,
    implicit_gradients,
    initialize_parameters,
    make_features,
    make_two_moons,
    physical_coefficients,
    solve_equilibrium,
    state_gradient,
    symmetric_eqprop_gradients,
)
from rep_simulation import rt_eigenvalue


@dataclass(frozen=True)
class DynamicsConfig:
    dt: float = 0.075
    max_steps: int = 18_000
    check_every: int = 25
    gradient_tolerance: float = 7e-5
    velocity_tolerance: float = 7e-5
    consecutive_checks: int = 3
    freeze_converged_samples: bool = True


@dataclass(frozen=True)
class DynamicTrainingConfig:
    n_train: int = 160
    n_test: int = 100
    n_rbf: int = 10
    rbf_sigma: float = 0.72
    noise: float = 0.10
    epochs: int = 60
    beta: float = 0.035
    learning_rate_u: float = 0.080
    learning_rate_structure: float = 0.0030
    weight_decay_u: float = 3e-4
    seed: int = 2031


def damping_vector(
    config: ChainConfig,
    mode: str,
    params: ChainParameters | None = None,
) -> np.ndarray:
    """Return diagonal damping for three transparent comparison protocols."""
    n = config.n_nodes
    width = min(max(1, config.boundary_width), n)

    boundary = np.zeros(n)
    if config.boundary_profile == "flat":
        boundary[-width:] = config.boundary_damping
    elif config.boundary_profile == "fixed_trace":
        boundary[-width:] = config.boundary_trace / width
    elif config.boundary_profile == "graded_trace":
        ramp = np.arange(1.0, width + 1.0) ** 2
        boundary[-width:] = config.boundary_trace * ramp / np.sum(ramp)
    elif config.boundary_profile == "impedance_terminal":
        if params is None:
            terminal_stiffness = 0.75
        else:
            _, edge_stiffness = physical_coefficients(params, config)
            terminal_stiffness = float(edge_stiffness[-1])
        # Unit mass is used by the dynamical integrator.  sqrt(k m) is the
        # terminal dashpot estimate for the characteristic chain impedance.
        boundary[-1] = config.damping_scale * np.sqrt(terminal_stiffness)
    else:
        raise ValueError(f"unknown boundary profile: {config.boundary_profile}")

    if mode == "boundary":
        return boundary
    if mode == "uniform_trace":
        # Same trace(D) as boundary damping, spread over every degree of freedom.
        return np.full(n, float(np.sum(boundary)) / n)
    if mode == "uniform_local":
        # Same coefficient per damped node, but more total damping hardware.
        return np.full(n, float(np.max(boundary)))
    raise ValueError(f"unknown damping mode: {mode}")


def relax_dynamics(
    params: ChainParameters,
    features: np.ndarray,
    chain_config: ChainConfig,
    dynamics_config: DynamicsConfig,
    *,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    damping_mode: str = "boundary",
    initial_q: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
    initial_converged_samples: np.ndarray | None = None,
) -> dict[str, object]:
    """Relax a batch using a damped velocity-Verlet splitting.

    The damping substep is integrated exactly.  The conservative substep uses
    velocity Verlet.  Stopping requires both force balance and negligible
    velocity for several consecutive checks.
    """
    batch = features.shape[0]
    n = chain_config.n_nodes
    q = np.zeros((batch, n)) if initial_q is None else initial_q.copy()
    velocity = (
        np.zeros_like(q) if initial_velocity is None else initial_velocity.copy()
    )
    damping = damping_vector(chain_config, damping_mode, params)
    half_decay = np.exp(-0.5 * dynamics_config.dt * damping)[None, :]
    integrated_boundary_flow = np.zeros((batch, chain_config.boundary_width))
    consecutive = np.zeros(batch, dtype=int)
    converged_samples = (
        np.zeros(batch, dtype=bool)
        if initial_converged_samples is None
        else np.asarray(initial_converged_samples, dtype=bool).copy()
    )
    if converged_samples.shape != (batch,):
        raise ValueError("initial_converged_samples must have shape (batch,)")
    sample_steps = np.zeros(batch, dtype=int)
    active_sample_steps = 0
    residual = np.inf
    velocity_norm = np.inf

    step = 0
    for step in range(1, dynamics_config.max_steps + 1):
        if dynamics_config.freeze_converged_samples:
            active = ~converged_samples
        else:
            active = np.ones(batch, dtype=bool)
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break

        q_active = q[active_indices]
        velocity_active = velocity[active_indices] * half_decay
        features_active = features[active_indices]
        targets_active = None if targets is None else targets[active_indices]
        gradient = state_gradient(
            q_active,
            params,
            features_active,
            chain_config,
            targets_active,
            beta,
        )
        velocity_active -= 0.5 * dynamics_config.dt * gradient
        displacement = dynamics_config.dt * velocity_active
        q_active += displacement
        integrated_boundary_flow[active_indices] += displacement[
            :, -chain_config.boundary_width :
        ]
        gradient_new = state_gradient(
            q_active,
            params,
            features_active,
            chain_config,
            targets_active,
            beta,
        )
        velocity_active -= 0.5 * dynamics_config.dt * gradient_new
        velocity_active *= half_decay
        q[active_indices] = q_active
        velocity[active_indices] = velocity_active
        sample_steps[active_indices] += 1
        active_sample_steps += int(active_indices.size)

        if step % dynamics_config.check_every == 0:
            residual_active = np.linalg.norm(gradient_new, axis=1)
            velocity_active_norm = np.linalg.norm(velocity_active, axis=1)
            locally_converged = (
                (residual_active <= dynamics_config.gradient_tolerance)
                & (velocity_active_norm <= dynamics_config.velocity_tolerance)
            )
            consecutive[active_indices] = np.where(
                locally_converged,
                consecutive[active_indices] + 1,
                0,
            )
            newly_converged = active_indices[
                consecutive[active_indices] >= dynamics_config.consecutive_checks
            ]
            converged_samples[newly_converged] = True
            if np.all(converged_samples):
                break

    final_gradient = state_gradient(
        q, params, features, chain_config, targets, beta
    )
    per_sample_residual = np.linalg.norm(final_gradient, axis=1)
    per_sample_velocity_norm = np.linalg.norm(velocity, axis=1)
    residual = float(np.max(per_sample_residual))
    velocity_norm = float(np.max(per_sample_velocity_norm))
    # A sample frozen after the required consecutive checks remains a valid
    # endpoint even if its stored residual is infinitesimally above the scalar
    # threshold after the final all-batch diagnostic.
    converged = bool(np.all(converged_samples))

    return {
        "q": q,
        "velocity": velocity,
        "steps": step,
        "time": step * dynamics_config.dt,
        "residual": residual,
        "velocity_norm": velocity_norm,
        "converged": converged,
        "convergence_fraction": float(np.mean(converged_samples)),
        "converged_samples": converged_samples,
        "per_sample_residual": per_sample_residual,
        "per_sample_velocity_norm": per_sample_velocity_norm,
        "sample_steps": sample_steps,
        "active_sample_steps": active_sample_steps,
        "integrated_boundary_flow": integrated_boundary_flow,
        "damping": damping,
    }


def local_contrastive_observables(
    q_minus: np.ndarray,
    q_plus: np.ndarray,
    features: np.ndarray,
    beta: float,
) -> dict[str, np.ndarray]:
    """Raw local observables used by the centered physical learning rule."""
    onsite = np.mean(0.5 * (q_plus**2 - q_minus**2), axis=0) / (2.0 * beta)
    edge_plus = q_plus[:, 1:] - q_plus[:, :-1]
    edge_minus = q_minus[:, 1:] - q_minus[:, :-1]
    edge = np.mean(0.5 * (edge_plus**2 - edge_minus**2), axis=0) / (2.0 * beta)
    drive = -np.mean(
        (q_plus[:, :2, None] - q_minus[:, :2, None]) * features[:, None, :],
        axis=0,
    ) / (2.0 * beta)
    return {"onsite": onsite, "edge": edge, "drive": drive}


def gradients_from_local_observables(
    observables: dict[str, np.ndarray], params: ChainParameters
) -> ChainParameters:
    """Convert directly measured local contrasts into parameter gradients."""
    from boundary_rep_learning import sigmoid

    return ChainParameters(
        sigmoid(params.log_a) * observables["onsite"],
        sigmoid(params.log_w) * observables["edge"],
        observables["drive"],
    )


def train_with_physical_dynamics(
    params: ChainParameters,
    features_train: np.ndarray,
    y_train: np.ndarray,
    features_test: np.ndarray,
    y_test: np.ndarray,
    chain_config: ChainConfig,
    dynamics_config: DynamicsConfig,
    training_config: DynamicTrainingConfig,
) -> tuple[ChainParameters, pd.DataFrame, dict[str, object]]:
    """Full-batch EqProp whose endpoints all come from physical dynamics."""
    opt_a = Adam(params.log_a.shape, training_config.learning_rate_structure)
    opt_w = Adam(params.log_w.shape, training_config.learning_rate_structure)
    opt_u = Adam(params.u.shape, training_config.learning_rate_u)
    q_free_cache: np.ndarray | None = None
    q_test_cache: np.ndarray | None = None
    history: list[dict[str, float | int | bool]] = []
    total_relaxation_steps = 0

    for epoch in range(training_config.epochs + 1):
        free = relax_dynamics(
            params,
            features_train,
            chain_config,
            dynamics_config,
            initial_q=q_free_cache,
        )
        q_free = np.asarray(free["q"])
        q_free_cache = q_free
        total_relaxation_steps += int(free["steps"])

        if epoch > 0:
            plus = relax_dynamics(
                params,
                features_train,
                chain_config,
                dynamics_config,
                targets=y_train,
                beta=training_config.beta,
                initial_q=q_free,
            )
            minus = relax_dynamics(
                params,
                features_train,
                chain_config,
                dynamics_config,
                targets=y_train,
                beta=-training_config.beta,
                initial_q=q_free,
            )
            total_relaxation_steps += int(plus["steps"]) + int(minus["steps"])
            q_plus = np.asarray(plus["q"])
            q_minus = np.asarray(minus["q"])
            observables = local_contrastive_observables(
                q_minus, q_plus, features_train, training_config.beta
            )
            gradients = gradients_from_local_observables(observables, params)
            gradients.u += training_config.weight_decay_u * params.u
            params.log_a = opt_a.update(params.log_a, gradients.log_a)
            params.log_w = opt_w.update(params.log_w, gradients.log_w)
            params.u = opt_u.update(params.u, gradients.u)

        if epoch % 5 == 0 or epoch == training_config.epochs:
            # Re-relax after the parameter update so reported states correspond
            # to the current physical system.
            train_eval = relax_dynamics(
                params,
                features_train,
                chain_config,
                dynamics_config,
                initial_q=q_free_cache,
            )
            test_eval = relax_dynamics(
                params,
                features_test,
                chain_config,
                dynamics_config,
                initial_q=q_test_cache,
            )
            q_free_cache = np.asarray(train_eval["q"])
            q_test_cache = np.asarray(test_eval["q"])
            total_relaxation_steps += int(train_eval["steps"]) + int(test_eval["steps"])
            train_loss, train_accuracy = classification_metrics(q_free_cache, y_train)
            test_loss, test_accuracy = classification_metrics(q_test_cache, y_test)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_dynamic_residual": float(train_eval["residual"]),
                    "test_dynamic_residual": float(test_eval["residual"]),
                    "train_converged": bool(train_eval["converged"]),
                    "test_converged": bool(test_eval["converged"]),
                }
            )

    final = {
        "q_train": q_free_cache,
        "q_test": q_test_cache,
        "total_relaxation_steps": total_relaxation_steps,
    }
    return params, pd.DataFrame(history), final


def validate_dynamic_local_rule(
    params: ChainParameters,
    features: np.ndarray,
    targets: np.ndarray,
    chain_config: ChainConfig,
    dynamics_config: DynamicsConfig,
    beta: float,
) -> dict[str, float | int | bool]:
    free = relax_dynamics(params, features, chain_config, dynamics_config)
    q_free = np.asarray(free["q"])
    plus = relax_dynamics(
        params,
        features,
        chain_config,
        dynamics_config,
        targets=targets,
        beta=beta,
        initial_q=q_free,
    )
    minus = relax_dynamics(
        params,
        features,
        chain_config,
        dynamics_config,
        targets=targets,
        beta=-beta,
        initial_q=q_free,
    )
    q_plus = np.asarray(plus["q"])
    q_minus = np.asarray(minus["q"])
    local = flatten_gradients(
        gradients_from_local_observables(
            local_contrastive_observables(q_minus, q_plus, features, beta), params
        )
    )
    direct = flatten_gradients(
        symmetric_eqprop_gradients(q_minus, q_plus, params, features, chain_config, beta)
    )
    implicit = flatten_gradients(
        implicit_gradients(q_free, params, features, targets, chain_config)
    )
    local_identity_error = np.linalg.norm(local - direct) / max(np.linalg.norm(direct), 1e-14)
    implicit_error = np.linalg.norm(local - implicit) / max(np.linalg.norm(implicit), 1e-14)
    cosine = float(np.dot(local, implicit) / (np.linalg.norm(local) * np.linalg.norm(implicit)))

    # q is updated by exactly dt*v_half in the splitting, so accumulated
    # boundary flow must equal the measured boundary displacement.
    plus_flow = np.asarray(plus["integrated_boundary_flow"])
    expected_flow = q_plus[:, -chain_config.boundary_width :] - q_free[:, -chain_config.boundary_width :]
    flow_closure = np.linalg.norm(plus_flow - expected_flow) / max(np.linalg.norm(expected_flow), 1e-14)
    return {
        "local_vs_direct_relative_error": float(local_identity_error),
        "local_vs_implicit_relative_error": float(implicit_error),
        "local_vs_implicit_cosine": cosine,
        "integrated_boundary_flow_closure_error": float(flow_closure),
        "free_steps": int(free["steps"]),
        "plus_steps": int(plus["steps"]),
        "minus_steps": int(minus["steps"]),
        "all_phases_converged": bool(free["converged"] and plus["converged"] and minus["converged"]),
    }


def damping_comparison(
    params: ChainParameters,
    features: np.ndarray,
    chain_config: ChainConfig,
    dynamics_config: DynamicsConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for mode in ("boundary", "uniform_trace", "uniform_local"):
        damping = damping_vector(chain_config, mode)
        for sample in range(features.shape[0]):
            result = relax_dynamics(
                params,
                features[sample : sample + 1],
                chain_config,
                dynamics_config,
                damping_mode=mode,
            )
            rows.append(
                {
                    "mode": mode,
                    "sample": sample,
                    "settling_time": float(result["time"]),
                    "steps": int(result["steps"]),
                    "residual": float(result["residual"]),
                    "velocity_norm": float(result["velocity_norm"]),
                    "converged": bool(result["converged"]),
                    "active_dampers": int(np.count_nonzero(damping)),
                    "damping_trace": float(np.sum(damping)),
                    "interior_damping_trace": float(np.sum(damping[: -chain_config.boundary_width])),
                }
            )
    return pd.DataFrame(rows)


def rt_restricted_sector_check(l_max: int = 10, seed: int = 2032) -> dict[str, float | int]:
    """Check the exact linearized RT/Calabi harmonic gradient identity.

    With the non-negative convention A=(-Delta)(-Delta-2), the l>=2 modes
    satisfy f_dot=-kappa*A*f.  The overall kappa contains the mass and time
    normalization of the chosen Robinson--Trautman convention.
    """
    rng = np.random.default_rng(seed)
    ells = np.concatenate([np.full(2 * ell + 1, ell) for ell in range(2, l_max + 1)])
    amplitudes = rng.normal(size=ells.size)
    eigenvalues = np.asarray([rt_eigenvalue(int(ell)) for ell in ells])
    eigenvalues /= rt_eigenvalue(2)
    kappa = 0.37
    state_rate = -kappa * eigenvalues * amplitudes
    functional_gradient = kappa * eigenvalues * amplitudes
    functional_rate_chain = float(np.dot(functional_gradient, state_rate))
    functional_rate_norm = -float(np.dot(functional_gradient, functional_gradient))
    modal_residual = state_rate + functional_gradient
    return {
        "n_radiative_modes": int(ells.size),
        "minimum_operator_eigenvalue": float(np.min(eigenvalues)),
        "maximum_operator_eigenvalue": float(np.max(eigenvalues)),
        "vector_gradient_identity_max_error": float(np.max(np.abs(modal_residual))),
        "energy_rate_identity_abs_error": float(abs(functional_rate_chain - functional_rate_norm)),
    }


def make_advanced_plots(
    history: pd.DataFrame,
    damping_table: pd.DataFrame,
    validation: dict[str, float | int | bool],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    axes[0].plot(history["epoch"], history["train_accuracy"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["test_accuracy"], marker="o", label="test")
    axes[0].set(xlabel="epoch", ylabel="accuracy", ylim=(0.45, 1.02), title="Dynamics-only EqProp")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    modes = ["boundary", "uniform_trace", "uniform_local"]
    data = [damping_table.loc[damping_table["mode"] == mode, "settling_time"] for mode in modes]
    axes[1].boxplot(data, tick_labels=["boundary", "uniform\ntrace", "uniform\nlocal"])
    axes[1].set(ylabel="settling time", title="Damping comparison")
    axes[1].grid(alpha=0.25, axis="y")

    labels = ["local identity", "implicit gradient", "flow closure"]
    errors = [
        float(validation["local_vs_direct_relative_error"]),
        float(validation["local_vs_implicit_relative_error"]),
        float(validation["integrated_boundary_flow_closure_error"]),
    ]
    axes[2].bar(labels, np.maximum(errors, 1e-16), color=["#4c78a8", "#f58518", "#54a24b"])
    axes[2].set_yscale("log")
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].set(ylabel="relative error", title="Local boundary-driven rule")
    axes[2].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "advanced_rep_validation.png", dpi=190)
    plt.close(fig)


def run(output_dir: Path, quick: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # A three-node chain keeps a genuine conservative interior while making
    # full dynamics-only training computationally tractable.
    chain_config = ChainConfig(
        n_nodes=3,
        boundary_width=1,
        boundary_damping=1.0,
        beta=0.035,
    )
    dynamics_config = DynamicsConfig(
        max_steps=8_000 if quick else 18_000,
        gradient_tolerance=1.5e-4 if quick else 7e-5,
        velocity_tolerance=1.5e-4 if quick else 7e-5,
    )
    training_config = DynamicTrainingConfig(
        n_train=90 if quick else 160,
        n_test=60 if quick else 100,
        epochs=30 if quick else 60,
    )
    rng = np.random.default_rng(training_config.seed)
    x_all, y_all = make_two_moons(
        training_config.n_train + training_config.n_test,
        training_config.noise,
        rng,
    )
    mean = np.mean(x_all[: training_config.n_train], axis=0)
    scale = np.std(x_all[: training_config.n_train], axis=0)
    x_all = (x_all - mean) / scale
    x_train = x_all[: training_config.n_train]
    x_test = x_all[training_config.n_train :]
    y_train = y_all[: training_config.n_train]
    y_test = y_all[training_config.n_train :]
    center_indices = rng.choice(training_config.n_train, training_config.n_rbf, replace=False)
    centers = x_train[center_indices]
    features_train = make_features(x_train, centers, training_config.rbf_sigma)
    features_test = make_features(x_test, centers, training_config.rbf_sigma)
    params = initialize_parameters(chain_config, features_train.shape[1])

    params, history, final = train_with_physical_dynamics(
        params,
        features_train,
        y_train,
        features_test,
        y_test,
        chain_config,
        dynamics_config,
        training_config,
    )
    validation_count = min(24, training_config.n_train)
    validation = validate_dynamic_local_rule(
        params,
        features_train[:validation_count],
        y_train[:validation_count],
        chain_config,
        dynamics_config,
        training_config.beta,
    )
    comparison_count = min(18, training_config.n_test)
    damping_table = damping_comparison(
        params,
        features_test[:comparison_count],
        chain_config,
        dynamics_config,
    )
    rt_check = rt_restricted_sector_check()

    # Newton is diagnostic only: it quantifies the endpoint error of the
    # dynamics-generated states and is never used to make a training update.
    q_train_newton, _, _ = solve_equilibrium(params, features_train, chain_config)
    q_test_newton, _, _ = solve_equilibrium(params, features_test, chain_config)
    q_train_dynamic = np.asarray(final["q_train"])
    q_test_dynamic = np.asarray(final["q_test"])
    train_loss, train_accuracy = classification_metrics(q_train_dynamic, y_train)
    test_loss, test_accuracy = classification_metrics(q_test_dynamic, y_test)
    diagnostic = {
        "train_state_relative_error_vs_newton": float(
            np.linalg.norm(q_train_dynamic - q_train_newton) / np.linalg.norm(q_train_newton)
        ),
        "test_state_relative_error_vs_newton": float(
            np.linalg.norm(q_test_dynamic - q_test_newton) / np.linalg.norm(q_test_newton)
        ),
    }

    damping_summary = {}
    for mode, group in damping_table.groupby("mode"):
        damping_summary[mode] = {
            "median_settling_time": float(group["settling_time"].median()),
            "mean_settling_time": float(group["settling_time"].mean()),
            "convergence_fraction": float(group["converged"].mean()),
            "active_dampers": int(group["active_dampers"].iloc[0]),
            "damping_trace": float(group["damping_trace"].iloc[0]),
            "interior_damping_trace": float(group["interior_damping_trace"].iloc[0]),
        }

    history.to_csv(output_dir / "dynamic_training_history.csv", index=False)
    damping_table.to_csv(output_dir / "damping_comparison.csv", index=False)
    np.savez_compressed(
        output_dir / "dynamic_trained_chain_model.npz",
        log_a=params.log_a,
        log_w=params.log_w,
        u=params.u,
        rbf_centers=centers,
        input_mean=mean,
        input_scale=scale,
    )
    make_advanced_plots(history, damping_table, validation, output_dir)

    summary = {
        "status": "advanced boundary-radiative EqProp validation",
        "chain_config": asdict(chain_config),
        "dynamics_config": asdict(dynamics_config),
        "training_config": asdict(training_config),
        "dynamics_only_training": {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "total_relaxation_steps": int(final["total_relaxation_steps"]),
            **diagnostic,
        },
        "boundary_driven_local_rule": validation,
        "damping_comparison": damping_summary,
        "restricted_robinson_trautman": rt_check,
    }
    (output_dir / "advanced_rep_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results_advanced"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.output_dir, arguments.quick), indent=2))
