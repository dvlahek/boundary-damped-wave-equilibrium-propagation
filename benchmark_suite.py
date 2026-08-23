#!/usr/bin/env python3
"""Scalable benchmark suite for boundary-radiative Equilibrium Propagation.

The suite compares four methods on identical datasets, features, parameters,
and optimization schedules:

* boundary_dynamic: second-order dynamics with damping only at the boundary;
* uniform_dynamic: second-order dynamics with the same damping trace spread
  over all coordinates;
* standard_eqprop: first-order gradient-flow relaxation;
* implicit_id: Newton equilibria with exact implicit differentiation.

Profiles are deliberately explicit.  ``quick`` checks the installation,
``standard`` is a moderate experiment, and ``paper`` is the full multi-seed
study.  Results are checkpointed after every run and can be resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-benchmark"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon

from advanced_rep_experiments import DynamicsConfig, damping_vector, relax_dynamics
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
    solve_equilibrium,
    state_gradient,
    state_hessian,
    symmetric_eqprop_gradients,
)


METHODS = ("boundary_dynamic", "uniform_dynamic", "standard_eqprop", "implicit_id")
DATASETS = ("moons", "circles", "xor", "spirals", "blobs")
DAMPING_SCHEDULES = ("legacy", "fixed_trace", "graded_trace", "impedance_terminal")
CHAIN_REGIMES = ("legacy", "propagating")
PHASE_PROTOCOLS = (
    "free_each_epoch",
    "centered_pair_cache",
    "exact_free_boundary_pair",
)


@dataclass(frozen=True)
class Profile:
    datasets: tuple[str, ...]
    chain_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    n_train: int
    n_test: int
    n_rbf: int
    epochs: int
    max_steps: int
    tolerance: float


PROFILES = {
    "quick": Profile(
        datasets=("moons", "circles"),
        chain_sizes=(3, 5),
        seeds=(101,),
        methods=METHODS,
        n_train=70,
        n_test=40,
        n_rbf=8,
        epochs=3,
        max_steps=3_000,
        tolerance=8e-4,
    ),
    "standard": Profile(
        datasets=("moons", "circles", "xor", "spirals"),
        chain_sizes=(3, 5, 7),
        seeds=(101, 202, 303),
        methods=METHODS,
        n_train=180,
        n_test=100,
        n_rbf=14,
        epochs=25,
        max_steps=20_000,
        tolerance=1.5e-4,
    ),
    "pilot": Profile(
        # Training-time gate on the chain sizes that were problematic in the
        # legacy benchmark.  These seeds are disjoint from the paper seeds.
        datasets=("moons", "circles", "xor"),
        chain_sizes=(7, 9),
        seeds=(17,),
        methods=("boundary_dynamic",),
        n_train=96,
        n_test=64,
        n_rbf=12,
        epochs=10,
        max_steps=40_000,
        tolerance=1e-5,
    ),
    "extended_pilot": Profile(
        # Full-training convergence gate on the difficult long chains.  The
        # seeds are disjoint from the paper study, while sample counts,
        # feature dimension, epochs, and strict endpoint tolerance match it.
        datasets=DATASETS,
        chain_sizes=(7, 9),
        seeds=(17, 29),
        methods=("boundary_dynamic",),
        n_train=280,
        n_test=160,
        n_rbf=18,
        epochs=50,
        max_steps=40_000,
        tolerance=7e-6,
    ),
    "paper": Profile(
        datasets=DATASETS,
        chain_sizes=(3, 5, 7, 9),
        seeds=(101, 202, 303, 404, 505),
        methods=METHODS,
        n_train=280,
        n_test=160,
        n_rbf=18,
        epochs=50,
        max_steps=40_000,
        # EqProp subtracts two nearby endpoints and divides by beta.  The old
        # 7e-5 state tolerance was therefore too loose and produced order-10%
        # dynamic gradient errors.  Calibration gives sub-1% median error at
        # this stricter endpoint tolerance.
        tolerance=7e-6,
    ),
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    chain_size: int
    seed: int
    method: str
    n_train: int
    n_test: int
    n_rbf: int
    n_input_nodes: int
    rbf_sigma: float
    epochs: int
    max_steps: int
    tolerance: float
    beta: float = 0.035
    damping_schedule: str = "fixed_trace"
    damping_trace: float = 1.00
    damping_scale: float = 1.0
    endpoint_step_multiplier: int = 10
    dynamics_dt: float = 0.20
    free_tolerance_multiplier: float = 100.0
    learning_rate_u: float = 0.0025
    learning_rate_structure: float = 0.0003
    chain_regime: str = "propagating"
    phase_protocol: str = "centered_pair_cache"

    @property
    def key(self) -> str:
        return (
            f"{self.dataset}|n={self.chain_size}|seed={self.seed}|{self.method}"
            f"|train={self.n_train}|test={self.n_test}|rbf={self.n_rbf}"
            f"|inputs={self.n_input_nodes}|rbfsigma={self.rbf_sigma:.4g}"
            f"|epochs={self.epochs}|steps={self.max_steps}"
            f"|tol={self.tolerance:.3g}|beta={self.beta:.3g}"
            f"|damping={self.damping_schedule}|stepmult={self.endpoint_step_multiplier}"
            f"|dt={self.dynamics_dt:.4g}"
            f"|freetolmult={self.free_tolerance_multiplier:.4g}"
            f"|trace={self.damping_trace:.4g}|dscale={self.damping_scale:.4g}"
            f"|lru={self.learning_rate_u:.4g}|lrs={self.learning_rate_structure:.4g}"
            f"|regime={self.chain_regime}"
            f"|protocol={self.phase_protocol}"
        )


def make_dataset(
    name: str, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if name == "moons":
        return make_two_moons(n_samples, 0.11, rng)

    if name == "circles":
        n_first = n_samples // 2
        n_second = n_samples - n_first
        angle_first = rng.uniform(0.0, 2.0 * np.pi, n_first)
        angle_second = rng.uniform(0.0, 2.0 * np.pi, n_second)
        first = 0.70 * np.column_stack((np.cos(angle_first), np.sin(angle_first)))
        second = 1.45 * np.column_stack((np.cos(angle_second), np.sin(angle_second)))
        x = np.vstack((first, second)) + rng.normal(scale=0.10, size=(n_samples, 2))
        y = np.concatenate((-np.ones(n_first), np.ones(n_second)))
    elif name == "xor":
        x = rng.uniform(-1.4, 1.4, size=(n_samples, 2))
        x += rng.normal(scale=0.08, size=x.shape)
        y = np.where(x[:, 0] * x[:, 1] >= 0.0, 1.0, -1.0)
    elif name == "spirals":
        n_first = n_samples // 2
        n_second = n_samples - n_first
        t_first = rng.uniform(0.2, 3.6 * np.pi, n_first)
        t_second = rng.uniform(0.2, 3.6 * np.pi, n_second)
        r_first = 0.13 * t_first
        r_second = 0.13 * t_second
        first = np.column_stack((r_first * np.cos(t_first), r_first * np.sin(t_first)))
        second = np.column_stack(
            (r_second * np.cos(t_second + np.pi), r_second * np.sin(t_second + np.pi))
        )
        x = np.vstack((first, second)) + rng.normal(scale=0.10, size=(n_samples, 2))
        y = np.concatenate((-np.ones(n_first), np.ones(n_second)))
    elif name == "blobs":
        n_first = n_samples // 2
        n_second = n_samples - n_first
        first = rng.normal(loc=(-0.8, -0.35), scale=(0.55, 0.65), size=(n_first, 2))
        second = rng.normal(loc=(0.8, 0.35), scale=(0.55, 0.65), size=(n_second, 2))
        x = np.vstack((first, second))
        y = np.concatenate((-np.ones(n_first), np.ones(n_second)))
    else:
        raise ValueError(f"unknown dataset: {name}")

    order = rng.permutation(n_samples)
    return x[order], y[order]


def chain_configuration(
    n_nodes: int,
    seed: int,
    beta: float,
    n_input_nodes: int = 2,
    damping_schedule: str = "fixed_trace",
    damping_trace: float = 1.00,
    damping_scale: float = 1.0,
    chain_regime: str = "propagating",
) -> ChainConfig:
    if damping_schedule not in DAMPING_SCHEDULES:
        raise ValueError(f"unknown damping schedule: {damping_schedule}")
    if chain_regime not in CHAIN_REGIMES:
        raise ValueError(f"unknown chain regime: {chain_regime}")
    if not 1 <= n_input_nodes < n_nodes:
        raise ValueError("n_input_nodes must be between 1 and n_nodes - 1")
    boundary_width = max(1, int(round(n_nodes / 3.0)))
    if damping_schedule == "legacy":
        boundary_damping = (3.0 / n_nodes) ** 2
        boundary_profile = "flat"
    elif damping_schedule == "fixed_trace":
        boundary_damping = 1.0 / boundary_width
        boundary_profile = "fixed_trace"
    elif damping_schedule == "graded_trace":
        boundary_damping = 1.0
        boundary_profile = "graded_trace"
    else:
        boundary_width = 1
        boundary_damping = float(np.sqrt(0.75))
        boundary_profile = "impedance_terminal"
    if chain_regime == "legacy":
        stiffness_floor = 0.20
        edge_floor = 0.45
        initial_onsite_stiffness = 0.85
        initial_edge_stiffness = 0.75
    else:
        # The legacy a/w ratio attenuates a static signal exponentially and
        # makes q_out almost zero for n=9.  This lower onsite-to-edge ratio is
        # still strictly convex but represents a chain designed to transmit.
        stiffness_floor = 0.05
        edge_floor = 0.45
        initial_onsite_stiffness = 0.25
        initial_edge_stiffness = 1.00
    return ChainConfig(
        n_nodes=n_nodes,
        n_input_nodes=n_input_nodes,
        stiffness_floor=stiffness_floor,
        edge_floor=edge_floor,
        boundary_width=boundary_width,
        boundary_damping=boundary_damping,
        boundary_profile=boundary_profile,
        boundary_trace=damping_trace,
        damping_scale=damping_scale,
        initial_onsite_stiffness=initial_onsite_stiffness,
        initial_edge_stiffness=initial_edge_stiffness,
        beta=beta,
        seed=seed + 10_000,
    )


def prepare_data(spec: RunSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    x, y = make_dataset(spec.dataset, spec.n_train + spec.n_test, rng)
    mean = np.mean(x[: spec.n_train], axis=0)
    scale = np.std(x[: spec.n_train], axis=0)
    x = (x - mean) / np.maximum(scale, 1e-12)
    x_train = x[: spec.n_train]
    x_test = x[spec.n_train :]
    y_train = y[: spec.n_train]
    y_test = y[spec.n_train :]
    center_indices = rng.choice(spec.n_train, spec.n_rbf, replace=False)
    centers = x_train[center_indices]
    return (
        make_features(x_train, centers, spec.rbf_sigma),
        y_train,
        make_features(x_test, centers, spec.rbf_sigma),
        y_test,
    )


def extended_classification_metrics(
    q: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    output = q[:, -1]
    prediction = np.where(output >= 0.0, 1.0, -1.0)
    recalls = []
    for label in (-1.0, 1.0):
        mask = targets == label
        recalls.append(float(np.mean(prediction[mask] == label)))
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "mean_signed_margin": float(np.mean(targets * output)),
        "median_signed_margin": float(np.median(targets * output)),
        "output_rms": float(np.sqrt(np.mean(output**2))),
    }


def overdamped_relax(
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    *,
    targets: np.ndarray | None,
    beta: float,
    initial_q: np.ndarray | None,
    max_steps: int,
    tolerance: float,
) -> dict[str, object]:
    q = np.zeros((features.shape[0], config.n_nodes)) if initial_q is None else initial_q.copy()
    consecutive = 0
    residual = np.inf
    for step in range(1, max_steps + 1):
        gradient = state_gradient(q, params, features, config, targets, beta)
        hessian = state_hessian(q, params, config, beta)
        # Gershgorin row-sum bound supplies a stable local gradient-flow step.
        lipschitz = float(np.max(np.sum(np.abs(hessian), axis=2)))
        q -= (0.85 / max(lipschitz, 1e-8)) * gradient
        if step % 10 == 0:
            residual = float(
                np.max(np.linalg.norm(state_gradient(q, params, features, config, targets, beta), axis=1))
            )
            if residual <= tolerance:
                consecutive += 1
                if consecutive >= 2:
                    break
            else:
                consecutive = 0
    return {
        "q": q,
        "steps": step,
        "residual": residual,
        "converged": consecutive >= 2,
    }


def endpoint(
    method: str,
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    spec: RunSpec,
    *,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial_q: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
    initial_converged_samples: np.ndarray | None = None,
    max_steps: int | None = None,
    tolerance: float | None = None,
) -> dict[str, object]:
    step_limit = spec.max_steps if max_steps is None else max_steps
    endpoint_tolerance = spec.tolerance if tolerance is None else tolerance
    if method in ("boundary_dynamic", "uniform_dynamic"):
        mode = "boundary" if method == "boundary_dynamic" else "uniform_trace"
        result = relax_dynamics(
            params,
            features,
            config,
            DynamicsConfig(
                dt=spec.dynamics_dt,
                max_steps=step_limit,
                gradient_tolerance=endpoint_tolerance,
                velocity_tolerance=endpoint_tolerance,
            ),
            targets=targets,
            beta=beta,
            damping_mode=mode,
            initial_q=initial_q,
            initial_velocity=initial_velocity,
            initial_converged_samples=initial_converged_samples,
        )
        return result
    if method == "standard_eqprop":
        return overdamped_relax(
            params,
            features,
            config,
            targets=targets,
            beta=beta,
            initial_q=initial_q,
            max_steps=step_limit,
            tolerance=endpoint_tolerance,
        )
    q, iterations, residual = solve_equilibrium(
        params,
        features,
        config,
        targets=targets,
        beta=beta,
        initial=initial_q,
    )
    return {"q": q, "steps": iterations, "residual": residual, "converged": True}


def run_one(
    spec: RunSpec,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    start_time = time.perf_counter()
    features_train, y_train, features_test, y_test = prepare_data(spec)
    config = chain_configuration(
        spec.chain_size,
        spec.seed,
        spec.beta,
        spec.n_input_nodes,
        spec.damping_schedule,
        spec.damping_trace,
        spec.damping_scale,
        spec.chain_regime,
    )
    params = initialize_parameters(config, features_train.shape[1])
    initial_damping_trace = float(np.sum(damping_vector(config, "boundary", params)))
    opt_a = Adam(params.log_a.shape, spec.learning_rate_structure)
    opt_w = Adam(params.log_w.shape, spec.learning_rate_structure)
    opt_u = Adam(params.u.shape, spec.learning_rate_u)
    q_free_cache: np.ndarray | None = None
    q_plus_cache: np.ndarray | None = None
    q_minus_cache: np.ndarray | None = None
    total_steps = 0
    total_active_sample_steps = 0
    endpoint_calls = 0
    endpoint_attempts = 0
    converged_calls = 0
    failed_endpoint_calls = 0
    exact_free_solver_calls = 0
    exact_free_solver_iterations = 0
    gradient_relative_errors: list[float] = []
    gradient_cosines: list[float] = []
    history: list[dict[str, object]] = []
    endpoint_diagnostics: list[dict[str, object]] = []
    failure_epoch: int | None = None
    failure_phase: str | None = None

    def dynamic_endpoint(
        *, epoch: int, phase: str, **kwargs
    ) -> dict[str, object]:
        # The centered EqProp update is formed only from the converged q+ and q-
        # endpoints.  The free phase supplies a prediction and a warm start, so
        # it may use a separately registered tolerance without weakening the
        # strict tolerances applied to either gradient-forming endpoint.
        phase_tolerance = spec.tolerance * (
            spec.free_tolerance_multiplier if phase == "free" else 1.0
        )
        result = endpoint(
            spec.method,
            params,
            features_train,
            config,
            spec,
            tolerance=phase_tolerance,
            **kwargs,
        )
        cumulative_steps = int(result["steps"])
        cumulative_active_sample_steps = int(result.get("active_sample_steps", 0))
        attempts = 1

        def record_attempt(attempt_result: dict[str, object], attempt: int) -> None:
            endpoint_diagnostics.append(
                {
                    "run_key": spec.key,
                    "dataset": spec.dataset,
                    "chain_size": spec.chain_size,
                    "seed": spec.seed,
                    "method": spec.method,
                    "epoch": epoch,
                    "phase": phase,
                    "attempt": attempt,
                    "steps": int(attempt_result["steps"]),
                    "active_sample_steps": int(
                        attempt_result.get("active_sample_steps", 0)
                    ),
                    "residual": float(attempt_result.get("residual", np.nan)),
                    "velocity_norm": float(
                        attempt_result.get("velocity_norm", np.nan)
                    ),
                    "sample_convergence_fraction": float(
                        attempt_result.get(
                            "convergence_fraction",
                            float(bool(attempt_result["converged"])),
                        )
                    ),
                    "converged": bool(attempt_result["converged"]),
                    "max_steps": spec.max_steps,
                    "tolerance": phase_tolerance,
                    "base_tolerance": spec.tolerance,
                    "free_tolerance_multiplier": spec.free_tolerance_multiplier,
                }
            )

        record_attempt(result, attempts)
        if spec.method in ("boundary_dynamic", "uniform_dynamic"):
            for _ in range(1, spec.endpoint_step_multiplier):
                if bool(result["converged"]):
                    break
                result = endpoint(
                    spec.method,
                    params,
                    features_train,
                    config,
                    spec,
                    targets=kwargs.get("targets"),
                    beta=kwargs.get("beta", 0.0),
                    initial_q=np.asarray(result["q"]),
                    initial_velocity=np.asarray(result["velocity"]),
                    initial_converged_samples=np.asarray(
                        result["converged_samples"], dtype=bool
                    ),
                    tolerance=phase_tolerance,
                )
                cumulative_steps += int(result["steps"])
                cumulative_active_sample_steps += int(
                    result.get("active_sample_steps", 0)
                )
                attempts += 1
                record_attempt(result, attempts)
        result = dict(result)
        result["steps"] = cumulative_steps
        result["active_sample_steps"] = cumulative_active_sample_steps
        result["attempts"] = attempts
        return result

    def register_endpoint(result: dict[str, object]) -> None:
        nonlocal total_steps, endpoint_calls, endpoint_attempts
        nonlocal converged_calls, failed_endpoint_calls
        nonlocal total_active_sample_steps
        total_steps += int(result["steps"])
        total_active_sample_steps += int(result.get("active_sample_steps", 0))
        endpoint_calls += 1
        endpoint_attempts += int(result["attempts"])
        converged_calls += int(bool(result["converged"]))
        failed_endpoint_calls += int(not bool(result["converged"]))

    for epoch in range(spec.epochs + 1):
        if spec.phase_protocol == "exact_free_boundary_pair":
            # Auxiliary gradient-usability control: remove long-chain free-phase
            # relaxation as a confounder, but still obtain both gradient-forming
            # nudged endpoints from the selected physical dynamics.  Recompute
            # q0 after every parameter update so q+ and q- start at the exact
            # free equilibrium of the current model.
            q_free_cache, free_iterations, _ = solve_equilibrium(
                params,
                features_train,
                config,
                initial=q_free_cache,
            )
            exact_free_solver_calls += 1
            exact_free_solver_iterations += int(free_iterations)
        else:
            needs_free_endpoint = (
                epoch == 0
                or spec.method == "implicit_id"
                or spec.phase_protocol == "free_each_epoch"
            )
        if spec.phase_protocol != "exact_free_boundary_pair" and needs_free_endpoint:
            free = dynamic_endpoint(
                epoch=epoch,
                phase="free",
                initial_q=q_free_cache,
            )
            register_endpoint(free)
            q_free_cache = np.asarray(free["q"])
            if not bool(free["converged"]):
                failure_epoch = epoch
                failure_phase = "free"
                break
        if q_free_cache is None:
            raise RuntimeError("free initialization cache was not created")
        q_free = q_free_cache

        if epoch > 0:
            if spec.method == "implicit_id":
                gradients = implicit_gradients(q_free, params, features_train, y_train, config)
            else:
                plus_initial = (
                    q_plus_cache
                    if spec.phase_protocol == "centered_pair_cache"
                    and q_plus_cache is not None
                    else q_free
                )
                minus_initial = (
                    q_minus_cache
                    if spec.phase_protocol == "centered_pair_cache"
                    and q_minus_cache is not None
                    else q_free
                )
                plus = dynamic_endpoint(
                    epoch=epoch,
                    phase="plus",
                    targets=y_train,
                    beta=spec.beta,
                    initial_q=plus_initial,
                )
                minus = dynamic_endpoint(
                    epoch=epoch,
                    phase="minus",
                    targets=y_train,
                    beta=-spec.beta,
                    initial_q=minus_initial,
                )
                register_endpoint(plus)
                register_endpoint(minus)
                if not bool(plus["converged"]) or not bool(minus["converged"]):
                    failure_epoch = epoch
                    failed_phases = [
                        name
                        for name, value in (("plus", plus), ("minus", minus))
                        if not bool(value["converged"])
                    ]
                    failure_phase = "+".join(failed_phases)
                    # Never form a contrastive gradient from invalid endpoints.
                    break
                q_plus_cache = np.asarray(plus["q"])
                q_minus_cache = np.asarray(minus["q"])
                gradients = symmetric_eqprop_gradients(
                    q_minus_cache,
                    q_plus_cache,
                    params,
                    features_train,
                    config,
                    spec.beta,
                )
                if spec.phase_protocol == "exact_free_boundary_pair":
                    # Quantify the numerical effect of the dynamic endpoint
                    # tolerance against the exact centered-EqProp estimator at
                    # the same beta and current parameters.  This turns the
                    # relaxed stopping threshold into an auditable quantity.
                    exact_plus, _, _ = solve_equilibrium(
                        params,
                        features_train,
                        config,
                        targets=y_train,
                        beta=spec.beta,
                        initial=q_plus_cache,
                    )
                    exact_minus, _, _ = solve_equilibrium(
                        params,
                        features_train,
                        config,
                        targets=y_train,
                        beta=-spec.beta,
                        initial=q_minus_cache,
                    )
                    exact_gradients = symmetric_eqprop_gradients(
                        exact_minus,
                        exact_plus,
                        params,
                        features_train,
                        config,
                        spec.beta,
                    )
                    dynamic_vector = flatten_gradients(gradients)
                    exact_vector = flatten_gradients(exact_gradients)
                    exact_norm = float(np.linalg.norm(exact_vector))
                    dynamic_norm = float(np.linalg.norm(dynamic_vector))
                    relative_error = float(
                        np.linalg.norm(dynamic_vector - exact_vector)
                        / max(exact_norm, np.finfo(float).eps)
                    )
                    cosine = float(
                        np.dot(dynamic_vector, exact_vector)
                        / max(
                            dynamic_norm * exact_norm,
                            np.finfo(float).eps,
                        )
                    )
                    gradient_relative_errors.append(relative_error)
                    gradient_cosines.append(cosine)
            gradients.u += 3e-4 * params.u
            params.log_a = opt_a.update(params.log_a, gradients.log_a)
            params.log_w = opt_w.update(params.log_w, gradients.log_w)
            params.u = opt_u.update(params.u, gradients.u)

        if epoch % 5 == 0 or epoch == spec.epochs:
            diagnostic_train, _, _ = solve_equilibrium(params, features_train, config)
            diagnostic_test, _, _ = solve_equilibrium(params, features_test, config)
            train_loss, train_accuracy = classification_metrics(diagnostic_train, y_train)
            test_loss, test_accuracy = classification_metrics(diagnostic_test, y_test)
            history.append(
                {
                    "run_key": spec.key,
                    "dataset": spec.dataset,
                    "chain_size": spec.chain_size,
                    "seed": spec.seed,
                    "method": spec.method,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                }
            )

    run_status = "completed" if failure_phase is None else "failed_endpoint"

    q_train, _, train_residual = solve_equilibrium(params, features_train, config)
    q_test, _, test_residual = solve_equilibrium(params, features_test, config)
    train_loss, train_accuracy = classification_metrics(q_train, y_train)
    test_loss, test_accuracy = classification_metrics(q_test, y_test)
    train_extended = extended_classification_metrics(q_train, y_train)
    test_extended = extended_classification_metrics(q_test, y_test)
    elapsed = time.perf_counter() - start_time
    result = {
        "run_key": spec.key,
        **asdict(spec),
        "boundary_width": config.boundary_width,
        "boundary_damping": config.boundary_damping,
        "boundary_profile": config.boundary_profile,
        "initial_damping_trace": initial_damping_trace,
        "final_damping_trace": float(np.sum(damping_vector(config, "boundary", params))),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        **{f"train_{key}": value for key, value in train_extended.items()},
        **{f"test_{key}": value for key, value in test_extended.items()},
        "train_equilibrium_residual": train_residual,
        "test_equilibrium_residual": test_residual,
        "total_relaxation_steps": total_steps,
        "total_active_sample_steps": total_active_sample_steps,
        "endpoint_calls": endpoint_calls,
        "endpoint_attempts": endpoint_attempts,
        "failed_endpoint_calls": failed_endpoint_calls,
        "exact_free_solver_calls": exact_free_solver_calls,
        "exact_free_solver_iterations": exact_free_solver_iterations,
        "gradient_relative_error_mean": (
            float(np.mean(gradient_relative_errors))
            if gradient_relative_errors
            else np.nan
        ),
        "gradient_relative_error_max": (
            float(np.max(gradient_relative_errors))
            if gradient_relative_errors
            else np.nan
        ),
        "gradient_cosine_mean": (
            float(np.mean(gradient_cosines)) if gradient_cosines else np.nan
        ),
        "gradient_cosine_min": (
            float(np.min(gradient_cosines)) if gradient_cosines else np.nan
        ),
        "endpoint_convergence_fraction": converged_calls / max(endpoint_calls, 1),
        "run_status": run_status,
        "failure_epoch": failure_epoch,
        "failure_phase": failure_phase,
        "valid_dynamic_run": run_status == "completed" and failed_endpoint_calls == 0,
        "wall_time_seconds": elapsed,
    }
    return result, history, endpoint_diagnostics


def bootstrap_interval(values: np.ndarray, seed: int = 991, draws: int = 2_000) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def aggregate_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = results.copy()
    if "run_status" in valid.columns:
        valid = valid.loc[valid["run_status"] == "completed"]
    if "valid_dynamic_run" in valid.columns:
        valid = valid.loc[valid["valid_dynamic_run"].astype(bool)]
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "total_active_sample_steps" not in valid.columns:
        valid["total_active_sample_steps"] = 0
    for column in (
        "test_balanced_accuracy",
        "test_mean_signed_margin",
        "test_output_rms",
    ):
        if column not in valid.columns:
            valid[column] = np.nan
    summary = (
        valid.groupby(["method", "dataset", "chain_size"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
            test_loss_mean=("test_loss", "mean"),
            test_mean_signed_margin=("test_mean_signed_margin", "mean"),
            test_output_rms_mean=("test_output_rms", "mean"),
            wall_time_mean=("wall_time_seconds", "mean"),
            relaxation_steps_mean=("total_relaxation_steps", "mean"),
            active_sample_steps_mean=("total_active_sample_steps", "mean"),
            convergence_fraction_mean=("endpoint_convergence_fraction", "mean"),
        )
    )
    rows: list[dict[str, object]] = []
    block_p_values: list[float] = []
    key_columns = ["dataset", "chain_size", "seed"]
    boundary = valid.loc[valid["method"] == "boundary_dynamic"].set_index(key_columns)
    for method in sorted(set(valid["method"]) - {"boundary_dynamic"}):
        baseline = valid.loc[valid["method"] == method].set_index(key_columns)
        common = boundary.index.intersection(baseline.index)
        if len(common) == 0:
            continue
        accuracy_difference = (
            boundary.loc[common, "test_accuracy"].to_numpy()
            - baseline.loc[common, "test_accuracy"].to_numpy()
        )
        loss_difference = (
            boundary.loc[common, "test_loss"].to_numpy()
            - baseline.loc[common, "test_loss"].to_numpy()
        )
        time_ratio = (
            boundary.loc[common, "wall_time_seconds"].to_numpy()
            / np.maximum(baseline.loc[common, "wall_time_seconds"].to_numpy(), 1e-12)
        )
        low, high = bootstrap_interval(accuracy_difference)
        loss_low, loss_high = bootstrap_interval(loss_difference, seed=1291)
        try:
            p_value = float(wilcoxon(accuracy_difference).pvalue) if np.any(accuracy_difference) else 1.0
        except ValueError:
            p_value = np.nan
        difference_table = pd.DataFrame(
            {
                "dataset": [item[0] for item in common],
                "chain_size": [item[1] for item in common],
                "seed": [item[2] for item in common],
                "accuracy_difference": accuracy_difference,
            }
        )
        block_difference = (
            difference_table.groupby(["dataset", "seed"], as_index=False)[
                "accuracy_difference"
            ]
            .mean()["accuracy_difference"]
            .to_numpy()
        )
        block_low, block_high = bootstrap_interval(block_difference, seed=1991)
        try:
            block_p = (
                float(wilcoxon(block_difference).pvalue)
                if np.any(block_difference)
                else 1.0
            )
        except ValueError:
            block_p = np.nan
        block_p_values.append(block_p)
        rows.append(
            {
                "comparison": f"boundary_dynamic - {method}",
                "n_pairs": len(common),
                "mean_test_accuracy_difference": float(np.mean(accuracy_difference)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "accuracy_equivalence_margin": 0.02,
                "accuracy_equivalent_within_2pp": bool(
                    low >= -0.02 and high <= 0.02
                ),
                "accuracy_noninferior_within_2pp": bool(low >= -0.02),
                "mean_test_loss_difference": float(np.mean(loss_difference)),
                "loss_bootstrap_95_low": loss_low,
                "loss_bootstrap_95_high": loss_high,
                "wilcoxon_two_sided_p": p_value,
                "n_dataset_seed_blocks": len(block_difference),
                "block_bootstrap_95_low": block_low,
                "block_bootstrap_95_high": block_high,
                "block_wilcoxon_two_sided_p": block_p,
                "median_boundary_to_baseline_wall_time_ratio": float(np.median(time_ratio)),
            }
        )
    for row, adjusted in zip(rows, holm_adjust(block_p_values)):
        row["block_wilcoxon_holm_p"] = adjusted
    return summary, pd.DataFrame(rows)


def completion_tables(
    specs: list[RunSpec],
    results: pd.DataFrame,
    failures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = pd.DataFrame(
        [
            {
                "run_key": spec.key,
                "method": spec.method,
                "dataset": spec.dataset,
                "chain_size": spec.chain_size,
                "seed": spec.seed,
            }
            for spec in specs
        ]
    )
    valid_keys = set(results.get("run_key", pd.Series(dtype=str)).astype(str))
    failed_keys = set(failures.get("run_key", pd.Series(dtype=str)).astype(str))
    expected["outcome"] = np.select(
        [
            expected["run_key"].isin(valid_keys),
            expected["run_key"].isin(failed_keys),
        ],
        ["valid", "failed"],
        default="pending",
    )
    expected["is_valid"] = expected["outcome"].eq("valid").astype(int)
    expected["is_failed"] = expected["outcome"].eq("failed").astype(int)
    expected["is_pending"] = expected["outcome"].eq("pending").astype(int)
    completion = (
        expected.groupby(["method", "dataset", "chain_size"], as_index=False)
        .agg(
            n_requested=("run_key", "size"),
            n_valid=("is_valid", "sum"),
            n_failed=("is_failed", "sum"),
            n_pending=("is_pending", "sum"),
        )
    )
    completion["completion_fraction"] = (
        completion["n_valid"] / completion["n_requested"].clip(lower=1)
    )

    comparisons: list[dict[str, object]] = []
    scopes: list[tuple[str, int | None]] = [("overall", None)] + [
        ("chain_size", int(size)) for size in sorted(expected["chain_size"].unique())
    ]
    for scope, chain_size in scopes:
        scoped = expected if chain_size is None else expected.loc[
            expected["chain_size"] == chain_size
        ]
        boundary = scoped.loc[scoped["method"] == "boundary_dynamic"]
        if boundary.empty:
            continue
        boundary_failed = int(boundary["is_failed"].sum())
        boundary_valid = int(boundary["is_valid"].sum())
        for method in sorted(set(scoped["method"]) - {"boundary_dynamic"}):
            baseline = scoped.loc[scoped["method"] == method]
            if baseline.empty:
                continue
            baseline_failed = int(baseline["is_failed"].sum())
            baseline_valid = int(baseline["is_valid"].sum())
            _, p_value = fisher_exact(
                [
                    [boundary_failed, boundary_valid],
                    [baseline_failed, baseline_valid],
                ],
                alternative="greater",
            )
            comparisons.append(
                {
                    "scope": scope,
                    "chain_size": chain_size,
                    "comparison": f"boundary_dynamic vs {method}",
                    "boundary_valid": boundary_valid,
                    "boundary_failed": boundary_failed,
                    "baseline_valid": baseline_valid,
                    "baseline_failed": baseline_failed,
                    "boundary_completion_fraction": boundary_valid
                    / max(boundary_valid + boundary_failed, 1),
                    "baseline_completion_fraction": baseline_valid
                    / max(baseline_valid + baseline_failed, 1),
                    "fisher_greater_failure_p": float(p_value),
                }
            )
    return completion, pd.DataFrame(comparisons)


def make_plot(
    results: pd.DataFrame,
    completion: pd.DataFrame,
    output_dir: Path,
) -> None:
    valid = results.copy()
    if "run_status" in valid.columns:
        valid = valid.loc[valid["run_status"] == "completed"]
    if "valid_dynamic_run" in valid.columns:
        valid = valid.loc[valid["valid_dynamic_run"].astype(bool)]
    if valid.empty and completion.empty:
        return
    grouped = (
        valid.groupby(["method", "chain_size"], as_index=False).agg(
            accuracy=("test_accuracy", "mean"),
            wall_time=("wall_time_seconds", "median"),
        )
        if not valid.empty
        else pd.DataFrame(columns=["method", "chain_size", "accuracy", "wall_time"])
    )
    completion_grouped = (
        completion.groupby(["method", "chain_size"], as_index=False)
        .agg(n_requested=("n_requested", "sum"), n_valid=("n_valid", "sum"))
    )
    completion_grouped["completion"] = (
        completion_grouped["n_valid"]
        / completion_grouped["n_requested"].clip(lower=1)
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    for method in METHODS:
        subset = grouped.loc[grouped["method"] == method]
        if not subset.empty:
            axes[0].plot(
                subset["chain_size"], subset["accuracy"], marker="o", label=method
            )
            axes[1].plot(
                subset["chain_size"], subset["wall_time"], marker="o", label=method
            )
        completion_subset = completion_grouped.loc[
            completion_grouped["method"] == method
        ]
        axes[2].plot(
            completion_subset["chain_size"],
            completion_subset["completion"],
            marker="o",
            label=method,
        )
    axes[0].set(
        xlabel="chain size",
        ylabel="valid-run conditional mean test accuracy",
        ylim=(0.45, 1.02),
    )
    axes[1].set(
        xlabel="chain size",
        ylabel="valid-run conditional median wall time [s]",
        yscale="log",
    )
    axes[2].set(
        xlabel="chain size",
        ylabel="run completion fraction",
        ylim=(-0.02, 1.02),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_overview.png", dpi=190)
    plt.close(fig)


def parse_csv_tuple(value: str, cast) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def build_specs(args: argparse.Namespace) -> tuple[list[RunSpec], dict[str, object]]:
    base = PROFILES[args.profile]
    if args.endpoint_step_multiplier < 1:
        raise ValueError("endpoint_step_multiplier must be at least 1")
    if args.free_tolerance_multiplier < 1.0:
        raise ValueError("free_tolerance_multiplier must be at least 1")
    if args.dynamics_dt <= 0.0:
        raise ValueError("dynamics_dt must be positive")
    if args.damping_trace <= 0.0 or args.damping_scale <= 0.0:
        raise ValueError("damping_trace and damping_scale must be positive")
    if args.learning_rate_u <= 0.0 or args.learning_rate_structure <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.n_input_nodes < 1:
        raise ValueError("n_input_nodes must be positive")
    if args.rbf_sigma <= 0.0:
        raise ValueError("rbf_sigma must be positive")
    datasets = parse_csv_tuple(args.datasets, str) if args.datasets else base.datasets
    sizes = parse_csv_tuple(args.chain_sizes, int) if args.chain_sizes else base.chain_sizes
    seeds = parse_csv_tuple(args.seeds, int) if args.seeds else base.seeds
    methods = parse_csv_tuple(args.methods, str) if args.methods else base.methods
    unknown_datasets = set(datasets) - set(DATASETS)
    unknown_methods = set(methods) - set(METHODS)
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {sorted(unknown_datasets)}")
    if unknown_methods:
        raise ValueError(f"unknown methods: {sorted(unknown_methods)}")
    settings = {
        "profile": args.profile,
        "datasets": datasets,
        "chain_sizes": sizes,
        "seeds": seeds,
        "methods": methods,
        "n_train": args.n_train or base.n_train,
        "n_test": args.n_test or base.n_test,
        "n_rbf": args.n_rbf or base.n_rbf,
        "n_input_nodes": args.n_input_nodes,
        "rbf_sigma": args.rbf_sigma,
        "epochs": args.epochs if args.epochs is not None else base.epochs,
        "max_steps": args.max_steps or base.max_steps,
        "tolerance": args.tolerance or base.tolerance,
        "beta": args.beta,
        "damping_schedule": args.damping_schedule,
        "damping_trace": args.damping_trace,
        "damping_scale": args.damping_scale,
        "endpoint_step_multiplier": args.endpoint_step_multiplier,
        "dynamics_dt": args.dynamics_dt,
        "free_tolerance_multiplier": args.free_tolerance_multiplier,
        "learning_rate_u": args.learning_rate_u,
        "learning_rate_structure": args.learning_rate_structure,
        "chain_regime": args.chain_regime,
        "phase_protocol": args.phase_protocol,
    }
    specs = [
        RunSpec(
            dataset=dataset,
            chain_size=size,
            seed=seed,
            method=method,
            n_train=settings["n_train"],
            n_test=settings["n_test"],
            n_rbf=settings["n_rbf"],
            n_input_nodes=settings["n_input_nodes"],
            rbf_sigma=settings["rbf_sigma"],
            epochs=settings["epochs"],
            max_steps=settings["max_steps"],
            tolerance=settings["tolerance"],
            beta=settings["beta"],
            damping_schedule=settings["damping_schedule"],
            damping_trace=settings["damping_trace"],
            damping_scale=settings["damping_scale"],
            endpoint_step_multiplier=settings["endpoint_step_multiplier"],
            dynamics_dt=settings["dynamics_dt"],
            free_tolerance_multiplier=settings["free_tolerance_multiplier"],
            learning_rate_u=settings["learning_rate_u"],
            learning_rate_structure=settings["learning_rate_structure"],
            chain_regime=settings["chain_regime"],
            phase_protocol=settings["phase_protocol"],
        )
        for dataset in datasets
        for size in sizes
        for seed in seeds
        for method in methods
    ]
    return specs, settings


def run_suite(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, settings = build_specs(args)
    results_path = output_dir / "benchmark_runs.csv"
    history_path = output_dir / "benchmark_history.csv"
    failures_path = output_dir / "benchmark_failures.csv"
    endpoint_diagnostics_path = output_dir / "endpoint_diagnostics.csv"
    progress_path = output_dir / "benchmark_progress.json"
    def read_nonempty_csv(path: Path) -> pd.DataFrame:
        if args.resume and path.exists() and path.stat().st_size > 0:
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
        return pd.DataFrame()

    existing_results = read_nonempty_csv(results_path)
    if not existing_results.empty:
        if "damping_schedule" not in existing_results.columns:
            raise ValueError(
                "The output directory contains legacy results. Use a new output directory "
                "for the corrected damping benchmark."
            )
        existing_schedules = set(existing_results["damping_schedule"].astype(str))
        if existing_schedules != {args.damping_schedule}:
            raise ValueError(
                f"Resume mismatch: existing damping schedules are {sorted(existing_schedules)}, "
                f"requested {args.damping_schedule}."
            )
        if "chain_regime" not in existing_results.columns or set(
            existing_results["chain_regime"].astype(str)
        ) != {settings["chain_regime"]}:
            raise ValueError(
                "Resume mismatch for chain_regime; use the original setting "
                "or a new output directory."
            )
        if "phase_protocol" not in existing_results.columns or set(
            existing_results["phase_protocol"].astype(str)
        ) != {settings["phase_protocol"]}:
            raise ValueError(
                "Resume mismatch for phase_protocol; use the original setting "
                "or a new output directory."
            )
        for column, requested in (
            ("damping_trace", settings["damping_trace"]),
            ("damping_scale", settings["damping_scale"]),
            ("endpoint_step_multiplier", settings["endpoint_step_multiplier"]),
            ("dynamics_dt", settings["dynamics_dt"]),
            ("free_tolerance_multiplier", settings["free_tolerance_multiplier"]),
            ("beta", settings["beta"]),
            ("n_train", settings["n_train"]),
            ("n_test", settings["n_test"]),
            ("n_rbf", settings["n_rbf"]),
            ("epochs", settings["epochs"]),
            ("max_steps", settings["max_steps"]),
            ("tolerance", settings["tolerance"]),
            ("learning_rate_u", settings["learning_rate_u"]),
            ("learning_rate_structure", settings["learning_rate_structure"]),
        ):
            if column not in existing_results.columns or not np.allclose(
                existing_results[column].astype(float), requested
            ):
                raise ValueError(
                    f"Resume mismatch for {column}; use the original settings or a new output directory."
                )
    if not existing_results.empty and "run_status" not in existing_results.columns:
        existing_valid = (
            existing_results["valid_dynamic_run"].astype(bool)
            if "valid_dynamic_run" in existing_results.columns
            else pd.Series(True, index=existing_results.index)
        )
        existing_results["run_status"] = np.where(
            existing_valid,
            "completed",
            "failed_endpoint",
        )
    valid_existing = existing_results.copy()
    if not valid_existing.empty:
        valid_existing = valid_existing.loc[
            (valid_existing["run_status"] == "completed")
            & valid_existing["valid_dynamic_run"].astype(bool)
        ]
    completed = set(valid_existing.get("run_key", pd.Series(dtype=str)).astype(str))
    pending = [spec for spec in specs if spec.key not in completed]
    all_results = valid_existing.to_dict(orient="records")
    all_history = read_nonempty_csv(history_path).to_dict(orient="records")
    failure_records = read_nonempty_csv(failures_path).to_dict(orient="records")
    if not existing_results.empty:
        invalid_existing = existing_results.loc[
            ~existing_results["run_key"].astype(str).isin(completed)
        ]
        failure_records.extend(invalid_existing.to_dict(orient="records"))
    all_diagnostics = read_nonempty_csv(endpoint_diagnostics_path).to_dict(
        orient="records"
    )
    # A pending key is being retried.  Remove its stale partial history and
    # endpoint diagnostics so resume does not duplicate failed attempts.
    pending_keys = {spec.key for spec in pending}
    all_history = [
        row for row in all_history if str(row.get("run_key", "")) not in pending_keys
    ]
    all_diagnostics = [
        row
        for row in all_diagnostics
        if str(row.get("run_key", "")) not in pending_keys
    ]
    print(f"profile={args.profile}; total={len(specs)}; pending={len(pending)}; workers={args.workers}")

    def save_checkpoint() -> None:
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        pd.DataFrame(all_history).to_csv(history_path, index=False)
        pd.DataFrame(failure_records).to_csv(failures_path, index=False)
        pd.DataFrame(all_diagnostics).to_csv(endpoint_diagnostics_path, index=False)
        progress = {
            "n_requested_runs": len(specs),
            "n_valid_runs": len(all_results),
            "n_failed_runs": len(failure_records),
            "n_pending_runs": max(
                len(specs) - len(all_results) - len(failure_records), 0
            ),
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    def register_run(
        result: dict[str, object],
        history: list[dict[str, object]],
        diagnostics: list[dict[str, object]],
    ) -> None:
        key = str(result["run_key"])
        all_diagnostics.extend(diagnostics)
        # Keep only the latest outcome for a retried run key.
        failure_records[:] = [
            row for row in failure_records if str(row.get("run_key", "")) != key
        ]
        if result.get("run_status") == "completed" and bool(
            result.get("valid_dynamic_run", False)
        ):
            all_results[:] = [
                row for row in all_results if str(row.get("run_key", "")) != key
            ]
            all_results.append(result)
            all_history.extend(history)
        else:
            failure_records.append(result)

    if args.workers == 1:
        for index, spec in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {spec.key}", flush=True)
            result, history, diagnostics = run_one(spec)
            register_run(result, history, diagnostics)
            if result["run_status"] != "completed":
                print(
                    f"FAILED endpoint: epoch={result['failure_epoch']} "
                    f"phase={result['failure_phase']}",
                    flush=True,
                )
            save_checkpoint()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_one, spec): spec for spec in pending}
            for index, future in enumerate(as_completed(future_map), start=1):
                spec = future_map[future]
                result, history, diagnostics = future.result()
                register_run(result, history, diagnostics)
                label = "completed" if result["run_status"] == "completed" else "FAILED"
                print(f"[{index}/{len(pending)}] {label} {spec.key}", flush=True)
                save_checkpoint()

    results = pd.DataFrame(all_results)
    if not results.empty:
        results = results.sort_values(["dataset", "chain_size", "seed", "method"])
    results.to_csv(results_path, index=False)
    pd.DataFrame(all_history).to_csv(history_path, index=False)
    failures = pd.DataFrame(failure_records)
    if not failures.empty and "run_key" in failures.columns:
        failures = failures.drop_duplicates("run_key", keep="last")
    failures.to_csv(failures_path, index=False)
    pd.DataFrame(all_diagnostics).to_csv(endpoint_diagnostics_path, index=False)
    summary, paired = aggregate_results(results)
    completion, completion_comparisons = completion_tables(specs, results, failures)
    if summary.empty:
        summary = completion.copy()
        summary["n_runs"] = 0
    else:
        summary = completion.merge(
            summary,
            on=["method", "dataset", "chain_size"],
            how="left",
        )
    if "n_runs" in summary.columns:
        summary["n_runs"] = summary["n_runs"].fillna(0).astype(int)
    if not paired.empty:
        requested_by_method = {
            method: len(
                {
                    (spec.dataset, spec.chain_size, spec.seed)
                    for spec in specs
                    if spec.method == method
                }
            )
            for method in METHODS
        }
        paired["n_requested_pairs"] = paired["comparison"].map(
            lambda value: min(
                requested_by_method.get("boundary_dynamic", 0),
                requested_by_method.get(str(value).split(" - ", 1)[-1], 0),
            )
        )
        paired["paired_coverage_fraction"] = (
            paired["n_pairs"] / paired["n_requested_pairs"].clip(lower=1)
        )
        paired["conditional_on_valid_endpoints"] = True
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    completion.to_csv(output_dir / "benchmark_completion.csv", index=False)
    completion_comparisons.to_csv(
        output_dir / "completion_comparisons.csv", index=False
    )
    make_plot(results, completion, output_dir)
    settings["n_requested_runs"] = len(specs)
    settings["n_completed_runs"] = len(results)
    settings["n_failed_runs"] = len(failures)
    settings["publication_tables_exclude_failed_runs"] = True
    settings["accuracy_and_runtime_summaries_are_conditional_on_valid_endpoints"] = True
    settings["completion_statistics_include_every_requested_run"] = True
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"Saved benchmark outputs to {output_dir}")
    if not failures.empty and not args.allow_failed_runs:
        raise SystemExit(
            f"{len(failures)} run(s) had an invalid endpoint. They were "
            "excluded from publication tables. Inspect benchmark_failures.csv "
            "and endpoint_diagnostics.csv, then resume after correcting the solver."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("results_benchmark"))
    parser.add_argument("--datasets", help="comma-separated override")
    parser.add_argument("--chain-sizes", help="comma-separated override")
    parser.add_argument("--seeds", help="comma-separated override")
    parser.add_argument("--methods", help="comma-separated override")
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--n-rbf", type=int)
    parser.add_argument(
        "--n-input-nodes",
        type=int,
        default=2,
        help="number of leading chain nodes that receive the feature input",
    )
    parser.add_argument(
        "--rbf-sigma",
        type=float,
        default=0.72,
        help="width of the radial-basis features",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--beta", type=float, default=0.035)
    parser.add_argument("--damping-schedule", choices=DAMPING_SCHEDULES, default="fixed_trace")
    parser.add_argument("--damping-trace", type=float, default=1.00)
    parser.add_argument("--damping-scale", type=float, default=1.0)
    parser.add_argument("--endpoint-step-multiplier", type=int, default=10)
    parser.add_argument("--dynamics-dt", type=float, default=0.20)
    parser.add_argument(
        "--free-tolerance-multiplier",
        type=float,
        default=100.0,
        help=(
            "multiplier applied only to the free-phase stopping tolerance; "
            "the +beta and -beta endpoints retain --tolerance"
        ),
    )
    parser.add_argument("--learning-rate-u", type=float, default=0.0025)
    parser.add_argument("--learning-rate-structure", type=float, default=0.0003)
    parser.add_argument("--chain-regime", choices=CHAIN_REGIMES, default="propagating")
    parser.add_argument(
        "--phase-protocol",
        choices=PHASE_PROTOCOLS,
        default="centered_pair_cache",
        help=(
            "centered_pair_cache relaxes the free endpoint only at "
            "initialization and then warm-starts q+ and q- from their own "
            "previous strict equilibria; exact_free_boundary_pair computes "
            "the current free equilibrium with Newton and uses the selected "
            "dynamics only for the gradient-forming +beta and -beta phases"
        ),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-failed-runs",
        action="store_true",
        help="return success even if invalid runs were recorded and excluded",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())
