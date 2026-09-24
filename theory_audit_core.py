#!/usr/bin/env python3
"""Shared numerical tools for the energy, gradient, and relaxation tests.

The tests use the strictly convex chain energy from the learning benchmarks,
with separate configurations for evaluating the assumptions and predictions
of boundary damping, centered gradients, and finite-relaxation error.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

from advanced_rep_experiments import DynamicsConfig, damping_vector, relax_dynamics
from benchmark_suite import chain_configuration, make_dataset
from boundary_rep_learning import (
    ChainConfig,
    ChainParameters,
    energy,
    flatten_gradients,
    implicit_gradients,
    initialize_parameters,
    make_features,
    parameter_energy_gradients,
    solve_equilibrium,
    state_gradient,
    state_hessian,
    symmetric_eqprop_gradients,
)


def make_audit_problem(
    chain_size: int,
    seed: int,
    *,
    dataset: str = "moons",
    n_samples: int = 16,
    n_rbf: int = 8,
    beta: float = 0.035,
    boundary_variant: str = "paper_boundary",
) -> tuple[ChainConfig, ChainParameters, np.ndarray, np.ndarray]:
    """Create a deterministic chain instance for theory validation."""
    rng = np.random.default_rng(seed)
    x, targets = make_dataset(dataset, n_samples, rng)
    x = (x - np.mean(x, axis=0)) / np.maximum(np.std(x, axis=0), 1e-12)
    centers = x[rng.choice(n_samples, min(n_rbf, n_samples), replace=False)]
    features = make_features(x, centers, 0.72)
    config = chain_configuration(
        chain_size,
        seed,
        beta,
        damping_schedule="fixed_trace",
        damping_trace=0.7,
        chain_regime="propagating",
    )
    if boundary_variant == "single_terminal":
        config = replace(
            config,
            boundary_width=1,
            boundary_damping=0.7,
            boundary_profile="fixed_trace",
            boundary_trace=0.7,
        )
    elif boundary_variant != "paper_boundary":
        raise ValueError(f"unknown boundary variant: {boundary_variant}")
    params = initialize_parameters(config, features.shape[1])
    return config, params, features, targets


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - reference)
        / max(float(np.linalg.norm(reference)), 1e-14)
    )


def cosine_similarity(estimate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(estimate) * np.linalg.norm(reference))
    if denominator <= 1e-14:
        return 1.0 if np.linalg.norm(estimate - reference) <= 1e-12 else 0.0
    return float(np.dot(estimate, reference) / denominator)


def selector_matrix(damping: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(damping) > 0.0)
    selector = np.zeros((indices.size, damping.size))
    selector[np.arange(indices.size), indices] = 1.0
    return selector


def modal_observability(
    hessian: np.ndarray,
    damping: np.ndarray,
    *,
    degeneracy_rtol: float = 1e-8,
) -> dict[str, object]:
    """Evaluate the undamped-mode criterion K v = lambda M v, B v = 0.

    Unit mass is used by the chain integrator.  Degenerate eigenspaces are
    checked as subspaces because individual eigenvectors inside such a space
    depend on the basis returned by the eigensolver.
    """
    n = hessian.shape[0]
    mass = np.eye(n)
    eigenvalues, eigenvectors = eigh(hessian, mass)
    selector = selector_matrix(damping)
    individual_visibility = np.linalg.norm(selector @ eigenvectors, axis=0)
    damping_visibility = np.sqrt(
        np.maximum(np.sum(damping[:, None] * eigenvectors**2, axis=0), 0.0)
    )

    eigenspace_margins: list[float] = []
    group_sizes: list[int] = []
    start = 0
    while start < n:
        stop = start + 1
        scale = max(1.0, abs(float(eigenvalues[start])))
        while stop < n and abs(float(eigenvalues[stop] - eigenvalues[start])) <= degeneracy_rtol * scale:
            stop += 1
        block = selector @ eigenvectors[:, start:stop]
        if block.shape[0] < block.shape[1]:
            margin = 0.0
        else:
            singular_values = np.linalg.svd(block, compute_uv=False)
            margin = float(np.min(singular_values)) if singular_values.size else 0.0
        eigenspace_margins.append(margin)
        group_sizes.append(stop - start)
        start = stop

    system = first_order_system_matrix(hessian, damping)
    system_eigenvalues = np.linalg.eigvals(system)
    spectral_abscissa = float(np.max(system_eigenvalues.real))
    decay_rate = max(0.0, -spectral_abscissa)
    minimum_margin = float(min(eigenspace_margins))
    observability_tolerance = float(
        100.0 * np.finfo(float).eps * max(1.0, np.sqrt(n))
    )
    observable = bool(minimum_margin > observability_tolerance)
    positive_stiffness = bool(float(np.min(eigenvalues)) > 0.0)
    return {
        "stiffness_eigenvalues": eigenvalues,
        "frequencies": np.sqrt(np.maximum(eigenvalues, 0.0)),
        "individual_visibility": individual_visibility,
        "damping_visibility": damping_visibility,
        "eigenspace_visibility_margins": np.asarray(eigenspace_margins),
        "eigenspace_group_sizes": np.asarray(group_sizes),
        "minimum_visibility": float(np.min(individual_visibility)),
        "minimum_damping_visibility": float(np.min(damping_visibility)),
        "minimum_eigenspace_visibility": minimum_margin,
        "observability_tolerance": observability_tolerance,
        "observable": observable,
        "spectral_abscissa": spectral_abscissa,
        "spectral_decay_rate": decay_rate,
        # In exact arithmetic, positive stiffness plus modal observability is
        # equivalent to exponential stability.  Direct state-matrix eigenvalues
        # become unresolved when the true rate approaches machine precision,
        # so the theorem criterion is the primary Boolean result.
        "linearly_exponentially_stable": bool(
            positive_stiffness and observable
        ),
        "spectral_rate_numerically_resolved": bool(
            spectral_abscissa
            < -100.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(system, ord=2))
        ),
        "n_damped": int(np.count_nonzero(damping)),
        "damping_trace": float(np.sum(damping)),
    }


def first_order_system_matrix(hessian: np.ndarray, damping: np.ndarray) -> np.ndarray:
    n = hessian.shape[0]
    zero = np.zeros((n, n))
    identity = np.eye(n)
    return np.block([[zero, identity], [-hessian, -np.diag(damping)]])


def hessian_summary(
    q: np.ndarray,
    params: ChainParameters,
    config: ChainConfig,
    beta: float,
) -> dict[str, float]:
    hessians = state_hessian(q, params, config, beta)
    eigenvalues = np.linalg.eigvalsh(hessians)
    minimum = float(np.min(eigenvalues))
    maximum = float(np.max(eigenvalues))
    return {
        "minimum_hessian_eigenvalue": minimum,
        "maximum_hessian_eigenvalue": maximum,
        "maximum_hessian_condition_number": float(
            np.max(eigenvalues[:, -1] / eigenvalues[:, 0])
        ),
        "strongly_convex_on_audited_states": bool(minimum > 0.0),
    }


def integrate_energy_trajectory(
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    *,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    damping_mode: str = "boundary",
    t_end: float = 80.0,
    n_times: int = 401,
) -> dict[str, object]:
    """High-accuracy single-sample trajectory with an integrated flux channel."""
    if features.shape[0] != 1:
        raise ValueError("integrate_energy_trajectory expects one sample")
    if targets is not None and targets.shape != (1,):
        raise ValueError("targets must contain one sample")
    n = config.n_nodes
    damping = damping_vector(config, damping_mode, params)

    def rhs(_: float, state: np.ndarray) -> np.ndarray:
        q = state[:n][None, :]
        velocity = state[n : 2 * n][None, :]
        gradient = state_gradient(q, params, features, config, targets, beta)
        acceleration = -gradient - damping[None, :] * velocity
        dissipative_power = float(np.sum(damping * velocity[0] ** 2))
        return np.concatenate((velocity[0], acceleration[0], [dissipative_power]))

    time = np.linspace(0.0, t_end, n_times)
    solution = solve_ivp(
        rhs,
        (0.0, t_end),
        np.zeros(2 * n + 1),
        t_eval=time,
        method="DOP853",
        rtol=2e-10,
        atol=2e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    q = solution.y[:n].T
    velocity = solution.y[n : 2 * n].T
    dissipated = solution.y[-1]
    repeated_features = np.repeat(features, n_times, axis=0)
    repeated_targets = None if targets is None else np.repeat(targets, n_times)
    potential = energy(
        q,
        params,
        repeated_features,
        config,
        repeated_targets,
        beta,
    )
    kinetic = 0.5 * np.sum(velocity**2, axis=1)
    total = potential + kinetic
    balance = total + dissipated - total[0]
    equilibrium, _, equilibrium_residual = solve_equilibrium(
        params,
        features,
        config,
        targets,
        beta,
    )
    equilibrium_energy = float(energy(equilibrium, params, features, config, targets, beta)[0])
    energy_gap = total - equilibrium_energy
    initial_scale = max(abs(float(energy_gap[0])), abs(float(dissipated[-1])), 1e-14)
    monotonic_tolerance = 2e-8 * initial_scale
    return {
        "time": time,
        "q": q,
        "velocity": velocity,
        "potential": potential,
        "kinetic": kinetic,
        "total_energy": total,
        "dissipated_energy": dissipated,
        "balance_residual": balance,
        "energy_gap": energy_gap,
        "distance_to_equilibrium": np.linalg.norm(q - equilibrium[0], axis=1),
        "relative_balance_error": float(np.max(np.abs(balance)) / initial_scale),
        "monotonicity_violations": int(np.count_nonzero(np.diff(total) > monotonic_tolerance)),
        "final_energy_gap_fraction": float(energy_gap[-1] / max(energy_gap[0], 1e-14)),
        "final_distance_to_equilibrium": float(np.linalg.norm(q[-1] - equilibrium[0])),
        "final_velocity_norm": float(np.linalg.norm(velocity[-1])),
        "equilibrium_residual": float(equilibrium_residual),
        "damping": damping,
    }


def exact_gradient_triplet(
    params: ChainParameters,
    features: np.ndarray,
    targets: np.ndarray,
    config: ChainConfig,
    beta: float,
) -> dict[str, object]:
    q_free, _, free_residual = solve_equilibrium(params, features, config)
    q_plus, _, plus_residual = solve_equilibrium(
        params, features, config, targets, beta, initial=q_free
    )
    q_minus, _, minus_residual = solve_equilibrium(
        params, features, config, targets, -beta, initial=q_free
    )
    implicit = flatten_gradients(
        implicit_gradients(q_free, params, features, targets, config)
    )
    centered = flatten_gradients(
        symmetric_eqprop_gradients(
            q_minus, q_plus, params, features, config, beta
        )
    )
    return {
        "q_free": q_free,
        "q_plus": q_plus,
        "q_minus": q_minus,
        "implicit_gradient": implicit,
        "centered_gradient": centered,
        "relative_error": relative_error(centered, implicit),
        "cosine": cosine_similarity(centered, implicit),
        "free_residual": float(free_residual),
        "plus_residual": float(plus_residual),
        "minus_residual": float(minus_residual),
    }


def dynamic_centered_gradient(
    params: ChainParameters,
    features: np.ndarray,
    targets: np.ndarray,
    config: ChainConfig,
    beta: float,
    *,
    tolerance: float,
    max_steps: int,
    dt: float = 0.2,
    damping_mode: str = "boundary",
    force_full_steps: bool = False,
) -> dict[str, object]:
    """Generate the two centered endpoints by second-order dynamics."""
    exact = exact_gradient_triplet(params, features, targets, config, beta)
    q_free = np.asarray(exact["q_free"])
    dynamics = DynamicsConfig(
        dt=dt,
        max_steps=max_steps,
        check_every=max_steps + 1 if force_full_steps else 25,
        gradient_tolerance=0.0 if force_full_steps else tolerance,
        velocity_tolerance=0.0 if force_full_steps else tolerance,
        consecutive_checks=3,
        freeze_converged_samples=not force_full_steps,
    )
    plus = relax_dynamics(
        params,
        features,
        config,
        dynamics,
        targets=targets,
        beta=beta,
        damping_mode=damping_mode,
        initial_q=q_free,
    )
    minus = relax_dynamics(
        params,
        features,
        config,
        dynamics,
        targets=targets,
        beta=-beta,
        damping_mode=damping_mode,
        initial_q=q_free,
    )
    q_plus = np.asarray(plus["q"])
    q_minus = np.asarray(minus["q"])
    dynamic_gradient = flatten_gradients(
        symmetric_eqprop_gradients(
            q_minus, q_plus, params, features, config, beta
        )
    )
    exact_centered = np.asarray(exact["centered_gradient"])
    implicit = np.asarray(exact["implicit_gradient"])
    plus_state_error = float(np.max(np.linalg.norm(q_plus - np.asarray(exact["q_plus"]), axis=1)))
    minus_state_error = float(np.max(np.linalg.norm(q_minus - np.asarray(exact["q_minus"]), axis=1)))
    return {
        "q_plus_dynamic": q_plus,
        "q_minus_dynamic": q_minus,
        "q_plus_exact": np.asarray(exact["q_plus"]),
        "q_minus_exact": np.asarray(exact["q_minus"]),
        "dynamic_gradient": dynamic_gradient,
        "exact_centered_gradient": exact_centered,
        "implicit_gradient": implicit,
        "dynamic_vs_implicit_error": relative_error(dynamic_gradient, implicit),
        "dynamic_vs_centered_error": relative_error(dynamic_gradient, exact_centered),
        "exact_centered_vs_implicit_error": relative_error(exact_centered, implicit),
        "dynamic_vs_implicit_cosine": cosine_similarity(dynamic_gradient, implicit),
        "plus_state_error": plus_state_error,
        "minus_state_error": minus_state_error,
        "plus_residual": float(plus["residual"]),
        "minus_residual": float(minus["residual"]),
        "plus_velocity_norm": float(plus["velocity_norm"]),
        "minus_velocity_norm": float(minus["velocity_norm"]),
        "plus_steps": int(plus["steps"]),
        "minus_steps": int(minus["steps"]),
        "plus_converged": bool(plus["converged"]),
        "minus_converged": bool(minus["converged"]),
    }


def parameter_map_jacobian_norm(
    params: ChainParameters,
    features: np.ndarray,
    q: np.ndarray,
) -> float:
    """Exact Jacobian norm of q -> mean partial_theta H(q) at one state."""
    from boundary_rep_learning import sigmoid

    batch, n = q.shape
    n_input = params.u.shape[0]
    n_features = features.shape[1]
    n_parameters = n + (n - 1) + n_input * n_features
    jacobian = np.zeros((n_parameters, batch * n))
    scale = 1.0 / batch
    row = 0
    onsite_scale = sigmoid(params.log_a)
    for node in range(n):
        for sample in range(batch):
            jacobian[row, sample * n + node] = (
                scale * onsite_scale[node] * q[sample, node]
            )
        row += 1
    edge_scale = sigmoid(params.log_w)
    differences = q[:, 1:] - q[:, :-1]
    for edge in range(n - 1):
        for sample in range(batch):
            value = scale * edge_scale[edge] * differences[sample, edge]
            jacobian[row, sample * n + edge] = -value
            jacobian[row, sample * n + edge + 1] = value
        row += 1
    for node in range(n_input):
        for feature in range(n_features):
            for sample in range(batch):
                jacobian[row, sample * n + node] = -scale * features[sample, feature]
            row += 1
    return float(np.linalg.norm(jacobian, ord=2))


def parameter_map_segment_lipschitz_bound(
    params: ChainParameters,
    features: np.ndarray,
    q_first: np.ndarray,
    q_second: np.ndarray,
) -> float:
    """Lipschitz bound along a segment; the Jacobian is affine in q."""
    return max(
        parameter_map_jacobian_norm(params, features, q_first),
        parameter_map_jacobian_norm(params, features, q_second),
    )


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan"), float("nan")
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    prediction = slope * log_x + intercept
    denominator = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - float(np.sum((log_y - prediction) ** 2)) / max(denominator, 1e-30)
    return float(slope), float(r_squared)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
