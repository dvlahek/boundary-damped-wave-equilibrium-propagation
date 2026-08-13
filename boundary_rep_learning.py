#!/usr/bin/env python3
"""Boundary-radiative equilibrium propagation on a nonlinear physical chain.

This experiment joins three pieces:

* conservative propagation in the interior of a chain;
* dissipation restricted to an output boundary;
* equilibrium-propagation gradients for a supervised learning task.

The benchmark uses a strictly convex nonlinear spring energy.  The unique free
and nudged equilibria are solved accurately by Newton iteration for training.
Separately, second-order boundary-damped dynamics is integrated to verify that
it reaches the same equilibrium while obeying a boundary-flux energy law.
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
from scipy.integrate import solve_ivp


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


@dataclass(frozen=True)
class ChainConfig:
    n_nodes: int = 7
    n_input_nodes: int = 2
    alpha: float = 0.12
    stiffness_floor: float = 0.20
    edge_floor: float = 0.45
    boundary_width: int = 3
    boundary_damping: float = 0.60
    boundary_profile: str = "flat"
    boundary_trace: float = 1.0
    damping_scale: float = 1.0
    initial_onsite_stiffness: float = 0.85
    initial_edge_stiffness: float = 0.75
    beta: float = 0.01
    seed: int = 2026


@dataclass(frozen=True)
class TrainingConfig:
    n_train: int = 280
    n_test: int = 160
    n_rbf: int = 10
    rbf_sigma: float = 0.72
    noise: float = 0.10
    epochs: int = 140
    batch_size: int = 40
    learning_rate_u: float = 0.025
    learning_rate_structure: float = 0.006
    weight_decay_u: float = 3e-4
    seed: int = 2027


@dataclass
class ChainParameters:
    log_a: np.ndarray
    log_w: np.ndarray
    u: np.ndarray

    def copy(self) -> "ChainParameters":
        return ChainParameters(self.log_a.copy(), self.log_w.copy(), self.u.copy())


class Adam:
    def __init__(self, shape: tuple[int, ...], learning_rate: float) -> None:
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.learning_rate = learning_rate
        self.step_number = 0

    def update(self, value: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.step_number += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * gradient**2
        m_hat = self.m / (1.0 - 0.9**self.step_number)
        v_hat = self.v / (1.0 - 0.999**self.step_number)
        return value - self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def make_two_moons(n_samples: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_first = n_samples // 2
    n_second = n_samples - n_first
    angle_first = rng.uniform(0.0, np.pi, n_first)
    angle_second = rng.uniform(0.0, np.pi, n_second)
    first = np.column_stack((np.cos(angle_first), np.sin(angle_first)))
    second = np.column_stack((1.0 - np.cos(angle_second), 0.45 - np.sin(angle_second)))
    x = np.vstack((first, second)) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.concatenate((-np.ones(n_first), np.ones(n_second)))
    order = rng.permutation(n_samples)
    return x[order], y[order]


def make_features(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    squared_distance = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    rbf = np.exp(-squared_distance / (2.0 * sigma**2))
    return np.column_stack((np.ones(x.shape[0]), x, rbf))


def initialize_parameters(config: ChainConfig, n_features: int) -> ChainParameters:
    rng = np.random.default_rng(config.seed)
    if config.initial_onsite_stiffness <= config.stiffness_floor:
        raise ValueError("initial onsite stiffness must exceed stiffness_floor")
    if config.initial_edge_stiffness <= config.edge_floor:
        raise ValueError("initial edge stiffness must exceed edge_floor")
    log_a = np.full(
        config.n_nodes,
        inverse_softplus(config.initial_onsite_stiffness - config.stiffness_floor),
    )
    log_a += rng.normal(scale=0.08, size=config.n_nodes)
    log_w = np.full(
        config.n_nodes - 1,
        inverse_softplus(config.initial_edge_stiffness - config.edge_floor),
    )
    log_w += rng.normal(scale=0.08, size=config.n_nodes - 1)
    u = rng.normal(scale=0.22, size=(config.n_input_nodes, n_features))
    return ChainParameters(log_a, log_w, u)


def physical_coefficients(params: ChainParameters, config: ChainConfig) -> tuple[np.ndarray, np.ndarray]:
    a = softplus(params.log_a) + config.stiffness_floor
    w = softplus(params.log_w) + config.edge_floor
    return a, w


def input_force(params: ChainParameters, features: np.ndarray, config: ChainConfig) -> np.ndarray:
    force = np.zeros((features.shape[0], config.n_nodes))
    force[:, : config.n_input_nodes] = features @ params.u.T
    return force


def state_gradient(
    q: np.ndarray,
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
) -> np.ndarray:
    a, w = physical_coefficients(params, config)
    gradient = a[None, :] * q + config.alpha * q**3 - input_force(params, features, config)
    difference = q[:, 1:] - q[:, :-1]
    gradient[:, :-1] -= w[None, :] * difference
    gradient[:, 1:] += w[None, :] * difference
    if beta != 0.0:
        if targets is None:
            raise ValueError("targets are required for a nudged equilibrium")
        gradient[:, -1] += beta * (q[:, -1] - targets)
    return gradient


def state_hessian(
    q: np.ndarray,
    params: ChainParameters,
    config: ChainConfig,
    beta: float = 0.0,
) -> np.ndarray:
    a, w = physical_coefficients(params, config)
    batch = q.shape[0]
    hessian = np.zeros((batch, config.n_nodes, config.n_nodes))
    diagonal = a[None, :] + 3.0 * config.alpha * q**2
    diagonal[:, :-1] += w[None, :]
    diagonal[:, 1:] += w[None, :]
    diagonal[:, -1] += beta
    indices = np.arange(config.n_nodes)
    hessian[:, indices, indices] = diagonal
    edge_indices = np.arange(config.n_nodes - 1)
    hessian[:, edge_indices, edge_indices + 1] = -w[None, :]
    hessian[:, edge_indices + 1, edge_indices] = -w[None, :]
    return hessian


def solve_equilibrium(
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial: np.ndarray | None = None,
    tolerance: float = 2e-11,
    max_iterations: int = 60,
) -> tuple[np.ndarray, int, float]:
    q = np.zeros((features.shape[0], config.n_nodes)) if initial is None else initial.copy()
    for iteration in range(1, max_iterations + 1):
        gradient = state_gradient(q, params, features, config, targets, beta)
        residual = float(np.max(np.linalg.norm(gradient, axis=1)))
        if residual <= tolerance:
            return q, iteration - 1, residual
        hessian = state_hessian(q, params, config, beta)
        step = np.linalg.solve(hessian, gradient[..., None])[..., 0]

        # The energy is strictly convex, but damping the first Newton steps
        # avoids rare overshoots from a large random input force.
        step_scale = 1.0 if iteration > 3 else 0.70
        q -= step_scale * step
    raise RuntimeError(f"Newton equilibrium solver did not converge; residual={residual:.3e}")


def energy(
    q: np.ndarray,
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
) -> np.ndarray:
    a, w = physical_coefficients(params, config)
    difference = q[:, 1:] - q[:, :-1]
    value = (
        0.5 * np.sum(a[None, :] * q**2, axis=1)
        + 0.5 * np.sum(w[None, :] * difference**2, axis=1)
        + 0.25 * config.alpha * np.sum(q**4, axis=1)
        - np.sum(q * input_force(params, features, config), axis=1)
    )
    if beta != 0.0:
        if targets is None:
            raise ValueError("targets are required for a nudged energy")
        value += 0.5 * beta * (q[:, -1] - targets) ** 2
    return value


def parameter_energy_gradients(
    q: np.ndarray,
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
) -> ChainParameters:
    difference = q[:, 1:] - q[:, :-1]
    grad_log_a = sigmoid(params.log_a) * np.mean(0.5 * q**2, axis=0)
    grad_log_w = sigmoid(params.log_w) * np.mean(0.5 * difference**2, axis=0)
    grad_u = -np.mean(q[:, : config.n_input_nodes, None] * features[:, None, :], axis=0)
    return ChainParameters(grad_log_a, grad_log_w, grad_u)


def eqprop_gradients(
    q_free: np.ndarray,
    q_nudged: np.ndarray,
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    beta: float,
) -> ChainParameters:
    free = parameter_energy_gradients(q_free, params, features, config)
    nudged = parameter_energy_gradients(q_nudged, params, features, config)
    return ChainParameters(
        (nudged.log_a - free.log_a) / beta,
        (nudged.log_w - free.log_w) / beta,
        (nudged.u - free.u) / beta,
    )


def symmetric_eqprop_gradients(
    q_minus: np.ndarray,
    q_plus: np.ndarray,
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    beta: float,
) -> ChainParameters:
    """Centered EqProp estimator with O(beta^2) truncation error."""
    minus = parameter_energy_gradients(q_minus, params, features, config)
    plus = parameter_energy_gradients(q_plus, params, features, config)
    return ChainParameters(
        (plus.log_a - minus.log_a) / (2.0 * beta),
        (plus.log_w - minus.log_w) / (2.0 * beta),
        (plus.u - minus.u) / (2.0 * beta),
    )


def implicit_gradients(
    q_free: np.ndarray,
    params: ChainParameters,
    features: np.ndarray,
    targets: np.ndarray,
    config: ChainConfig,
) -> ChainParameters:
    hessian = state_hessian(q_free, params, config)
    cost_gradient = np.zeros_like(q_free)
    cost_gradient[:, -1] = q_free[:, -1] - targets
    adjoint = np.linalg.solve(hessian, cost_gradient[..., None])[..., 0]
    difference_q = q_free[:, 1:] - q_free[:, :-1]
    difference_adjoint = adjoint[:, 1:] - adjoint[:, :-1]
    grad_log_a = -sigmoid(params.log_a) * np.mean(adjoint * q_free, axis=0)
    grad_log_w = -sigmoid(params.log_w) * np.mean(
        difference_adjoint * difference_q, axis=0
    )
    grad_u = np.mean(
        adjoint[:, : config.n_input_nodes, None] * features[:, None, :], axis=0
    )
    return ChainParameters(grad_log_a, grad_log_w, grad_u)


def flatten_gradients(gradients: ChainParameters) -> np.ndarray:
    return np.concatenate((gradients.log_a.ravel(), gradients.log_w.ravel(), gradients.u.ravel()))


def gradient_validation(
    params: ChainParameters,
    features: np.ndarray,
    targets: np.ndarray,
    config: ChainConfig,
) -> pd.DataFrame:
    q_free, _, _ = solve_equilibrium(params, features, config)
    implicit = flatten_gradients(implicit_gradients(q_free, params, features, targets, config))
    rows = []
    for beta in (0.12, 0.08, 0.05, 0.02, 0.01):
        q_plus, _, _ = solve_equilibrium(
            params, features, config, targets, beta, initial=q_free
        )
        q_minus, _, _ = solve_equilibrium(
            params, features, config, targets, -beta, initial=q_free
        )
        estimate = flatten_gradients(
            symmetric_eqprop_gradients(q_minus, q_plus, params, features, config, beta)
        )
        relative_error = np.linalg.norm(estimate - implicit) / np.linalg.norm(implicit)
        cosine = float(np.dot(estimate, implicit) / (np.linalg.norm(estimate) * np.linalg.norm(implicit)))
        rows.append(
            {
                "beta": beta,
                "relative_gradient_error": float(relative_error),
                "gradient_cosine_similarity": cosine,
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(q: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    output = q[:, -1]
    loss = float(np.mean(0.5 * (output - targets) ** 2))
    accuracy = float(np.mean(np.where(output >= 0.0, 1.0, -1.0) == targets))
    return loss, accuracy


def train_eqprop(
    params: ChainParameters,
    features_train: np.ndarray,
    y_train: np.ndarray,
    features_test: np.ndarray,
    y_test: np.ndarray,
    chain_config: ChainConfig,
    training_config: TrainingConfig,
) -> tuple[ChainParameters, pd.DataFrame]:
    rng = np.random.default_rng(training_config.seed + 91)
    optimizers = {
        "log_a": Adam(params.log_a.shape, training_config.learning_rate_structure),
        "log_w": Adam(params.log_w.shape, training_config.learning_rate_structure),
        "u": Adam(params.u.shape, training_config.learning_rate_u),
    }
    history = []
    for epoch in range(training_config.epochs + 1):
        if epoch > 0:
            order = rng.permutation(features_train.shape[0])
            for start in range(0, features_train.shape[0], training_config.batch_size):
                batch_indices = order[start : start + training_config.batch_size]
                features_batch = features_train[batch_indices]
                targets_batch = y_train[batch_indices]
                q_free, _, _ = solve_equilibrium(params, features_batch, chain_config)
                q_plus, _, _ = solve_equilibrium(
                    params,
                    features_batch,
                    chain_config,
                    targets_batch,
                    chain_config.beta,
                    initial=q_free,
                )
                q_minus, _, _ = solve_equilibrium(
                    params,
                    features_batch,
                    chain_config,
                    targets_batch,
                    -chain_config.beta,
                    initial=q_free,
                )
                gradients = symmetric_eqprop_gradients(
                    q_minus,
                    q_plus,
                    params,
                    features_batch,
                    chain_config,
                    chain_config.beta,
                )
                gradients.u += training_config.weight_decay_u * params.u
                params.log_a = optimizers["log_a"].update(params.log_a, gradients.log_a)
                params.log_w = optimizers["log_w"].update(params.log_w, gradients.log_w)
                params.u = optimizers["u"].update(params.u, gradients.u)

        if epoch % 5 == 0 or epoch == training_config.epochs:
            q_train, _, train_residual = solve_equilibrium(params, features_train, chain_config)
            q_test, _, test_residual = solve_equilibrium(params, features_test, chain_config)
            train_loss, train_accuracy = classification_metrics(q_train, y_train)
            test_loss, test_accuracy = classification_metrics(q_test, y_test)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "max_train_equilibrium_residual": train_residual,
                    "max_test_equilibrium_residual": test_residual,
                }
            )
    return params, pd.DataFrame(history)


def boundary_dynamics(
    params: ChainParameters,
    features: np.ndarray,
    config: ChainConfig,
    t_end: float = 2000.0,
    n_times: int = 4001,
) -> dict[str, np.ndarray | float | int]:
    if features.shape[0] != 1:
        raise ValueError("boundary_dynamics expects exactly one sample")
    n = config.n_nodes
    boundary_mask = np.zeros(n)
    boundary_mask[-config.boundary_width :] = 1.0
    gamma = config.boundary_damping

    def rhs(_: float, state: np.ndarray) -> np.ndarray:
        q = state[:n][None, :]
        velocity = state[n : 2 * n][None, :]
        acceleration = -state_gradient(q, params, features, config) - gamma * boundary_mask * velocity
        flux = gamma * np.sum((boundary_mask * velocity[0]) ** 2)
        return np.concatenate((velocity[0], acceleration[0], np.asarray([flux])))

    times = np.linspace(0.0, t_end, n_times)
    solution = solve_ivp(
        rhs,
        (0.0, t_end),
        np.zeros(2 * n + 1),
        t_eval=times,
        method="DOP853",
        rtol=2e-10,
        atol=2e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    q = solution.y[:n].T
    velocity = solution.y[n : 2 * n].T
    dissipated = solution.y[-1]
    potential = energy(q, params, np.repeat(features, n_times, axis=0), config)
    kinetic = 0.5 * np.sum(velocity**2, axis=1)
    total = potential + kinetic
    balance = total + dissipated - total[0]
    equilibrium, _, _ = solve_equilibrium(params, features, config)
    equilibrium_energy = float(energy(equilibrium, params, features, config)[0])
    energy_gap = total - equilibrium_energy
    distance = np.linalg.norm(q - equilibrium[0], axis=1)
    return {
        "time": times,
        "q": q,
        "velocity": velocity,
        "potential": potential,
        "kinetic": kinetic,
        "total_energy": total,
        "dissipated": dissipated,
        "balance": balance,
        "equilibrium": equilibrium[0],
        "equilibrium_energy": equilibrium_energy,
        "energy_gap": energy_gap,
        "distance_to_equilibrium": distance,
        "relative_balance_error": float(
            np.max(np.abs(balance))
            / max(float(np.max(total) - np.min(total)), abs(float(total[-1])), 1.0)
        ),
        "monotonicity_violations": int(
            np.count_nonzero(
                np.diff(total) > 2e-8 * max(float(np.max(total) - np.min(total)), 1.0)
            )
        ),
        "final_distance_to_equilibrium": float(distance[-1]),
        "final_velocity_norm": float(np.linalg.norm(velocity[-1])),
        "final_energy_gap": float(energy_gap[-1]),
    }


def make_plots(
    gradient_table: pd.DataFrame,
    history: pd.DataFrame,
    dynamics: dict[str, np.ndarray | float | int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    q_train: np.ndarray,
    q_test: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    axes[0].loglog(
        gradient_table["beta"],
        gradient_table["relative_gradient_error"],
        marker="o",
        lw=2.0,
    )
    axes[0].invert_xaxis()
    axes[0].set(xlabel=r"nudging strength $\beta$", ylabel="relative gradient error", title="EqProp gradient convergence")
    axes[0].grid(alpha=0.25, which="both")
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["test_accuracy"], label="test")
    axes[1].set(xlabel="epoch", ylabel="accuracy", ylim=(0.45, 1.02), title="Nonlinear chain classification")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    time = dynamics["time"]
    axes[2].semilogy(time, np.maximum(dynamics["energy_gap"], 1e-14), label="energy above equilibrium")
    axes[2].semilogy(time, np.maximum(dynamics["distance_to_equilibrium"], 1e-14), label="state error")
    axes[2].set(xlabel="time", ylabel="normalized/error quantity", title="Boundary-radiative convergence")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_rep_validation.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    propagation_mask = time <= 120.0
    propagation_time = time[propagation_mask]
    image = axes[0].imshow(
        dynamics["q"][propagation_mask].T,
        aspect="auto",
        origin="lower",
        extent=[float(propagation_time[0]), float(propagation_time[-1]), 0, dynamics["q"].shape[1] - 1],
        cmap="coolwarm",
    )
    axes[0].set(xlabel="time", ylabel="chain node", title="Signal propagation to the radiative boundary")
    fig.colorbar(image, ax=axes[0], label="state amplitude")

    prediction_train = np.where(q_train[:, -1] >= 0.0, 1.0, -1.0)
    prediction_test = np.where(q_test[:, -1] >= 0.0, 1.0, -1.0)
    correct_train = prediction_train == y_train
    correct_test = prediction_test == y_test
    axes[1].scatter(x_train[correct_train, 0], x_train[correct_train, 1], c=y_train[correct_train], cmap="coolwarm", s=16, alpha=0.45)
    axes[1].scatter(x_test[correct_test, 0], x_test[correct_test, 1], c=y_test[correct_test], cmap="coolwarm", s=28, edgecolor="k", linewidth=0.25)
    if np.any(~correct_test):
        axes[1].scatter(
            x_test[~correct_test, 0],
            x_test[~correct_test, 1],
            facecolors="none",
            edgecolors="gold",
            linewidth=1.4,
            s=70,
            label="misclassified test",
        )
        axes[1].legend(fontsize=8)
    axes[1].set(xlabel="$x_1$", ylabel="$x_2$", title="Learned equilibrium classifier")
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_propagation_and_classification.png", dpi=190)
    plt.close(fig)


def run(output_dir: Path, quick: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chain_config = ChainConfig()
    training_config = TrainingConfig(
        n_train=180 if quick else 280,
        n_test=100 if quick else 160,
        epochs=45 if quick else 140,
    )
    rng = np.random.default_rng(training_config.seed)
    x_all, y_all = make_two_moons(
        training_config.n_train + training_config.n_test,
        training_config.noise,
        rng,
    )
    mean = np.mean(x_all[: training_config.n_train], axis=0)
    scale = np.std(x_all[: training_config.n_train], axis=0)
    x_all_standard = (x_all - mean) / scale
    x_train = x_all_standard[: training_config.n_train]
    y_train = y_all[: training_config.n_train]
    x_test = x_all_standard[training_config.n_train :]
    y_test = y_all[training_config.n_train :]
    center_indices = rng.choice(training_config.n_train, training_config.n_rbf, replace=False)
    centers = x_train[center_indices]
    features_train = make_features(x_train, centers, training_config.rbf_sigma)
    features_test = make_features(x_test, centers, training_config.rbf_sigma)
    params = initialize_parameters(chain_config, features_train.shape[1])

    validation_count = min(48, training_config.n_train)
    gradient_table = gradient_validation(
        params,
        features_train[:validation_count],
        y_train[:validation_count],
        chain_config,
    )
    params, history = train_eqprop(
        params,
        features_train,
        y_train,
        features_test,
        y_test,
        chain_config,
        training_config,
    )
    q_train, train_iterations, train_residual = solve_equilibrium(params, features_train, chain_config)
    q_test, test_iterations, test_residual = solve_equilibrium(params, features_test, chain_config)
    train_loss, train_accuracy = classification_metrics(q_train, y_train)
    test_loss, test_accuracy = classification_metrics(q_test, y_test)

    # Choose a representative, non-extreme drive. Very large nonlinear drives
    # can form long-lived localized oscillations and obscure the boundary-flow
    # mechanism that this controlled test is intended to isolate.
    drive_norm = np.linalg.norm(input_force(params, features_test, chain_config), axis=1)
    ordered_drive = np.argsort(drive_norm)
    sample_index = int(ordered_drive[len(ordered_drive) // 2])
    dynamics = boundary_dynamics(params, features_test[sample_index : sample_index + 1], chain_config)

    gradient_table.to_csv(output_dir / "gradient_validation.csv", index=False)
    history.to_csv(output_dir / "classification_history.csv", index=False)
    dynamics_table = pd.DataFrame(
        {
            "time": dynamics["time"],
            "total_energy": dynamics["total_energy"],
            "potential": dynamics["potential"],
            "kinetic": dynamics["kinetic"],
            "dissipated_boundary_energy": dynamics["dissipated"],
            "balance_residual": dynamics["balance"],
            "distance_to_equilibrium": dynamics["distance_to_equilibrium"],
            "energy_gap": dynamics["energy_gap"],
        }
    )
    dynamics_table.to_csv(output_dir / "boundary_dynamics.csv", index=False)
    np.savez_compressed(
        output_dir / "trained_chain_model.npz",
        log_a=params.log_a,
        log_w=params.log_w,
        u=params.u,
        rbf_centers=centers,
        input_mean=mean,
        input_scale=scale,
    )
    make_plots(
        gradient_table,
        history,
        dynamics,
        x_train,
        y_train,
        x_test,
        y_test,
        q_train,
        q_test,
        output_dir,
    )

    summary = {
        "status": "nonlinear physical-chain proof-of-concept",
        "chain_config": asdict(chain_config),
        "training_config": asdict(training_config),
        "gradient_validation": gradient_table.to_dict(orient="records"),
        "training": {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_newton_iterations": train_iterations,
            "test_newton_iterations": test_iterations,
            "max_train_equilibrium_residual": train_residual,
            "max_test_equilibrium_residual": test_residual,
        },
        "boundary_dynamics": {
            "sample_index": sample_index,
            "relative_energy_balance_error": dynamics["relative_balance_error"],
            "monotonicity_violations": dynamics["monotonicity_violations"],
            "final_distance_to_equilibrium": dynamics["final_distance_to_equilibrium"],
            "final_velocity_norm": dynamics["final_velocity_norm"],
            "final_energy_gap": dynamics["final_energy_gap"],
        },
    }
    (output_dir / "boundary_rep_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results_boundary"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.output_dir, arguments.quick), indent=2))
