#!/usr/bin/env python3
"""Ten-class MNIST benchmark for parallel boundary-damped REP chains.

Ten independent physical chains form a simplex-coded multiclass output.  The
experiment first trains one common model with accurate centered EqProp
equilibria.  It then freezes that model and audits boundary-damped dynamics,
matched-trace uniform dynamics, overdamped EqProp, and implicit differentiation
against the same free and nudged equilibria.  Completed seeds are checkpointed.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-mnist")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

from boundary_rep_learning import Adam, inverse_softplus, sigmoid, softplus


METHODS = ("boundary_dynamic", "uniform_dynamic", "standard_eqprop", "implicit_id")


@dataclass(frozen=True)
class Profile:
    n_nodes: int
    pca_components: int
    seeds: tuple[int, ...]
    epochs: int
    batches_per_epoch: int
    batch_size: int
    audit_samples: int
    max_dynamic_steps: int
    dynamic_tolerance: float
    evaluation_chunk_size: int


PROFILES = {
    "quick": Profile(
        n_nodes=9,
        pca_components=16,
        seeds=(17,),
        epochs=3,
        batches_per_epoch=10,
        batch_size=128,
        audit_samples=32,
        max_dynamic_steps=40_000,
        dynamic_tolerance=1e-4,
        evaluation_chunk_size=1_000,
    ),
    "paper": Profile(
        n_nodes=17,
        pca_components=32,
        seeds=(17, 29, 43),
        epochs=24,
        batches_per_epoch=40,
        batch_size=256,
        audit_samples=128,
        max_dynamic_steps=120_000,
        dynamic_tolerance=2e-5,
        evaluation_chunk_size=1_000,
    ),
}


@dataclass(frozen=True)
class ModelConfig:
    n_nodes: int
    n_features: int
    n_classes: int = 10
    n_input_nodes: int = 2
    boundary_width: int = 3
    alpha: float = 0.05
    stiffness_floor: float = 0.03
    edge_floor: float = 0.45
    initial_onsite_stiffness: float = 0.15
    initial_edge_stiffness: float = 1.00
    beta: float = 0.01
    damping_trace: float = 3.00


@dataclass
class Parameters:
    log_a: np.ndarray
    log_w: np.ndarray
    u: np.ndarray

    def copy(self) -> "Parameters":
        return Parameters(self.log_a.copy(), self.log_w.copy(), self.u.copy())


def make_config(profile: Profile, n_features: int) -> ModelConfig:
    return ModelConfig(
        n_nodes=profile.n_nodes,
        n_features=n_features,
        boundary_width=max(1, round(profile.n_nodes / 3)),
    )


def initialize_parameters(config: ModelConfig, seed: int) -> Parameters:
    rng = np.random.default_rng(seed + 10_000)
    log_a = np.full(
        config.n_nodes,
        inverse_softplus(config.initial_onsite_stiffness - config.stiffness_floor),
    )
    log_w = np.full(
        config.n_nodes - 1,
        inverse_softplus(config.initial_edge_stiffness - config.edge_floor),
    )
    log_a += rng.normal(scale=0.04, size=log_a.shape)
    log_w += rng.normal(scale=0.04, size=log_w.shape)
    u = rng.normal(
        scale=0.08,
        size=(config.n_classes, config.n_input_nodes, config.n_features),
    )
    return Parameters(log_a, log_w, u)


def physical_coefficients(
    params: Parameters, config: ModelConfig
) -> tuple[np.ndarray, np.ndarray]:
    return (
        softplus(params.log_a) + config.stiffness_floor,
        softplus(params.log_w) + config.edge_floor,
    )


def simplex_targets(labels: np.ndarray, n_classes: int) -> np.ndarray:
    targets = np.full((labels.size, n_classes), -1.0 / (n_classes - 1))
    targets[np.arange(labels.size), labels] = 1.0
    return targets


def state_gradient(
    q: np.ndarray,
    params: Parameters,
    features: np.ndarray,
    config: ModelConfig,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
) -> np.ndarray:
    """Gradient for q with shape (samples, classes, chain nodes)."""
    a, w = physical_coefficients(params, config)
    gradient = a[None, None, :] * q + config.alpha * q**3
    drive = np.einsum("bf,kif->bki", features, params.u)
    gradient[:, :, : config.n_input_nodes] -= drive
    difference = q[:, :, 1:] - q[:, :, :-1]
    gradient[:, :, :-1] -= w[None, None, :] * difference
    gradient[:, :, 1:] += w[None, None, :] * difference
    if beta != 0.0:
        if targets is None:
            raise ValueError("targets are required for a nudged equilibrium")
        gradient[:, :, -1] += beta * (q[:, :, -1] - targets)
    return gradient


def state_hessian(
    q: np.ndarray,
    params: Parameters,
    config: ModelConfig,
    beta: float = 0.0,
) -> np.ndarray:
    a, w = physical_coefficients(params, config)
    batch, classes, nodes = q.shape
    hessian = np.zeros((batch, classes, nodes, nodes))
    diagonal = a[None, None, :] + 3.0 * config.alpha * q**2
    diagonal[:, :, :-1] += w[None, None, :]
    diagonal[:, :, 1:] += w[None, None, :]
    diagonal[:, :, -1] += beta
    indices = np.arange(nodes)
    hessian[:, :, indices, indices] = diagonal
    edges = np.arange(nodes - 1)
    hessian[:, :, edges, edges + 1] = -w[None, None, :]
    hessian[:, :, edges + 1, edges] = -w[None, None, :]
    return hessian


def solve_equilibrium(
    params: Parameters,
    features: np.ndarray,
    config: ModelConfig,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial: np.ndarray | None = None,
    tolerance: float = 2e-10,
    max_iterations: int = 60,
) -> tuple[np.ndarray, int, float]:
    shape = (features.shape[0], config.n_classes, config.n_nodes)
    q = np.zeros(shape) if initial is None else initial.copy()
    for iteration in range(max_iterations + 1):
        gradient = state_gradient(q, params, features, config, targets, beta)
        residual = float(np.max(np.linalg.norm(gradient, axis=2)))
        if residual <= tolerance:
            return q, iteration, residual
        hessian = state_hessian(q, params, config, beta)
        if beta < 0.0 and iteration == 0:
            minimum_eigenvalue = float(
                np.min(np.linalg.eigvalsh(hessian.reshape(-1, config.n_nodes, config.n_nodes)))
            )
            if minimum_eigenvalue <= 1e-8:
                raise RuntimeError(
                    f"nudged Hessian lost positive definiteness ({minimum_eigenvalue:.3e})"
                )
        step = np.linalg.solve(
            hessian.reshape(-1, config.n_nodes, config.n_nodes),
            gradient.reshape(-1, config.n_nodes, 1),
        ).reshape(shape)
        q -= (0.70 if iteration < 3 else 1.0) * step
    raise RuntimeError(f"Newton solver did not converge; residual={residual:.3e}")


def physical_energy_gradients(
    q: np.ndarray, params: Parameters, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = q[:, :, 1:] - q[:, :, :-1]
    grad_a = sigmoid(params.log_a) * np.mean(
        np.sum(0.5 * q**2, axis=1), axis=0
    )
    grad_w = sigmoid(params.log_w) * np.mean(
        np.sum(0.5 * difference**2, axis=1), axis=0
    )
    grad_u = -np.einsum(
        "bki,bf->kif", q[:, :, : params.u.shape[1]], features
    ) / features.shape[0]
    return grad_a, grad_w, grad_u


def centered_physical_gradients(
    q_minus: np.ndarray,
    q_plus: np.ndarray,
    params: Parameters,
    features: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minus = physical_energy_gradients(q_minus, params, features)
    plus = physical_energy_gradients(q_plus, params, features)
    return tuple((p - m) / (2.0 * beta) for m, p in zip(minus, plus))


def implicit_physical_gradients(
    q_free: np.ndarray,
    params: Parameters,
    features: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cost_gradient = np.zeros_like(q_free)
    cost_gradient[:, :, -1] = q_free[:, :, -1] - targets
    hessian = state_hessian(q_free, params, config)
    adjoint = np.linalg.solve(
        hessian.reshape(-1, config.n_nodes, config.n_nodes),
        cost_gradient.reshape(-1, config.n_nodes, 1),
    ).reshape(q_free.shape)
    difference_q = q_free[:, :, 1:] - q_free[:, :, :-1]
    difference_adjoint = adjoint[:, :, 1:] - adjoint[:, :, :-1]
    grad_a = -sigmoid(params.log_a) * np.mean(
        np.sum(adjoint * q_free, axis=1), axis=0
    )
    grad_w = -sigmoid(params.log_w) * np.mean(
        np.sum(difference_adjoint * difference_q, axis=1), axis=0
    )
    grad_u = np.einsum(
        "bki,bf->kif", adjoint[:, :, : config.n_input_nodes], features
    ) / features.shape[0]
    return grad_a, grad_w, grad_u


def flatten_physical(gradient: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate([part.ravel() for part in gradient])


def classification_metrics(q: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    targets = simplex_targets(labels, q.shape[1])
    prediction = np.argmax(q[:, :, -1], axis=1)
    return {
        "loss": float(np.mean(0.5 * np.sum((q[:, :, -1] - targets) ** 2, axis=1))),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
    }


def load_mnist(cache_dir: Path, pca_components: int) -> tuple[np.ndarray, ...]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared = cache_dir / f"mnist_pca_{pca_components}.npz"
    if prepared.exists():
        with np.load(prepared, allow_pickle=False) as stored:
            return (
                np.asarray(stored["x_train"]).copy(),
                np.asarray(stored["y_train"]).copy(),
                np.asarray(stored["x_test"]).copy(),
                np.asarray(stored["y_test"]).copy(),
            )
    print("Downloading/loading MNIST from OpenML. This occurs only once per PCA cache.")
    dataset = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto",
        data_home=cache_dir / "openml",
    )
    x = np.asarray(dataset.data, dtype=np.float32) / 255.0
    y = np.asarray(dataset.target, dtype=np.int64)
    x_train_raw, x_test_raw = x[:60_000], x[60_000:]
    y_train, y_test = y[:60_000], y[60_000:]
    pca = PCA(n_components=pca_components, whiten=True, random_state=2026)
    x_train = pca.fit_transform(x_train_raw).astype(np.float32)
    x_test = pca.transform(x_test_raw).astype(np.float32)
    x_train = 0.35 * np.clip(x_train, -5.0, 5.0)
    x_test = 0.35 * np.clip(x_test, -5.0, 5.0)
    x_train = np.column_stack((np.ones(x_train.shape[0], dtype=np.float32), x_train))
    x_test = np.column_stack((np.ones(x_test.shape[0], dtype=np.float32), x_test))
    np.savez_compressed(
        prepared, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )
    return x_train, y_train, x_test, y_test


def evaluate(
    params: Parameters,
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    chunk_size: int,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    weighted_loss = 0.0
    maximum_residual = 0.0
    for start in range(0, labels.size, chunk_size):
        stop = min(start + chunk_size, labels.size)
        q, _, residual = solve_equilibrium(params, features[start:stop], config)
        metrics = classification_metrics(q, labels[start:stop])
        weighted_loss += metrics["loss"] * (stop - start)
        predictions.append(np.argmax(q[:, :, -1], axis=1))
        maximum_residual = max(maximum_residual, residual)
    prediction = np.concatenate(predictions)
    return {
        "loss": weighted_loss / labels.size,
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "max_equilibrium_residual": maximum_residual,
    }


def train_seed(
    seed: int,
    profile: Profile,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[Parameters, list[dict[str, object]], dict[str, object]]:
    config = make_config(profile, x_train.shape[1])
    params = initialize_parameters(config, seed)
    rng = np.random.default_rng(seed)
    opt_a = Adam(params.log_a.shape, 1e-4)
    opt_w = Adam(params.log_w.shape, 1e-4)
    opt_u = Adam(params.u.shape, 2e-2)
    history: list[dict[str, object]] = []
    start_time = time.perf_counter()
    report_every = max(1, profile.epochs // 6)

    for epoch in range(profile.epochs + 1):
        if epoch > 0:
            for _ in range(profile.batches_per_epoch):
                indices = rng.integers(0, x_train.shape[0], size=profile.batch_size)
                features_batch = x_train[indices]
                targets_batch = simplex_targets(y_train[indices], config.n_classes)
                q_free, _, _ = solve_equilibrium(params, features_batch, config)
                q_plus, _, _ = solve_equilibrium(
                    params, features_batch, config, targets_batch, config.beta, q_free
                )
                q_minus, _, _ = solve_equilibrium(
                    params, features_batch, config, targets_batch, -config.beta, q_free
                )
                grad_a, grad_w, grad_u = centered_physical_gradients(
                    q_minus, q_plus, params, features_batch, config.beta
                )
                grad_u += 2e-4 * params.u
                params.log_a = opt_a.update(params.log_a, grad_a)
                params.log_w = opt_w.update(params.log_w, grad_w)
                params.u = opt_u.update(params.u, grad_u)

        if epoch == 0 or epoch == profile.epochs or epoch % report_every == 0:
            train_indices = rng.choice(
                x_train.shape[0], size=min(2_000, x_train.shape[0]), replace=False
            )
            train_metrics = evaluate(
                params,
                x_train[train_indices],
                y_train[train_indices],
                config,
                profile.evaluation_chunk_size,
            )
            test_count = min(2_000, x_test.shape[0]) if profile.epochs < 5 else x_test.shape[0]
            test_metrics = evaluate(
                params,
                x_test[:test_count],
                y_test[:test_count],
                config,
                profile.evaluation_chunk_size,
            )
            row: dict[str, object] = {"seed": seed, "epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"test_{key}": value for key, value in test_metrics.items()})
            history.append(row)
            print(
                f"seed={seed} epoch={epoch}/{profile.epochs} "
                f"test_accuracy={test_metrics['accuracy']:.4f}"
            )

    final_train = evaluate(params, x_train, y_train, config, profile.evaluation_chunk_size)
    final_test = evaluate(params, x_test, y_test, config, profile.evaluation_chunk_size)
    result: dict[str, object] = {
        "seed": seed,
        "n_parallel_chains": config.n_classes,
        "n_nodes_per_chain": config.n_nodes,
        "boundary_width": config.boundary_width,
        "pca_components": profile.pca_components,
        "epochs": profile.epochs,
        "batches_per_epoch": profile.batches_per_epoch,
        "batch_size": profile.batch_size,
        "training_pool_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "wall_time_seconds": time.perf_counter() - start_time,
        **{f"train_{key}": value for key, value in final_train.items()},
        **{f"test_{key}": value for key, value in final_test.items()},
    }
    return params, history, result


def damping_vector(config: ModelConfig, method: str) -> np.ndarray:
    ramp = np.arange(1.0, config.boundary_width + 1.0) ** 2
    boundary = np.zeros(config.n_nodes)
    boundary[-config.boundary_width :] = config.damping_trace * ramp / np.sum(ramp)
    if method == "boundary_dynamic":
        return boundary
    if method == "uniform_dynamic":
        return np.full(config.n_nodes, config.damping_trace / config.n_nodes)
    raise ValueError(method)


def relax_wave(
    params: Parameters,
    features: np.ndarray,
    config: ModelConfig,
    profile: Profile,
    method: str,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial: np.ndarray | None = None,
) -> dict[str, object]:
    shape = (features.shape[0], config.n_classes, config.n_nodes)
    q = np.zeros(shape) if initial is None else initial.copy()
    velocity = np.zeros_like(q)
    damping = damping_vector(config, method)
    dt = 0.16
    half_decay = np.exp(-0.5 * dt * damping)[None, None, :]
    consecutive = 0
    ramp_steps = min(5_000, max(500, profile.max_dynamic_steps // 8))
    phase_tolerance = profile.dynamic_tolerance * (5.0 if beta == 0.0 else 1.0)
    start = time.perf_counter()
    for step in range(1, profile.max_dynamic_steps + 1):
        if initial is None:
            drive_scale = min(1.0, step / ramp_steps)
            active_features = features * drive_scale
        else:
            drive_scale = 1.0
            active_features = features
        velocity *= half_decay
        gradient = state_gradient(q, params, active_features, config, targets, beta)
        velocity -= 0.5 * dt * gradient
        q += dt * velocity
        gradient = state_gradient(q, params, active_features, config, targets, beta)
        velocity -= 0.5 * dt * gradient
        velocity *= half_decay
        if step % 25 == 0:
            residual = float(np.max(np.linalg.norm(gradient, axis=2)))
            velocity_norm = float(np.max(np.linalg.norm(velocity, axis=2)))
            reached_full_drive = drive_scale >= 1.0
            consecutive = consecutive + 1 if reached_full_drive and max(residual, velocity_norm) <= phase_tolerance else 0
            if consecutive >= 3:
                break
    residual = float(
        np.max(np.linalg.norm(state_gradient(q, params, features, config, targets, beta), axis=2))
    )
    velocity_norm = float(np.max(np.linalg.norm(velocity, axis=2)))
    return {
        "q": q,
        "steps": step,
        "residual": residual,
        "velocity_norm": velocity_norm,
        "converged": consecutive >= 3,
        "wall_time_seconds": time.perf_counter() - start,
    }


def relax_overdamped(
    params: Parameters,
    features: np.ndarray,
    config: ModelConfig,
    profile: Profile,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial: np.ndarray | None = None,
) -> dict[str, object]:
    shape = (features.shape[0], config.n_classes, config.n_nodes)
    q = np.zeros(shape) if initial is None else initial.copy()
    consecutive = 0
    lipschitz = 1.0
    phase_tolerance = profile.dynamic_tolerance * (5.0 if beta == 0.0 else 1.0)
    start = time.perf_counter()
    for step in range(1, profile.max_dynamic_steps + 1):
        gradient = state_gradient(q, params, features, config, targets, beta)
        if step == 1 or step % 25 == 0:
            hessian = state_hessian(q, params, config, beta)
            lipschitz = float(np.max(np.sum(np.abs(hessian), axis=3)))
        q -= (0.85 / max(lipschitz, 1e-8)) * gradient
        if step % 25 == 0:
            residual = float(
                np.max(
                    np.linalg.norm(
                        state_gradient(q, params, features, config, targets, beta),
                        axis=2,
                    )
                )
            )
            consecutive = consecutive + 1 if residual <= phase_tolerance else 0
            if consecutive >= 3:
                break
    residual = float(
        np.max(np.linalg.norm(state_gradient(q, params, features, config, targets, beta), axis=2))
    )
    return {
        "q": q,
        "steps": step,
        "residual": residual,
        "velocity_norm": 0.0,
        "converged": consecutive >= 3,
        "wall_time_seconds": time.perf_counter() - start,
    }


def solver_endpoint(
    method: str,
    params: Parameters,
    features: np.ndarray,
    config: ModelConfig,
    profile: Profile,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial: np.ndarray | None = None,
) -> dict[str, object]:
    if method in ("boundary_dynamic", "uniform_dynamic"):
        return relax_wave(params, features, config, profile, method, targets, beta, initial)
    if method == "standard_eqprop":
        return relax_overdamped(params, features, config, profile, targets, beta, initial)
    start = time.perf_counter()
    q, steps, residual = solve_equilibrium(params, features, config, targets, beta, initial)
    return {
        "q": q,
        "steps": steps,
        "residual": residual,
        "velocity_norm": 0.0,
        "converged": True,
        "wall_time_seconds": time.perf_counter() - start,
    }


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-14))


def cosine(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(value) * np.linalg.norm(reference)
    return float(np.dot(value, reference) / max(denominator, 1e-14))


def audit_solvers(
    seed: int,
    params: Parameters,
    profile: Profile,
    features: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, object]]:
    config = make_config(profile, features.shape[1])
    rng = np.random.default_rng(seed + 900_000)
    indices = rng.choice(labels.size, size=min(profile.audit_samples, labels.size), replace=False)
    x = features[indices]
    targets = simplex_targets(labels[indices], config.n_classes)
    q_free_exact, _, _ = solve_equilibrium(params, x, config)
    q_plus_exact, _, _ = solve_equilibrium(params, x, config, targets, config.beta, q_free_exact)
    q_minus_exact, _, _ = solve_equilibrium(params, x, config, targets, -config.beta, q_free_exact)
    exact_centered = flatten_physical(
        centered_physical_gradients(q_minus_exact, q_plus_exact, params, x, config.beta)
    )
    exact_implicit = flatten_physical(
        implicit_physical_gradients(q_free_exact, params, x, targets, config)
    )
    rows: list[dict[str, object]] = []

    for method in METHODS:
        free = solver_endpoint(method, params, x, config, profile)
        plus = solver_endpoint(
            method, params, x, config, profile, targets, config.beta, np.asarray(free["q"])
        )
        minus = solver_endpoint(
            method, params, x, config, profile, targets, -config.beta, np.asarray(free["q"])
        )
        if method == "implicit_id":
            estimate = exact_implicit
        else:
            estimate = flatten_physical(
                centered_physical_gradients(
                    np.asarray(minus["q"]), np.asarray(plus["q"]), params, x, config.beta
                )
            )
        rows.append(
            {
                "seed": seed,
                "method": method,
                "audit_samples": x.shape[0],
                "all_endpoints_converged": bool(free["converged"] and plus["converged"] and minus["converged"]),
                "free_state_relative_error": relative_error(np.asarray(free["q"]), q_free_exact),
                "plus_state_relative_error": relative_error(np.asarray(plus["q"]), q_plus_exact),
                "minus_state_relative_error": relative_error(np.asarray(minus["q"]), q_minus_exact),
                "gradient_vs_implicit_relative_error": relative_error(estimate, exact_implicit),
                "gradient_vs_implicit_cosine": cosine(estimate, exact_implicit),
                "exact_centered_vs_implicit_relative_error": relative_error(exact_centered, exact_implicit),
                "total_steps": int(free["steps"]) + int(plus["steps"]) + int(minus["steps"]),
                "wall_time_seconds": float(free["wall_time_seconds"]) + float(plus["wall_time_seconds"]) + float(minus["wall_time_seconds"]),
                "maximum_residual": max(float(free["residual"]), float(plus["residual"]), float(minus["residual"])),
                "maximum_velocity_norm": max(float(free["velocity_norm"]), float(plus["velocity_norm"]), float(minus["velocity_norm"])),
            }
        )
    return rows


def save_model(path: Path, params: Parameters, config: ModelConfig) -> None:
    np.savez_compressed(
        path,
        log_a=params.log_a,
        log_w=params.log_w,
        u=params.u,
        config_json=json.dumps(asdict(config)),
    )


def logistic_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    start = time.perf_counter()
    model = LogisticRegression(max_iter=700, solver="lbfgs", random_state=2026)
    model.fit(x_train[:, 1:], y_train)
    prediction = model.predict(x_test[:, 1:])
    return {
        "method": "pca_logistic_regression",
        "test_accuracy": float(np.mean(prediction == y_test)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "test_macro_f1": float(f1_score(y_test, prediction, average="macro")),
        "wall_time_seconds": time.perf_counter() - start,
    }


def make_plot(
    history: pd.DataFrame,
    audit: pd.DataFrame,
    baseline: dict[str, object],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for seed, block in history.groupby("seed"):
        axes[0].plot(block["epoch"], block["test_accuracy"], marker="o", label=f"seed={seed}")
    axes[0].axhline(float(baseline["test_accuracy"]), color="black", linestyle="--", label="PCA logistic")
    axes[0].set(xlabel="epoch", ylabel="test accuracy", title="MNIST learning")
    axes[0].legend(fontsize=8)
    grouped = audit.groupby("method", as_index=False)["gradient_vs_implicit_relative_error"].median()
    axes[1].bar(grouped["method"], grouped["gradient_vs_implicit_relative_error"])
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(ylabel="relative error", title="Gradient audit")
    runtime = audit.groupby("method", as_index=False)["wall_time_seconds"].median()
    axes[2].bar(runtime["method"], runtime["wall_time_seconds"])
    axes[2].set_yscale("log")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set(ylabel="wall time [s]", title="Endpoint audit runtime")
    fig.tight_layout()
    fig.savefig(output_dir / "mnist_rep_overview.png", dpi=190)
    plt.close(fig)


def run(profile_name: str, output_dir: Path, cache_dir: Path) -> dict[str, object]:
    profile = PROFILES[profile_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_mnist(cache_dir, profile.pca_components)
    baseline_path = output_dir / "mnist_pca_logistic_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = logistic_baseline(x_train, y_train, x_test, y_test)
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    runs_path = output_dir / "mnist_runs.csv"
    history_path = output_dir / "mnist_history.csv"
    audit_path = output_dir / "mnist_endpoint_audit.csv"
    runs = pd.read_csv(runs_path).to_dict(orient="records") if runs_path.exists() else []
    history = pd.read_csv(history_path).to_dict(orient="records") if history_path.exists() else []
    audits = pd.read_csv(audit_path).to_dict(orient="records") if audit_path.exists() else []
    completed = {int(row["seed"]) for row in runs}

    for seed in profile.seeds:
        if seed in completed:
            print(f"seed={seed} already completed; skipping")
            continue
        # Remove remnants of an interrupted, not-yet-committed seed before
        # rerunning it. mnist_runs.csv is the completion marker.
        history = [row for row in history if int(row["seed"]) != seed]
        audits = [row for row in audits if int(row["seed"]) != seed]
        params, seed_history, result = train_seed(
            seed, profile, x_train, y_train, x_test, y_test
        )
        config = make_config(profile, x_train.shape[1])
        seed_audit = audit_solvers(seed, params, profile, x_test, y_test)
        save_model(output_dir / f"mnist_model_seed_{seed}.npz", params, config)
        runs.append(result)
        history.extend(seed_history)
        audits.extend(seed_audit)
        pd.DataFrame(history).to_csv(history_path, index=False)
        pd.DataFrame(audits).to_csv(audit_path, index=False)
        # Write the completion marker last.
        pd.DataFrame(runs).to_csv(runs_path, index=False)

    runs_df = pd.DataFrame(runs)
    history_df = pd.DataFrame(history)
    audit_df = pd.DataFrame(audits)
    make_plot(history_df, audit_df, baseline, output_dir)
    dynamic = audit_df[audit_df["method"].isin(["boundary_dynamic", "uniform_dynamic", "standard_eqprop"])]
    boundary = audit_df[audit_df["method"] == "boundary_dynamic"]
    summary = {
        "version": "3.1-mnist",
        "profile": profile_name,
        "task": "official ten-class MNIST",
        "architecture": "ten parallel simplex-coded nonlinear chains",
        "training_protocol": "centered EqProp with exact equilibria; stochastic mini-batches from all 60000 training images",
        "endpoint_protocol": "same trained model audited with boundary, uniform, overdamped, and implicit solvers",
        "profile_config": asdict(profile),
        "n_completed_seeds": int(runs_df.shape[0]),
        "mean_test_accuracy": float(runs_df["test_accuracy"].mean()),
        "std_test_accuracy": float(runs_df["test_accuracy"].std(ddof=1)) if runs_df.shape[0] > 1 else 0.0,
        "mean_test_balanced_accuracy": float(runs_df["test_balanced_accuracy"].mean()),
        "mean_test_macro_f1": float(runs_df["test_macro_f1"].mean()),
        "pca_logistic_test_accuracy": float(baseline["test_accuracy"]),
        "mean_accuracy_difference_vs_pca_logistic": float(runs_df["test_accuracy"].mean() - float(baseline["test_accuracy"])),
        "all_dynamic_endpoints_converged": bool(dynamic["all_endpoints_converged"].all()),
        "boundary_median_gradient_relative_error": float(boundary["gradient_vs_implicit_relative_error"].median()),
        "boundary_minimum_gradient_cosine": float(boundary["gradient_vs_implicit_cosine"].min()),
        "exact_centered_median_gradient_relative_error": float(audit_df["exact_centered_vs_implicit_relative_error"].median()),
        "passes_core_gate": bool(
            runs_df.shape[0] == len(profile.seeds)
            and np.isfinite(runs_df["test_accuracy"]).all()
            and dynamic["all_endpoints_converged"].all()
            and boundary["gradient_vs_implicit_cosine"].min() >= 0.99
        ),
    }
    (output_dir / "mnist_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="paper")
    parser.add_argument("--output-dir", type=Path, default=Path("results_v3_mnist"))
    parser.add_argument("--cache-dir", type=Path, default=Path("mnist_cache"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.profile, args.output_dir, args.cache_dir)
