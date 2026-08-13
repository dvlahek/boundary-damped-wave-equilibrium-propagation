#!/usr/bin/env python3
"""Disjoint calibration and confirmatory audit for MNIST boundary damping.

The trained v3.1 MNIST models are never changed.  A boundary damping profile is
selected using seed 17 and training-pool calibration samples only.  The profile
is then locked before evaluation on official test samples.  Model seeds 29 and
43 are the confirmatory model seeds; seed 17 is retained as a transparent
calibration-support result.  Budget curves are obtained from single continued
physical trajectories rather than independent reruns at every horizon.
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
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-mnist-v33")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnist_rep_benchmark import (
    ModelConfig,
    Parameters,
    centered_physical_gradients,
    flatten_physical,
    implicit_physical_gradients,
    physical_coefficients,
    simplex_targets,
    solve_equilibrium,
    state_gradient,
    state_hessian,
)


FREE_TOLERANCE = 1e-4
NUDGED_TOLERANCE = 2e-5
STATE_ERROR_THRESHOLD = 1e-4
GRADIENT_ERROR_THRESHOLD = 1e-2
GRADIENT_COSINE_THRESHOLD = 0.999
CHECK_EVERY = 25
CONSECUTIVE_CHECKS = 3
DT = 0.16


@dataclass(frozen=True)
class DampingCandidate:
    width: int
    trace: float
    power: float

    @property
    def key(self) -> str:
        return f"width={self.width}|trace={self.trace:g}|power={self.power:g}"


def candidate_grid(n_nodes: int) -> list[DampingCandidate]:
    base = max(1, round(n_nodes / 3))
    wide = max(base + 1, n_nodes // 2)
    candidates = [
        DampingCandidate(base, trace, power)
        for trace in (1.5, 3.0, 5.0, 8.0)
        for power in (1.0, 2.0)
    ]
    candidates.extend(
        DampingCandidate(wide, trace, power)
        for trace in (1.5, 3.0, 5.0)
        for power in (1.0, 2.0)
    )
    candidates.extend(
        (DampingCandidate(base, 3.0, 0.0), DampingCandidate(wide, 3.0, 0.0))
    )
    unique: dict[str, DampingCandidate] = {candidate.key: candidate for candidate in candidates}
    return list(unique.values())


def damping_vector(candidate: DampingCandidate, n_nodes: int) -> np.ndarray:
    if not 1 <= candidate.width < n_nodes:
        raise ValueError("boundary width must be between 1 and n_nodes - 1")
    damping = np.zeros(n_nodes)
    if candidate.power == 0.0:
        weights = np.ones(candidate.width)
    else:
        weights = np.arange(1.0, candidate.width + 1.0) ** candidate.power
    damping[-candidate.width :] = candidate.trace * weights / np.sum(weights)
    return damping


def load_model(path: Path) -> tuple[Parameters, ModelConfig]:
    with np.load(path, allow_pickle=False) as stored:
        config_data = json.loads(str(stored["config_json"].item()))
        params = Parameters(
            np.asarray(stored["log_a"]).copy(),
            np.asarray(stored["log_w"]).copy(),
            np.asarray(stored["u"]).copy(),
        )
    return params, ModelConfig(**config_data)


def load_models(models_dir: Path) -> dict[int, tuple[Parameters, ModelConfig]]:
    models: dict[int, tuple[Parameters, ModelConfig]] = {}
    for path in sorted(models_dir.glob("mnist_model_seed_*.npz")):
        seed = int(path.stem.rsplit("_", 1)[-1])
        models[seed] = load_model(path)
    if not models:
        raise FileNotFoundError(
            f"No mnist_model_seed_*.npz files were found in {models_dir}"
        )
    return models


def pair_gradient(
    q: np.ndarray,
    sample_indices: np.ndarray,
    class_indices: np.ndarray,
    params: Parameters,
    features: np.ndarray,
    config: ModelConfig,
    targets: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    a, w = physical_coefficients(params, config)
    gradient = a[None, :] * q + config.alpha * q**3
    selected_features = features[sample_indices]
    selected_u = params.u[class_indices]
    drive = np.einsum("pf,pif->pi", selected_features, selected_u)
    gradient[:, : config.n_input_nodes] -= drive
    difference = q[:, 1:] - q[:, :-1]
    gradient[:, :-1] -= w[None, :] * difference
    gradient[:, 1:] += w[None, :] * difference
    if beta != 0.0:
        if targets is None:
            raise ValueError("targets are required for nudged dynamics")
        gradient[:, -1] += beta * (
            q[:, -1] - targets[sample_indices, class_indices]
        )
    return gradient


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-14))


def cosine(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(value) * np.linalg.norm(reference)
    return float(np.dot(value, reference) / max(denominator, 1e-14))


def relax_phase(
    params: Parameters,
    features: np.ndarray,
    config: ModelConfig,
    candidate: DampingCandidate,
    budgets: tuple[int, ...],
    *,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial: np.ndarray | None = None,
) -> dict[str, object]:
    budgets = tuple(sorted(set(int(value) for value in budgets)))
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    maximum_budget = budgets[-1]
    shape = (features.shape[0], config.n_classes, config.n_nodes)
    q = np.zeros(shape) if initial is None else initial.copy()
    velocity = np.zeros_like(q)
    damping = damping_vector(candidate, config.n_nodes)
    half_decay = np.exp(-0.5 * DT * damping)[None, :]
    consecutive = np.zeros(shape[:2], dtype=np.int16)
    converged = np.zeros(shape[:2], dtype=bool)
    channel_steps = np.zeros(shape[:2], dtype=np.int64)
    active_channel_steps = 0
    tolerance = FREE_TOLERANCE if beta == 0.0 else NUDGED_TOLERANCE
    ramp_steps = min(5_000, max(500, maximum_budget // 8)) if initial is None else 0
    snapshots: dict[int, dict[str, object]] = {}
    start_time = time.perf_counter()
    flat_q = q.reshape(-1, config.n_nodes)
    flat_velocity = velocity.reshape(-1, config.n_nodes)
    n_classes = config.n_classes

    for step in range(1, maximum_budget + 1):
        active_flat = np.flatnonzero(~converged.ravel())
        if active_flat.size:
            sample_indices = active_flat // n_classes
            class_indices = active_flat % n_classes
            drive_scale = min(1.0, step / ramp_steps) if ramp_steps else 1.0
            active_features = features * drive_scale
            q_active = flat_q[active_flat].copy()
            velocity_active = flat_velocity[active_flat].copy() * half_decay
            gradient = pair_gradient(
                q_active,
                sample_indices,
                class_indices,
                params,
                active_features,
                config,
                targets,
                beta,
            )
            velocity_active -= 0.5 * DT * gradient
            q_active += DT * velocity_active
            gradient_new = pair_gradient(
                q_active,
                sample_indices,
                class_indices,
                params,
                active_features,
                config,
                targets,
                beta,
            )
            velocity_active -= 0.5 * DT * gradient_new
            velocity_active *= half_decay
            flat_q[active_flat] = q_active
            flat_velocity[active_flat] = velocity_active
            channel_steps.ravel()[active_flat] += 1
            active_channel_steps += int(active_flat.size)

            if step % CHECK_EVERY == 0:
                residual = np.linalg.norm(gradient_new, axis=1)
                speed = np.linalg.norm(velocity_active, axis=1)
                locally_converged = (residual <= tolerance) & (speed <= tolerance)
                if drive_scale < 1.0:
                    locally_converged[:] = False
                flat_consecutive = consecutive.ravel()
                flat_consecutive[active_flat] = np.where(
                    locally_converged,
                    flat_consecutive[active_flat] + 1,
                    0,
                )
                newly_converged = active_flat[
                    flat_consecutive[active_flat] >= CONSECUTIVE_CHECKS
                ]
                converged.ravel()[newly_converged] = True

        if step in budgets:
            final_gradient = state_gradient(
                q, params, features, config, targets, beta
            )
            residuals = np.linalg.norm(final_gradient, axis=2)
            speeds = np.linalg.norm(velocity, axis=2)
            snapshots[step] = {
                "q": q.copy(),
                "converged": converged.copy(),
                "residuals": residuals,
                "speeds": speeds,
                "active_channel_steps": int(active_channel_steps),
                "channel_steps": channel_steps.copy(),
                "wall_time_seconds": time.perf_counter() - start_time,
            }

        if not active_flat.size:
            for budget in budgets:
                if budget > step and budget not in snapshots:
                    final_gradient = state_gradient(
                        q, params, features, config, targets, beta
                    )
                    snapshots[budget] = {
                        "q": q.copy(),
                        "converged": converged.copy(),
                        "residuals": np.linalg.norm(final_gradient, axis=2),
                        "speeds": np.linalg.norm(velocity, axis=2),
                        "active_channel_steps": int(active_channel_steps),
                        "channel_steps": channel_steps.copy(),
                        "wall_time_seconds": time.perf_counter() - start_time,
                    }
            break

    return {
        "snapshots": snapshots,
        "final_q": q,
        "damping": damping,
        "tolerance": tolerance,
        "solver_steps": min(step, maximum_budget),
        "wall_time_seconds": time.perf_counter() - start_time,
    }


def exact_references(
    params: Parameters,
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
) -> dict[str, object]:
    targets = simplex_targets(labels, config.n_classes)
    q_free, _, _ = solve_equilibrium(params, features, config)
    q_plus, _, _ = solve_equilibrium(
        params, features, config, targets, config.beta, q_free
    )
    q_minus, _, _ = solve_equilibrium(
        params, features, config, targets, -config.beta, q_free
    )
    centered = flatten_physical(
        centered_physical_gradients(
            q_minus, q_plus, params, features, config.beta
        )
    )
    implicit = flatten_physical(
        implicit_physical_gradients(
            q_free, params, features, targets, config
        )
    )
    return {
        "targets": targets,
        "free": q_free,
        "plus": q_plus,
        "minus": q_minus,
        "centered": centered,
        "implicit": implicit,
    }


def phase_row(
    seed: int,
    phase: str,
    budget: int,
    snapshot: dict[str, object],
    exact_state: np.ndarray,
    tolerance: float,
    candidate: DampingCandidate,
    confirmatory_model_seed: bool,
) -> dict[str, object]:
    converged = np.asarray(snapshot["converged"], dtype=bool)
    residuals = np.asarray(snapshot["residuals"])
    speeds = np.asarray(snapshot["speeds"])
    channel_steps = np.asarray(snapshot["channel_steps"])
    return {
        "seed": seed,
        "confirmatory_model_seed": confirmatory_model_seed,
        "candidate": candidate.key,
        "phase": phase,
        "budget": budget,
        "strict_converged": bool(np.all(converged)),
        "converged_fraction": float(np.mean(converged)),
        "n_converged_channels": int(np.sum(converged)),
        "n_total_channels": int(converged.size),
        "state_relative_error": relative_error(np.asarray(snapshot["q"]), exact_state),
        "maximum_residual": float(np.max(residuals)),
        "maximum_velocity_norm": float(np.max(speeds)),
        "median_residual": float(np.median(residuals)),
        "median_velocity_norm": float(np.median(speeds)),
        "registered_tolerance": tolerance,
        "active_channel_steps": int(snapshot["active_channel_steps"]),
        "maximum_channel_steps": int(np.max(channel_steps)),
        "median_channel_steps": float(np.median(channel_steps)),
        "wall_time_seconds": float(snapshot["wall_time_seconds"]),
    }


def run_locked_audit(
    seed: int,
    params: Parameters,
    config: ModelConfig,
    features: np.ndarray,
    labels: np.ndarray,
    candidate: DampingCandidate,
    budgets: tuple[int, ...],
    *,
    confirmatory_model_seed: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    reference = exact_references(params, features, labels, config)
    targets = np.asarray(reference["targets"])
    free = relax_phase(
        params, features, config, candidate, budgets, beta=0.0
    )
    # The final dynamically generated free state is the physical warm start for
    # both centered nudged trajectories. No exact state initializes dynamics.
    plus = relax_phase(
        params,
        features,
        config,
        candidate,
        budgets,
        targets=targets,
        beta=config.beta,
        initial=np.asarray(free["final_q"]),
    )
    minus = relax_phase(
        params,
        features,
        config,
        candidate,
        budgets,
        targets=targets,
        beta=-config.beta,
        initial=np.asarray(free["final_q"]),
    )

    phase_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for budget in budgets:
        snapshots = {
            "free": free["snapshots"][budget],
            "plus": plus["snapshots"][budget],
            "minus": minus["snapshots"][budget],
        }
        exact_states = {
            "free": np.asarray(reference["free"]),
            "plus": np.asarray(reference["plus"]),
            "minus": np.asarray(reference["minus"]),
        }
        tolerances = {
            "free": FREE_TOLERANCE,
            "plus": NUDGED_TOLERANCE,
            "minus": NUDGED_TOLERANCE,
        }
        for phase in ("free", "plus", "minus"):
            phase_rows.append(
                phase_row(
                    seed,
                    phase,
                    budget,
                    snapshots[phase],
                    exact_states[phase],
                    tolerances[phase],
                    candidate,
                    confirmatory_model_seed,
                )
            )
        estimate = flatten_physical(
            centered_physical_gradients(
                np.asarray(snapshots["minus"]["q"]),
                np.asarray(snapshots["plus"]["q"]),
                params,
                features,
                config.beta,
            )
        )
        exact_centered = np.asarray(reference["centered"])
        exact_implicit = np.asarray(reference["implicit"])
        state_errors = [
            relative_error(np.asarray(snapshots[phase]["q"]), exact_states[phase])
            for phase in ("free", "plus", "minus")
        ]
        convergence_fractions = [
            float(np.mean(np.asarray(snapshots[phase]["converged"])))
            for phase in ("free", "plus", "minus")
        ]
        gradient_error = relative_error(estimate, exact_centered)
        gradient_cosine = cosine(estimate, exact_implicit)
        curve_rows.append(
            {
                "seed": seed,
                "confirmatory_model_seed": confirmatory_model_seed,
                "candidate": candidate.key,
                "budget": budget,
                "minimum_phase_convergence_fraction": min(convergence_fractions),
                "all_phases_strictly_converged": bool(
                    all(np.all(np.asarray(snapshots[phase]["converged"])) for phase in snapshots)
                ),
                "maximum_state_relative_error": max(state_errors),
                "gradient_vs_exact_centered_relative_error": gradient_error,
                "gradient_vs_implicit_relative_error": relative_error(estimate, exact_implicit),
                "gradient_vs_implicit_cosine": gradient_cosine,
                "exact_centered_vs_implicit_relative_error": relative_error(
                    exact_centered, exact_implicit
                ),
                "finite_horizon_certificate_pass": bool(
                    max(state_errors) <= STATE_ERROR_THRESHOLD
                    and gradient_error <= GRADIENT_ERROR_THRESHOLD
                    and gradient_cosine >= GRADIENT_COSINE_THRESHOLD
                ),
                "total_active_channel_steps": int(
                    sum(int(snapshots[phase]["active_channel_steps"]) for phase in snapshots)
                ),
            }
        )
    final_curve = curve_rows[-1]
    final_method = {
        "seed": seed,
        "confirmatory_model_seed": confirmatory_model_seed,
        "candidate": candidate.key,
        "audit_samples": features.shape[0],
        "budget": budgets[-1],
        "strict_all_phases_converged": final_curve["all_phases_strictly_converged"],
        "finite_horizon_certificate_pass": final_curve["finite_horizon_certificate_pass"],
        "maximum_state_relative_error": final_curve["maximum_state_relative_error"],
        "gradient_vs_exact_centered_relative_error": final_curve[
            "gradient_vs_exact_centered_relative_error"
        ],
        "gradient_vs_implicit_relative_error": final_curve[
            "gradient_vs_implicit_relative_error"
        ],
        "gradient_vs_implicit_cosine": final_curve["gradient_vs_implicit_cosine"],
        "exact_centered_vs_implicit_relative_error": final_curve[
            "exact_centered_vs_implicit_relative_error"
        ],
        "total_active_channel_steps": final_curve["total_active_channel_steps"],
        "total_wall_time_seconds": float(
            free["wall_time_seconds"] + plus["wall_time_seconds"] + minus["wall_time_seconds"]
        ),
    }
    return phase_rows, curve_rows, final_method


def spectral_candidate_table(
    params: Parameters,
    config: ModelConfig,
    features: np.ndarray,
) -> pd.DataFrame:
    q_free, _, _ = solve_equilibrium(params, features, config)
    hessians = state_hessian(q_free, params, config).reshape(
        -1, config.n_nodes, config.n_nodes
    )
    identity = np.eye(config.n_nodes)
    zero = np.zeros_like(identity)
    rows: list[dict[str, object]] = []
    for candidate in candidate_grid(config.n_nodes):
        damping = np.diag(damping_vector(candidate, config.n_nodes))
        rates = []
        for hessian in hessians:
            state_matrix = np.block([[zero, identity], [-hessian, -damping]])
            spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(state_matrix))))
            rates.append(max(0.0, -spectral_abscissa))
        values = np.asarray(rates)
        rows.append(
            {
                "candidate": candidate.key,
                "width": candidate.width,
                "trace": candidate.trace,
                "power": candidate.power,
                "minimum_spectral_decay_rate": float(np.min(values)),
                "q05_spectral_decay_rate": float(np.quantile(values, 0.05)),
                "median_spectral_decay_rate": float(np.median(values)),
                "maximum_spectral_decay_rate": float(np.max(values)),
                "n_calibration_channels": int(values.size),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["q05_spectral_decay_rate", "median_spectral_decay_rate"],
        ascending=False,
    )


def calibration_selection_key(row: dict[str, object]) -> tuple[float, ...]:
    state = float(row["maximum_state_relative_error"])
    gradient = float(row["gradient_vs_exact_centered_relative_error"])
    cosine_value = float(row["gradient_vs_implicit_cosine"])
    violations = (
        max(0.0, state / STATE_ERROR_THRESHOLD - 1.0)
        + max(0.0, gradient / GRADIENT_ERROR_THRESHOLD - 1.0)
        + max(0.0, (GRADIENT_COSINE_THRESHOLD - cosine_value) * 1000.0)
    )
    return (
        0.0 if bool(row["finite_horizon_certificate_pass"]) else 1.0,
        violations,
        1.0 - float(row["minimum_phase_convergence_fraction"]),
        gradient,
        float(row["total_active_channel_steps"]),
    )


def make_plots(
    spectral: pd.DataFrame,
    calibration: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0))
    spectral_plot = spectral.head(10).iloc[::-1]
    axes[0, 0].barh(spectral_plot["candidate"], spectral_plot["q05_spectral_decay_rate"])
    axes[0, 0].set_xscale("log")
    axes[0, 0].set(title="Calibration spectral shortlist", xlabel="5th-percentile decay rate")
    axes[0, 0].tick_params(axis="y", labelsize=7)

    axes[0, 1].bar(
        calibration["candidate"],
        calibration["gradient_vs_exact_centered_relative_error"],
    )
    axes[0, 1].axhline(GRADIENT_ERROR_THRESHOLD, color="black", linestyle="--", label="registered 1%")
    axes[0, 1].set_yscale("log")
    axes[0, 1].tick_params(axis="x", rotation=25, labelsize=7)
    axes[0, 1].set(title="Calibration gradient fidelity", ylabel="relative error")
    axes[0, 1].legend(fontsize=8)

    for seed, block in curves.groupby("seed"):
        axes[1, 0].plot(
            block["budget"],
            block["gradient_vs_exact_centered_relative_error"],
            marker="o",
            label=f"seed={seed}",
        )
        axes[1, 1].plot(
            block["budget"],
            block["minimum_phase_convergence_fraction"],
            marker="o",
            label=f"seed={seed}",
        )
    axes[1, 0].axhline(GRADIENT_ERROR_THRESHOLD, color="black", linestyle="--")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        title="Boundary solver gradient vs exact-centered",
        xlabel="per-phase step budget",
        ylabel="relative error",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_ylim(0.0, 1.01)
    axes[1, 1].set(
        title="Worst boundary phase convergence",
        xlabel="per-phase step budget",
        ylabel="converged channel fraction",
    )
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "mnist_boundary_calibration_audit_v33.png", dpi=190)
    plt.close(fig)


def run(
    models_dir: Path,
    cache_file: Path,
    output_dir: Path,
    mode: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(models_dir)
    with np.load(cache_file, allow_pickle=False) as cache:
        x_train = np.asarray(cache["x_train"]).copy()
        y_train = np.asarray(cache["y_train"], dtype=np.int64).copy()
        x_test = np.asarray(cache["x_test"]).copy()
        y_test = np.asarray(cache["y_test"], dtype=np.int64).copy()
    calibration_seed = 17 if 17 in models else sorted(models)[0]
    calibration_params, calibration_config = models[calibration_seed]

    if mode == "quick":
        calibration_samples = 8
        audit_samples = 8
        shortlist_size = 2
        calibration_budget = 2_000
        budgets = (500, 1_000, 2_000)
        audit_seeds = (calibration_seed,)
    else:
        required = {17, 29, 43}
        missing = required - set(models)
        if missing:
            raise FileNotFoundError(f"Paper audit requires model seeds {sorted(required)}; missing {sorted(missing)}")
        calibration_samples = 32
        audit_samples = 128
        shortlist_size = 4
        calibration_budget = 60_000
        budgets = (10_000, 30_000, 60_000, 120_000, 240_000)
        audit_seeds = (17, 29, 43)

    rng = np.random.default_rng(730_017)
    calibration_indices = rng.choice(
        x_train.shape[0], size=calibration_samples, replace=False
    )
    x_calibration = x_train[calibration_indices]
    y_calibration = y_train[calibration_indices]
    spectral = spectral_candidate_table(
        calibration_params, calibration_config, x_calibration
    )
    spectral.to_csv(output_dir / "mnist_damping_spectral_candidates_v33.csv", index=False)
    shortlist = spectral.head(shortlist_size)

    calibration_rows: list[dict[str, object]] = []
    for candidate_row in shortlist.to_dict(orient="records"):
        candidate = DampingCandidate(
            int(candidate_row["width"]),
            float(candidate_row["trace"]),
            float(candidate_row["power"]),
        )
        _, curves, method = run_locked_audit(
            calibration_seed,
            calibration_params,
            calibration_config,
            x_calibration,
            y_calibration,
            candidate,
            (calibration_budget,),
            confirmatory_model_seed=False,
        )
        row = dict(method)
        row.update(
            {
                "minimum_phase_convergence_fraction": curves[-1][
                    "minimum_phase_convergence_fraction"
                ],
                "spectral_q05_decay_rate": candidate_row[
                    "q05_spectral_decay_rate"
                ],
                "selection_key": json.dumps(calibration_selection_key(curves[-1])),
            }
        )
        calibration_rows.append(row)
    calibration = pd.DataFrame(calibration_rows)
    selection_order = sorted(
        range(len(calibration_rows)),
        key=lambda index: calibration_selection_key(calibration_rows[index]),
    )
    selected_row = calibration_rows[selection_order[0]]
    selected_parts = {
        part.split("=")[0]: part.split("=")[1]
        for part in str(selected_row["candidate"]).split("|")
    }
    selected = DampingCandidate(
        int(selected_parts["width"]),
        float(selected_parts["trace"]),
        float(selected_parts["power"]),
    )
    calibration["selected"] = calibration["candidate"] == selected.key
    calibration.to_csv(output_dir / "mnist_damping_dynamic_calibration_v33.csv", index=False)
    lock = {
        "version": "3.3",
        "trained_models_unchanged": True,
        "calibration_model_seed": calibration_seed,
        "calibration_data_source": "training pool only",
        "calibration_indices": calibration_indices.tolist(),
        "selected_candidate": asdict(selected),
        "selected_candidate_key": selected.key,
        "selection_thresholds": {
            "maximum_state_relative_error": STATE_ERROR_THRESHOLD,
            "maximum_gradient_vs_exact_centered_relative_error": GRADIENT_ERROR_THRESHOLD,
            "minimum_gradient_cosine": GRADIENT_COSINE_THRESHOLD,
        },
        "locked_before_test_audit": True,
    }
    (output_dir / "mnist_damping_lock_v33.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )

    phases_path = output_dir / "mnist_boundary_phases_v33.csv"
    curves_path = output_dir / "mnist_boundary_budget_curves_v33.csv"
    methods_path = output_dir / "mnist_boundary_methods_v33.csv"
    phase_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    if methods_path.exists() and phases_path.exists() and curves_path.exists():
        existing_methods = pd.read_csv(methods_path)
        if (
            not existing_methods.empty
            and set(existing_methods["candidate"].astype(str)) == {selected.key}
            and set(existing_methods["budget"].astype(int)) == {budgets[-1]}
        ):
            method_rows = existing_methods.to_dict(orient="records")
            phase_rows = pd.read_csv(phases_path).to_dict(orient="records")
            curve_rows = pd.read_csv(curves_path).to_dict(orient="records")
    completed_seeds = {int(row["seed"]) for row in method_rows}
    for seed in audit_seeds:
        if seed in completed_seeds:
            print(f"seed={seed} locked audit already completed; skipping")
            continue
        params, config = models[seed]
        audit_rng = np.random.default_rng(seed + 900_000)
        audit_indices = audit_rng.choice(
            x_test.shape[0], size=audit_samples, replace=False
        )
        phases, curves, method = run_locked_audit(
            seed,
            params,
            config,
            x_test[audit_indices],
            y_test[audit_indices],
            selected,
            budgets,
            confirmatory_model_seed=seed in (29, 43),
        )
        for row in phases:
            row["audit_indices_json"] = json.dumps(audit_indices.tolist())
        phase_rows.extend(phases)
        curve_rows.extend(curves)
        method_rows.append(method)
        # The method row is the per-seed completion marker and is written last.
        pd.DataFrame(phase_rows).to_csv(phases_path, index=False)
        pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
        pd.DataFrame(method_rows).to_csv(methods_path, index=False)

    phases_df = pd.DataFrame(phase_rows)
    curves_df = pd.DataFrame(curve_rows)
    methods_df = pd.DataFrame(method_rows)
    make_plots(spectral, calibration, curves_df, output_dir)

    confirmatory = methods_df[methods_df["confirmatory_model_seed"]]
    summary = {
        "version": "3.3-mnist-boundary-calibration",
        "mode": mode,
        "trained_models_unchanged": True,
        "selected_candidate": asdict(selected),
        "selected_candidate_key": selected.key,
        "calibration_finite_horizon_pass": bool(selected_row["finite_horizon_certificate_pass"]),
        "audit_budgets": list(budgets),
        "n_model_seeds": int(methods_df.shape[0]),
        "confirmatory_model_seeds": confirmatory["seed"].astype(int).tolist(),
        "all_seed_strict_pass": bool(methods_df["strict_all_phases_converged"].all()),
        "all_seed_finite_horizon_pass": bool(methods_df["finite_horizon_certificate_pass"].all()),
        "confirmatory_strict_pass": bool(confirmatory["strict_all_phases_converged"].all()) if not confirmatory.empty else None,
        "confirmatory_finite_horizon_pass": bool(confirmatory["finite_horizon_certificate_pass"].all()) if not confirmatory.empty else None,
        "maximum_state_relative_error": float(methods_df["maximum_state_relative_error"].max()),
        "maximum_gradient_vs_exact_centered_relative_error": float(methods_df["gradient_vs_exact_centered_relative_error"].max()),
        "minimum_gradient_cosine": float(methods_df["gradient_vs_implicit_cosine"].min()),
        "median_wall_time_seconds": float(methods_df["total_wall_time_seconds"].median()),
        "registered_thresholds": {
            "free_tolerance": FREE_TOLERANCE,
            "nudged_tolerance": NUDGED_TOLERANCE,
            "maximum_state_relative_error": STATE_ERROR_THRESHOLD,
            "maximum_gradient_vs_exact_centered_relative_error": GRADIENT_ERROR_THRESHOLD,
            "minimum_gradient_cosine": GRADIENT_COSINE_THRESHOLD,
        },
        "interpretation": "Calibration uses training-pool samples only. The selected boundary profile is locked before test audit. Seeds 29 and 43 are confirmatory model seeds.",
    }
    (output_dir / "mnist_boundary_summary_v33.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir", type=Path, default=Path("results_v3_mnist")
    )
    parser.add_argument(
        "--cache-file", type=Path, default=Path("mnist_cache/mnist_pca_32.npz")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_v3_mnist_boundary_v33")
    )
    parser.add_argument("--mode", choices=("quick", "paper"), default="paper")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.models_dir, arguments.cache_file, arguments.output_dir, arguments.mode)
