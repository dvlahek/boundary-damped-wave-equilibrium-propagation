#!/usr/bin/env python3
"""Locked-tolerance MNIST audit for radiative equilibrium propagation V3.4.

V3.4 never retrains the MNIST models and never recalibrates the boundary
damping profile.  It reads the profile that V3.3 locked before its test audit,
calibrates only solver tolerances on fixed MNIST training-pool samples, writes
a second lock, and then evaluates fixed official-test samples.  Seeds 29 and 43
are confirmatory model seeds; seed 17 is calibration-support evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-mnist-v34")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnist_boundary_calibration_v33 import (
    CHECK_EVERY,
    CONSECUTIVE_CHECKS,
    DT,
    DampingCandidate,
    cosine,
    damping_vector,
    exact_references,
    load_models,
    pair_gradient,
    relative_error,
)
from mnist_rep_benchmark import (
    ModelConfig,
    Parameters,
    centered_physical_gradients,
    flatten_physical,
    state_gradient,
)


STATE_ERROR_THRESHOLD = 1e-4
GRADIENT_ERROR_THRESHOLD = 1e-2
GRADIENT_COSINE_THRESHOLD = 0.999
GRADIENT_STABILITY_THRESHOLD = 2e-3


@dataclass(frozen=True)
class ToleranceCandidate:
    free: float
    nudged: float

    @property
    def key(self) -> str:
        return f"free={self.free:.0e}|nudged={self.nudged:.0e}"


def tolerance_grid() -> list[ToleranceCandidate]:
    """Registered calibration grid, from the V3.3 control to 20x tighter."""
    return [
        ToleranceCandidate(1e-4, 2e-5),
        ToleranceCandidate(1e-4, 1e-5),
        ToleranceCandidate(5e-5, 5e-6),
        ToleranceCandidate(2e-5, 2e-6),
        ToleranceCandidate(1e-5, 1e-6),
    ]


def load_locked_damping(path: Path) -> tuple[DampingCandidate, dict[str, object], str]:
    if not path.exists():
        raise FileNotFoundError(
            f"V3.3 damping lock was not found: {path}. Run V3.3 first or copy its results folder."
        )
    raw = path.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    if not bool(lock.get("locked_before_test_audit", False)):
        raise ValueError("The supplied V3.3 damping profile was not locked before test audit")
    selected = lock.get("selected_candidate")
    if not isinstance(selected, dict):
        raise ValueError("The V3.3 lock does not contain selected_candidate")
    candidate = DampingCandidate(
        int(selected["width"]), float(selected["trace"]), float(selected["power"])
    )
    if lock.get("selected_candidate_key") not in (None, candidate.key):
        raise ValueError("The V3.3 candidate key does not match selected_candidate")
    return candidate, lock, hashlib.sha256(raw).hexdigest()


def load_v33_audit_indices(path: Path) -> tuple[dict[int, set[int]], str]:
    """Read only the sample identities needed to form a disjoint V3.4 subset."""
    if not path.exists():
        raise FileNotFoundError(
            f"V3.3 phase results were not found: {path}. They are required for a fresh V3.4 test subset."
        )
    raw = path.read_bytes()
    table = pd.read_csv(path, usecols=["seed", "audit_indices_json"])
    excluded: dict[int, set[int]] = {}
    for seed, block in table.groupby("seed"):
        parsed = {
            tuple(int(value) for value in json.loads(str(text)))
            for text in block["audit_indices_json"].dropna().unique()
        }
        if len(parsed) != 1:
            raise ValueError(f"V3.3 seed {seed} does not have one consistent audit subset")
        excluded[int(seed)] = set(next(iter(parsed)))
    return excluded, hashlib.sha256(raw).hexdigest()


def gradient_change(current: np.ndarray, previous: np.ndarray) -> float:
    return float(
        np.linalg.norm(current - previous) / max(np.linalg.norm(current), 1e-14)
    )


def relax_phase(
    params: Parameters,
    features: np.ndarray,
    config: ModelConfig,
    candidate: DampingCandidate,
    budgets: tuple[int, ...],
    tolerance: float,
    *,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial: np.ndarray | None = None,
) -> dict[str, object]:
    """Continue one physical trajectory and freeze channels individually."""
    budgets = tuple(sorted(set(int(value) for value in budgets)))
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
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
                    locally_converged, flat_consecutive[active_flat] + 1, 0
                )
                newly_converged = active_flat[
                    flat_consecutive[active_flat] >= CONSECUTIVE_CHECKS
                ]
                converged.ravel()[newly_converged] = True

        if step in budgets:
            final_gradient = state_gradient(q, params, features, config, targets, beta)
            snapshots[step] = {
                "q": q.copy(),
                "converged": converged.copy(),
                "residuals": np.linalg.norm(final_gradient, axis=2),
                "speeds": np.linalg.norm(velocity, axis=2),
                "active_channel_steps": int(active_channel_steps),
                "channel_steps": channel_steps.copy(),
                "wall_time_seconds": time.perf_counter() - start_time,
            }

        if not active_flat.size:
            final_gradient = state_gradient(q, params, features, config, targets, beta)
            for budget in budgets:
                if budget > step and budget not in snapshots:
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


def phase_row(
    seed: int,
    phase: str,
    budget: int,
    snapshot: dict[str, object],
    exact_state: np.ndarray,
    tolerance: float,
    candidate: DampingCandidate,
    tolerance_candidate: ToleranceCandidate,
    confirmatory_model_seed: bool,
) -> dict[str, object]:
    converged = np.asarray(snapshot["converged"], dtype=bool)
    residuals = np.asarray(snapshot["residuals"])
    speeds = np.asarray(snapshot["speeds"])
    channel_steps = np.asarray(snapshot["channel_steps"])
    return {
        "seed": seed,
        "confirmatory_model_seed": confirmatory_model_seed,
        "damping_candidate": candidate.key,
        "tolerance_candidate": tolerance_candidate.key,
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
        "phase_tolerance": tolerance,
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
    damping_candidate: DampingCandidate,
    tolerance_candidate: ToleranceCandidate,
    budgets: tuple[int, ...],
    *,
    confirmatory_model_seed: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    reference = exact_references(params, features, labels, config)
    targets = np.asarray(reference["targets"])
    free = relax_phase(
        params,
        features,
        config,
        damping_candidate,
        budgets,
        tolerance_candidate.free,
        beta=0.0,
    )
    plus = relax_phase(
        params,
        features,
        config,
        damping_candidate,
        budgets,
        tolerance_candidate.nudged,
        targets=targets,
        beta=config.beta,
        initial=np.asarray(free["final_q"]),
    )
    minus = relax_phase(
        params,
        features,
        config,
        damping_candidate,
        budgets,
        tolerance_candidate.nudged,
        targets=targets,
        beta=-config.beta,
        initial=np.asarray(free["final_q"]),
    )

    phase_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    previous_estimate: np.ndarray | None = None
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
            "free": tolerance_candidate.free,
            "plus": tolerance_candidate.nudged,
            "minus": tolerance_candidate.nudged,
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
                    damping_candidate,
                    tolerance_candidate,
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
        stability = (
            float("nan")
            if previous_estimate is None
            else gradient_change(estimate, previous_estimate)
        )
        stability_pass = bool(
            np.isfinite(stability) and stability <= GRADIENT_STABILITY_THRESHOLD
        )
        certificate_pass = bool(
            max(state_errors) <= STATE_ERROR_THRESHOLD
            and gradient_error <= GRADIENT_ERROR_THRESHOLD
            and gradient_cosine >= GRADIENT_COSINE_THRESHOLD
            and stability_pass
        )
        curve_rows.append(
            {
                "seed": seed,
                "confirmatory_model_seed": confirmatory_model_seed,
                "damping_candidate": damping_candidate.key,
                "tolerance_candidate": tolerance_candidate.key,
                "free_tolerance": tolerance_candidate.free,
                "nudged_tolerance": tolerance_candidate.nudged,
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
                "gradient_change_from_previous_budget": stability,
                "gradient_stability_pass": stability_pass,
                "finite_horizon_certificate_pass": certificate_pass,
                "total_active_channel_steps": int(
                    sum(int(snapshots[phase]["active_channel_steps"]) for phase in snapshots)
                ),
            }
        )
        previous_estimate = estimate.copy()

    final_curve = curve_rows[-1]
    final_method = {
        "seed": seed,
        "confirmatory_model_seed": confirmatory_model_seed,
        "damping_candidate": damping_candidate.key,
        "tolerance_candidate": tolerance_candidate.key,
        "free_tolerance": tolerance_candidate.free,
        "nudged_tolerance": tolerance_candidate.nudged,
        "audit_samples": features.shape[0],
        "budget": budgets[-1],
        "strict_all_phases_converged": final_curve["all_phases_strictly_converged"],
        "gradient_stability_pass": final_curve["gradient_stability_pass"],
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
        "gradient_change_from_previous_budget": final_curve[
            "gradient_change_from_previous_budget"
        ],
        "minimum_phase_convergence_fraction": final_curve[
            "minimum_phase_convergence_fraction"
        ],
        "total_active_channel_steps": final_curve["total_active_channel_steps"],
        "total_wall_time_seconds": float(
            free["wall_time_seconds"] + plus["wall_time_seconds"] + minus["wall_time_seconds"]
        ),
    }
    return phase_rows, curve_rows, final_method


def calibration_selection_key(row: dict[str, object]) -> tuple[float, ...]:
    state = float(row["maximum_state_relative_error"])
    gradient = float(row["gradient_vs_exact_centered_relative_error"])
    cosine_value = float(row["gradient_vs_implicit_cosine"])
    stability = float(row["gradient_change_from_previous_budget"])
    violations = (
        max(0.0, state / STATE_ERROR_THRESHOLD - 1.0)
        + max(0.0, gradient / GRADIENT_ERROR_THRESHOLD - 1.0)
        + max(0.0, (GRADIENT_COSINE_THRESHOLD - cosine_value) * 1000.0)
        + max(0.0, stability / GRADIENT_STABILITY_THRESHOLD - 1.0)
    )
    return (
        0.0 if bool(row["finite_horizon_certificate_pass"]) else 1.0,
        violations,
        0.0 if bool(row["strict_all_phases_converged"]) else 1.0,
        1.0 - float(row["minimum_phase_convergence_fraction"]),
        gradient,
        float(row["total_active_channel_steps"]),
    )


def make_plots(
    calibration: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0))
    axes[0, 0].bar(
        calibration["tolerance_candidate"],
        calibration["gradient_vs_exact_centered_relative_error"],
    )
    axes[0, 0].axhline(
        GRADIENT_ERROR_THRESHOLD, color="black", linestyle="--", label="registered 1%"
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].tick_params(axis="x", rotation=25, labelsize=7)
    axes[0, 0].set(title="Training-pool tolerance calibration", ylabel="relative gradient error")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(
        calibration["tolerance_candidate"], calibration["total_active_channel_steps"]
    )
    axes[0, 1].tick_params(axis="x", rotation=25, labelsize=7)
    axes[0, 1].set(title="Calibration work", ylabel="active channel steps")

    for seed, block in curves.groupby("seed"):
        axes[1, 0].plot(
            block["budget"],
            block["gradient_vs_exact_centered_relative_error"],
            marker="o",
            label=f"seed={seed}",
        )
        stable = block[np.isfinite(block["gradient_change_from_previous_budget"])]
        axes[1, 1].plot(
            stable["budget"],
            stable["gradient_change_from_previous_budget"],
            marker="o",
            label=f"seed={seed}",
        )
    axes[1, 0].axhline(GRADIENT_ERROR_THRESHOLD, color="black", linestyle="--")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        title="Locked boundary gradient vs exact-centered",
        xlabel="per-phase step budget",
        ylabel="relative error",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].axhline(
        GRADIENT_STABILITY_THRESHOLD, color="black", linestyle="--", label="registered 0.2%"
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        title="Physically measurable gradient stability",
        xlabel="per-phase step budget",
        ylabel="relative change from previous budget",
    )
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "mnist_boundary_tolerance_audit_v34.png", dpi=190)
    plt.close(fig)


def run(
    models_dir: Path,
    cache_file: Path,
    v33_lock_file: Path,
    v33_phases_file: Path,
    output_dir: Path,
    mode: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(models_dir)
    damping_candidate, v33_lock, v33_lock_sha256 = load_locked_damping(v33_lock_file)
    with np.load(cache_file, allow_pickle=False) as cache:
        x_train = np.asarray(cache["x_train"]).copy()
        y_train = np.asarray(cache["y_train"], dtype=np.int64).copy()
        x_test = np.asarray(cache["x_test"]).copy()
        y_test = np.asarray(cache["y_test"], dtype=np.int64).copy()
    calibration_seed = 17 if 17 in models else sorted(models)[0]
    calibration_params, calibration_config = models[calibration_seed]

    if mode == "quick":
        calibration_samples = 6
        audit_samples = 6
        calibration_budgets = (500, 1_000, 2_000)
        audit_budgets = (500, 1_000, 2_000)
        profiles = tolerance_grid()[:2]
        audit_seeds = (calibration_seed,)
        v33_audit_indices: dict[int, set[int]] = {}
        v33_phases_sha256 = None
    else:
        required = {17, 29, 43}
        missing = required - set(models)
        if missing:
            raise FileNotFoundError(
                f"Paper audit requires model seeds {sorted(required)}; missing {sorted(missing)}"
            )
        calibration_samples = 32
        audit_samples = 128
        calibration_budgets = (30_000, 60_000, 120_000, 240_000)
        audit_budgets = (30_000, 60_000, 120_000, 240_000, 480_000)
        profiles = tolerance_grid()
        audit_seeds = (17, 29, 43)
        v33_audit_indices, v33_phases_sha256 = load_v33_audit_indices(v33_phases_file)
        missing_index_seeds = required - set(v33_audit_indices)
        if missing_index_seeds:
            raise ValueError(
                f"V3.3 phase results are missing audit indices for seeds {sorted(missing_index_seeds)}"
            )

    rng = np.random.default_rng(740_017)
    v33_calibration_indices = {
        int(value) for value in v33_lock.get("calibration_indices", [])
    }
    available_calibration_indices = np.asarray(
        sorted(set(range(x_train.shape[0])) - v33_calibration_indices), dtype=np.int64
    )
    if available_calibration_indices.size < calibration_samples:
        raise ValueError("Not enough training-pool samples remain for V3.4 calibration")
    calibration_indices = rng.choice(
        available_calibration_indices, size=calibration_samples, replace=False
    )
    x_calibration = x_train[calibration_indices]
    y_calibration = y_train[calibration_indices]

    calibration_methods_path = output_dir / "mnist_tolerance_calibration_v34.csv"
    calibration_curves_path = output_dir / "mnist_tolerance_calibration_curves_v34.csv"
    calibration_rows: list[dict[str, object]] = []
    calibration_curve_rows: list[dict[str, object]] = []
    if calibration_methods_path.exists() and calibration_curves_path.exists():
        old_methods = pd.read_csv(calibration_methods_path)
        old_curves = pd.read_csv(calibration_curves_path)
        if (
            not old_methods.empty
            and set(old_methods["damping_candidate"].astype(str)) == {damping_candidate.key}
            and set(old_methods["budget"].astype(int)) == {calibration_budgets[-1]}
        ):
            calibration_rows = old_methods.to_dict(orient="records")
            calibration_curve_rows = old_curves.to_dict(orient="records")
    completed_profiles = {str(row["tolerance_candidate"]) for row in calibration_rows}
    for profile in profiles:
        if profile.key in completed_profiles:
            print(f"calibration {profile.key} already completed; skipping")
            continue
        _, curves, method = run_locked_audit(
            calibration_seed,
            calibration_params,
            calibration_config,
            x_calibration,
            y_calibration,
            damping_candidate,
            profile,
            calibration_budgets,
            confirmatory_model_seed=False,
        )
        calibration_curve_rows.extend(curves)
        method["selection_key"] = json.dumps(calibration_selection_key(method))
        calibration_rows.append(method)
        pd.DataFrame(calibration_curve_rows).to_csv(calibration_curves_path, index=False)
        pd.DataFrame(calibration_rows).to_csv(calibration_methods_path, index=False)

    eligible_rows = [
        row for row in calibration_rows if str(row["tolerance_candidate"]) in {p.key for p in profiles}
    ]
    selected_row = min(eligible_rows, key=calibration_selection_key)
    selected_profile = next(
        profile for profile in profiles if profile.key == selected_row["tolerance_candidate"]
    )
    calibration = pd.DataFrame(eligible_rows)
    calibration["selected"] = calibration["tolerance_candidate"] == selected_profile.key
    calibration.to_csv(calibration_methods_path, index=False)

    tolerance_lock = {
        "version": "3.4",
        "trained_models_unchanged": True,
        "damping_profile_recalibrated": False,
        "source_v33_lock_file": v33_lock_file.name,
        "source_v33_lock_sha256": v33_lock_sha256,
        "source_v33_phases_file": v33_phases_file.name if mode == "paper" else None,
        "source_v33_phases_sha256": v33_phases_sha256,
        "locked_damping_candidate": asdict(damping_candidate),
        "locked_damping_candidate_key": damping_candidate.key,
        "calibration_model_seed": calibration_seed,
        "calibration_data_source": "training pool only",
        "calibration_indices": calibration_indices.tolist(),
        "v33_excluded_calibration_indices": sorted(v33_calibration_indices),
        "calibration_subset_disjoint_from_v33": not bool(
            v33_calibration_indices.intersection(calibration_indices.tolist())
        ),
        "tolerance_grid": [asdict(profile) | {"key": profile.key} for profile in profiles],
        "selected_tolerance": asdict(selected_profile),
        "selected_tolerance_key": selected_profile.key,
        "selection_thresholds": {
            "maximum_state_relative_error": STATE_ERROR_THRESHOLD,
            "maximum_gradient_vs_exact_centered_relative_error": GRADIENT_ERROR_THRESHOLD,
            "minimum_gradient_cosine": GRADIENT_COSINE_THRESHOLD,
            "maximum_gradient_change_from_previous_budget": GRADIENT_STABILITY_THRESHOLD,
        },
        "locked_before_v34_test_audit": True,
        "v34_test_samples_exclude_all_v33_audit_indices": mode == "paper",
        "v33_lock_declared_locked_before_test_audit": bool(
            v33_lock["locked_before_test_audit"]
        ),
    }
    (output_dir / "mnist_tolerance_lock_v34.json").write_text(
        json.dumps(tolerance_lock, indent=2), encoding="utf-8"
    )

    phases_path = output_dir / "mnist_boundary_phases_v34.csv"
    curves_path = output_dir / "mnist_boundary_budget_curves_v34.csv"
    methods_path = output_dir / "mnist_boundary_methods_v34.csv"
    phase_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    if methods_path.exists() and phases_path.exists() and curves_path.exists():
        existing_methods = pd.read_csv(methods_path)
        if (
            not existing_methods.empty
            and set(existing_methods["damping_candidate"].astype(str)) == {damping_candidate.key}
            and set(existing_methods["tolerance_candidate"].astype(str)) == {selected_profile.key}
            and set(existing_methods["budget"].astype(int)) == {audit_budgets[-1]}
        ):
            method_rows = existing_methods.to_dict(orient="records")
            phase_rows = pd.read_csv(phases_path).to_dict(orient="records")
            curve_rows = pd.read_csv(curves_path).to_dict(orient="records")
    completed_seeds = {int(row["seed"]) for row in method_rows}
    for seed in audit_seeds:
        if seed in completed_seeds:
            print(f"seed={seed} locked V3.4 audit already completed; skipping")
            continue
        params, config = models[seed]
        audit_rng = np.random.default_rng(seed + 1_100_000)
        excluded_indices = v33_audit_indices.get(seed, set())
        available_indices = np.asarray(
            sorted(set(range(x_test.shape[0])) - excluded_indices), dtype=np.int64
        )
        if available_indices.size < audit_samples:
            raise ValueError(f"Not enough fresh test samples remain for seed {seed}")
        audit_indices = audit_rng.choice(
            available_indices, size=audit_samples, replace=False
        )
        if excluded_indices.intersection(audit_indices.tolist()):
            raise RuntimeError(f"V3.4 seed {seed} overlaps the V3.3 audit subset")
        phases, curves, method = run_locked_audit(
            seed,
            params,
            config,
            x_test[audit_indices],
            y_test[audit_indices],
            damping_candidate,
            selected_profile,
            audit_budgets,
            confirmatory_model_seed=seed in (29, 43),
        )
        for row in phases:
            row["audit_indices_json"] = json.dumps(audit_indices.tolist())
            row["v33_excluded_audit_indices_json"] = json.dumps(
                sorted(excluded_indices)
            )
            row["fresh_test_subset_disjoint_from_v33"] = mode == "paper"
        method["n_v33_excluded_test_samples"] = len(excluded_indices)
        method["fresh_test_subset_disjoint_from_v33"] = mode == "paper"
        phase_rows.extend(phases)
        curve_rows.extend(curves)
        method_rows.append(method)
        pd.DataFrame(phase_rows).to_csv(phases_path, index=False)
        pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
        pd.DataFrame(method_rows).to_csv(methods_path, index=False)

    curves_df = pd.DataFrame(curve_rows)
    methods_df = pd.DataFrame(method_rows)
    make_plots(calibration, curves_df, output_dir)
    confirmatory = methods_df[methods_df["confirmatory_model_seed"].astype(bool)]
    summary = {
        "version": "3.4-mnist-locked-tolerance-audit",
        "mode": mode,
        "trained_models_unchanged": True,
        "damping_profile_recalibrated": False,
        "locked_damping_candidate": asdict(damping_candidate),
        "locked_damping_candidate_key": damping_candidate.key,
        "source_v33_lock_sha256": v33_lock_sha256,
        "source_v33_phases_sha256": v33_phases_sha256,
        "selected_tolerance": asdict(selected_profile),
        "selected_tolerance_key": selected_profile.key,
        "calibration_finite_horizon_pass": bool(
            selected_row["finite_horizon_certificate_pass"]
        ),
        "calibration_gradient_stability_pass": bool(selected_row["gradient_stability_pass"]),
        "audit_budgets": list(audit_budgets),
        "n_model_seeds": int(methods_df.shape[0]),
        "confirmatory_model_seeds": confirmatory["seed"].astype(int).tolist(),
        "fresh_test_subsets_disjoint_from_v33": bool(
            methods_df["fresh_test_subset_disjoint_from_v33"].all()
        ),
        "all_seed_strict_pass": bool(methods_df["strict_all_phases_converged"].all()),
        "all_seed_finite_horizon_pass": bool(
            methods_df["finite_horizon_certificate_pass"].all()
        ),
        "confirmatory_strict_pass": bool(
            confirmatory["strict_all_phases_converged"].all()
        ) if not confirmatory.empty else None,
        "confirmatory_finite_horizon_pass": bool(
            confirmatory["finite_horizon_certificate_pass"].all()
        ) if not confirmatory.empty else None,
        "maximum_state_relative_error": float(methods_df["maximum_state_relative_error"].max()),
        "maximum_gradient_vs_exact_centered_relative_error": float(
            methods_df["gradient_vs_exact_centered_relative_error"].max()
        ),
        "minimum_gradient_cosine": float(methods_df["gradient_vs_implicit_cosine"].min()),
        "maximum_final_gradient_change": float(
            methods_df["gradient_change_from_previous_budget"].max()
        ),
        "median_wall_time_seconds": float(methods_df["total_wall_time_seconds"].median()),
        "registered_thresholds": {
            "maximum_state_relative_error": STATE_ERROR_THRESHOLD,
            "maximum_gradient_vs_exact_centered_relative_error": GRADIENT_ERROR_THRESHOLD,
            "minimum_gradient_cosine": GRADIENT_COSINE_THRESHOLD,
            "maximum_gradient_change_from_previous_budget": GRADIENT_STABILITY_THRESHOLD,
        },
        "interpretation": (
            "V3.3 damping is unchanged. Tolerances are calibrated on training-pool samples "
            "and locked before official-test audit. Seeds 29 and 43 are confirmatory."
        ),
    }
    (output_dir / "mnist_boundary_summary_v34.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("results_v3_mnist"))
    parser.add_argument(
        "--cache-file", type=Path, default=Path("mnist_cache/mnist_pca_32.npz")
    )
    parser.add_argument(
        "--v33-lock",
        type=Path,
        default=Path("results_v3_mnist_boundary_v33/mnist_damping_lock_v33.json"),
    )
    parser.add_argument(
        "--v33-phases",
        type=Path,
        default=Path("results_v3_mnist_boundary_v33/mnist_boundary_phases_v33.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_v3_mnist_boundary_v34")
    )
    parser.add_argument("--mode", choices=("quick", "paper"), default="paper")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.models_dir,
        arguments.cache_file,
        arguments.v33_lock,
        arguments.v33_phases,
        arguments.output_dir,
        arguments.mode,
    )
