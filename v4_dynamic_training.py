#!/usr/bin/env python3
"""End-to-end image training whose free and nudged phases come from wave dynamics."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_rep_learning import Adam
from mnist_rep_benchmark import (
    ModelConfig,
    Parameters,
    centered_physical_gradients,
    evaluate,
    initialize_parameters,
    save_model,
    simplex_targets,
)
from mnist_tolerance_audit_v34 import relax_phase
from v4_locked_replication import load_gold_lock
from v4_train_image_models import load_image_dataset, parse_seeds


@dataclass(frozen=True)
class DynamicProfile:
    n_nodes: int
    pca_components: int
    epochs: int
    batches_per_epoch: int
    batch_size: int
    maximum_phase_steps: int
    evaluation_chunk_size: int
    learning_rate_structure: float
    learning_rate_u: float
    gradient_clip_norm: float


PROFILES = {
    "quick": DynamicProfile(9, 16, 2, 4, 32, 2_000, 500, 1e-4, 1e-2, 5.0),
    "paper": DynamicProfile(17, 32, 12, 20, 128, 60_000, 1_000, 1e-4, 1e-2, 5.0),
}


def make_config(profile: DynamicProfile, n_features: int) -> ModelConfig:
    return ModelConfig(
        n_nodes=profile.n_nodes,
        n_features=n_features,
        boundary_width=max(1, round(profile.n_nodes / 3)),
    )


def clip_gradients(
    gradients: tuple[np.ndarray, np.ndarray, np.ndarray], maximum_norm: float
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    norm = float(np.sqrt(sum(float(np.sum(value**2)) for value in gradients)))
    scale = min(1.0, maximum_norm / max(norm, 1e-14))
    return tuple(value * scale for value in gradients), norm


def save_checkpoint(
    path: Path,
    epoch: int,
    params: Parameters,
    optimizers: tuple[Adam, Adam, Adam],
) -> None:
    opt_a, opt_w, opt_u = optimizers
    np.savez_compressed(
        path,
        epoch=epoch,
        log_a=params.log_a,
        log_w=params.log_w,
        u=params.u,
        a_m=opt_a.m,
        a_v=opt_a.v,
        a_step=opt_a.step_number,
        w_m=opt_w.m,
        w_v=opt_w.v,
        w_step=opt_w.step_number,
        u_m=opt_u.m,
        u_v=opt_u.v,
        u_step=opt_u.step_number,
    )


def load_checkpoint(
    path: Path,
    profile: DynamicProfile,
) -> tuple[int, Parameters, tuple[Adam, Adam, Adam]]:
    with np.load(path, allow_pickle=False) as stored:
        params = Parameters(
            np.asarray(stored["log_a"]).copy(),
            np.asarray(stored["log_w"]).copy(),
            np.asarray(stored["u"]).copy(),
        )
        optimizers = (
            Adam(params.log_a.shape, profile.learning_rate_structure),
            Adam(params.log_w.shape, profile.learning_rate_structure),
            Adam(params.u.shape, profile.learning_rate_u),
        )
        for optimizer, prefix in zip(optimizers, ("a", "w", "u")):
            optimizer.m = np.asarray(stored[f"{prefix}_m"]).copy()
            optimizer.v = np.asarray(stored[f"{prefix}_v"]).copy()
            optimizer.step_number = int(stored[f"{prefix}_step"])
        epoch = int(stored["epoch"])
    return epoch, params, optimizers


def train_seed_dynamic(
    seed: int,
    profile: DynamicProfile,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    damping,
    tolerance,
    output_dir: Path,
) -> tuple[Parameters, list[dict[str, object]], dict[str, object]]:
    config = make_config(profile, x_train.shape[1])
    checkpoint = output_dir / f"dynamic_checkpoint_seed_{seed}.npz"
    history_path = output_dir / f"dynamic_history_seed_{seed}.csv"
    if checkpoint.exists():
        completed_epoch, params, optimizers = load_checkpoint(checkpoint, profile)
        history = pd.read_csv(history_path).to_dict(orient="records") if history_path.exists() else []
        print(f"seed={seed} resuming after epoch={completed_epoch}")
    else:
        completed_epoch = 0
        params = initialize_parameters(config, seed)
        optimizers = (
            Adam(params.log_a.shape, profile.learning_rate_structure),
            Adam(params.log_w.shape, profile.learning_rate_structure),
            Adam(params.u.shape, profile.learning_rate_u),
        )
        history = []
    opt_a, opt_w, opt_u = optimizers
    start_time = time.perf_counter()
    for epoch in range(completed_epoch + 1, profile.epochs + 1):
        epoch_rng = np.random.default_rng(seed * 1_000_003 + epoch * 10_007)
        convergence_values: list[float] = []
        active_steps: list[int] = []
        gradient_norms: list[float] = []
        for batch in range(profile.batches_per_epoch):
            indices = epoch_rng.integers(0, x_train.shape[0], size=profile.batch_size)
            features = x_train[indices]
            targets = simplex_targets(y_train[indices], config.n_classes)
            budgets = (profile.maximum_phase_steps,)
            free = relax_phase(
                params,
                features,
                config,
                damping,
                budgets,
                tolerance.free,
                beta=0.0,
            )
            plus = relax_phase(
                params,
                features,
                config,
                damping,
                budgets,
                tolerance.nudged,
                targets=targets,
                beta=config.beta,
                initial=np.asarray(free["final_q"]),
            )
            minus = relax_phase(
                params,
                features,
                config,
                damping,
                budgets,
                tolerance.nudged,
                targets=targets,
                beta=-config.beta,
                initial=np.asarray(free["final_q"]),
            )
            snapshot_free = free["snapshots"][profile.maximum_phase_steps]
            snapshot_plus = plus["snapshots"][profile.maximum_phase_steps]
            snapshot_minus = minus["snapshots"][profile.maximum_phase_steps]
            gradients = centered_physical_gradients(
                np.asarray(snapshot_minus["q"]),
                np.asarray(snapshot_plus["q"]),
                params,
                features,
                config.beta,
            )
            gradients = (gradients[0], gradients[1], gradients[2] + 2e-4 * params.u)
            gradients, raw_norm = clip_gradients(gradients, profile.gradient_clip_norm)
            params.log_a = opt_a.update(params.log_a, gradients[0])
            params.log_w = opt_w.update(params.log_w, gradients[1])
            params.u = opt_u.update(params.u, gradients[2])
            if not all(np.isfinite(value).all() for value in (params.log_a, params.log_w, params.u)):
                raise FloatingPointError(f"Non-finite parameters at seed={seed}, epoch={epoch}, batch={batch}")
            convergence_values.append(
                min(
                    float(np.mean(snapshot_free["converged"])),
                    float(np.mean(snapshot_plus["converged"])),
                    float(np.mean(snapshot_minus["converged"])),
                )
            )
            active_steps.append(
                int(snapshot_free["active_channel_steps"])
                + int(snapshot_plus["active_channel_steps"])
                + int(snapshot_minus["active_channel_steps"])
            )
            gradient_norms.append(raw_norm)
            print(
                f"dynamic seed={seed} epoch={epoch}/{profile.epochs} "
                f"batch={batch + 1}/{profile.batches_per_epoch} "
                f"min_convergence={convergence_values[-1]:.4f}"
            )
        test_count = min(2_000, x_test.shape[0])
        test_metrics = evaluate(
            params,
            x_test[:test_count],
            y_test[:test_count],
            config,
            profile.evaluation_chunk_size,
        )
        row = {
            "seed": seed,
            "epoch": epoch,
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "minimum_batch_phase_convergence_fraction": min(convergence_values),
            "mean_batch_phase_convergence_fraction": float(np.mean(convergence_values)),
            "mean_active_channel_steps_per_batch": float(np.mean(active_steps)),
            "maximum_raw_gradient_norm": max(gradient_norms),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        save_checkpoint(checkpoint, epoch, params, optimizers)
        save_model(output_dir / f"mnist_model_seed_{seed}.npz", params, config)
        print(f"dynamic seed={seed} epoch={epoch} test_accuracy={test_metrics['accuracy']:.4f}")
    final_train = evaluate(params, x_train, y_train, config, profile.evaluation_chunk_size)
    final_test = evaluate(params, x_test, y_test, config, profile.evaluation_chunk_size)
    result = {
        "seed": seed,
        "training_endpoint_mechanism": "boundary wave dynamics for free, plus, and minus phases",
        "epochs": profile.epochs,
        "batches_per_epoch": profile.batches_per_epoch,
        "batch_size": profile.batch_size,
        "maximum_phase_steps": profile.maximum_phase_steps,
        "free_tolerance": tolerance.free,
        "nudged_tolerance": tolerance.nudged,
        "wall_time_seconds_this_invocation": time.perf_counter() - start_time,
        **{f"train_{key}": value for key, value in final_train.items()},
        **{f"test_{key}": value for key, value in final_test.items()},
    }
    return params, history, result


def run(
    dataset: str,
    profile_name: str,
    seeds: tuple[int, ...],
    cache_dir: Path,
    gold_lock_file: Path,
    output_dir: Path,
) -> dict[str, object]:
    profile = PROFILES[profile_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_image_dataset(
        dataset, cache_dir, profile.pca_components
    )
    damping, tolerance, lock_hash = load_gold_lock(gold_lock_file)
    runs_path = output_dir / f"{dataset}_dynamic_runs.csv"
    runs = pd.read_csv(runs_path).to_dict(orient="records") if runs_path.exists() else []
    completed = {int(row["seed"]) for row in runs}
    for seed in seeds:
        if seed in completed:
            print(f"dynamic {dataset} seed={seed} already completed; skipping")
            continue
        params, _, result = train_seed_dynamic(
            seed,
            profile,
            x_train,
            y_train,
            x_test,
            y_test,
            damping,
            tolerance,
            output_dir,
        )
        config = make_config(profile, x_train.shape[1])
        save_model(output_dir / f"mnist_model_seed_{seed}.npz", params, config)
        result["dataset"] = dataset
        result["gold_lock_sha256"] = lock_hash
        runs.append(result)
        pd.DataFrame(runs).to_csv(runs_path, index=False)
    runs_df = pd.DataFrame(runs)
    summary = {
        "version": "4.1-dynamic-end-to-end",
        "dataset": dataset,
        "training_protocol": "all gradient-forming endpoints generated by boundary wave dynamics",
        "evaluation_protocol": "exact equilibria for accuracy; physical endpoint validation is performed separately",
        "profile": profile_name,
        "profile_config": asdict(profile),
        "gold_lock_sha256": lock_hash,
        "damping_candidate": asdict(damping),
        "tolerance_candidate": asdict(tolerance),
        "requested_seeds": list(seeds),
        "n_completed_seeds": int(runs_df.shape[0]),
        "mean_test_accuracy": float(runs_df["test_accuracy"].mean()),
        "std_test_accuracy": float(runs_df["test_accuracy"].std(ddof=1)) if runs_df.shape[0] > 1 else 0.0,
        "complete": set(runs_df["seed"].astype(int)) == set(seeds),
    }
    (output_dir / f"{dataset}_dynamic_summary_v4.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "fashion_mnist"), default="mnist")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="paper")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--cache-dir", type=Path, default=Path("image_cache_v4"))
    parser.add_argument("--gold-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results_v4_dynamic_training"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.dataset,
        arguments.profile,
        parse_seeds(arguments.seeds),
        arguments.cache_dir,
        arguments.gold_lock,
        arguments.output_dir,
    )
