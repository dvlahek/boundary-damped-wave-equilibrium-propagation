#!/usr/bin/env python3
"""Validate image-model endpoints using fixed damping and solver tolerances."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v4-lock")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnist_boundary_calibration_v33 import DampingCandidate, load_models
from mnist_tolerance_audit_v34 import ToleranceCandidate, run_locked_audit
from v4_train_image_models import parse_seeds


STATE_ERROR_THRESHOLD = 1e-4
GRADIENT_ERROR_THRESHOLD = 1e-2
GRADIENT_COSINE_THRESHOLD = 0.999
GRADIENT_STABILITY_THRESHOLD = 2e-3


def load_gold_lock(path: Path) -> tuple[DampingCandidate, ToleranceCandidate, str]:
    raw = path.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    if not bool(lock.get("locked_before_v34_test_audit", False)):
        raise ValueError("The supplied damping and tolerance profile was not fixed before test evaluation")
    damping = lock["locked_damping_candidate"]
    tolerance = lock["selected_tolerance"]
    return (
        DampingCandidate(int(damping["width"]), float(damping["trace"]), float(damping["power"])),
        ToleranceCandidate(float(tolerance["free"]), float(tolerance["nudged"])),
        hashlib.sha256(raw).hexdigest(),
    )


def prior_indices(paths: tuple[Path, ...]) -> tuple[set[int], dict[str, str]]:
    excluded: set[int] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Prior phase file not found: {path}")
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        table = pd.read_csv(path, usecols=["audit_indices_json"])
        for text in table["audit_indices_json"].dropna().unique():
            excluded.update(int(value) for value in json.loads(str(text)))
    return excluded, hashes


def make_plot(curves: pd.DataFrame, methods: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for seed, block in curves.groupby("seed"):
        axes[0].plot(
            block["budget"], block["gradient_vs_exact_centered_relative_error"], marker="o", label=f"seed={seed}"
        )
        axes[1].plot(
            block["budget"], block["minimum_phase_convergence_fraction"], marker="o", label=f"seed={seed}"
        )
    axes[0].axhline(GRADIENT_ERROR_THRESHOLD, color="black", linestyle="--", label="1% threshold")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(title=f"{dataset}: gradient fidelity", xlabel="budget", ylabel="relative error")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set(title="Worst phase convergence", xlabel="budget", ylabel="converged fraction")
    axes[1].legend(fontsize=7)
    axes[2].bar(methods["seed"].astype(str), methods["gradient_vs_exact_centered_relative_error"])
    axes[2].axhline(GRADIENT_ERROR_THRESHOLD, color="black", linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set(title="Final relaxation-gradient error", xlabel="seed", ylabel="relative gradient error")
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset}_locked_replication_v4.png", dpi=190)
    plt.close(fig)


def run(
    dataset: str,
    models_dir: Path,
    cache_file: Path,
    gold_lock_file: Path,
    seeds: tuple[int, ...],
    exclude_phase_files: tuple[Path, ...],
    output_dir: Path,
    mode: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(models_dir)
    missing = set(seeds) - set(models)
    if missing:
        raise FileNotFoundError(f"Missing trained model seeds: {sorted(missing)}")
    damping, tolerance, lock_hash = load_gold_lock(gold_lock_file)
    excluded, prior_hashes = prior_indices(exclude_phase_files) if exclude_phase_files else (set(), {})
    with np.load(cache_file, allow_pickle=False) as cache:
        x_test = np.asarray(cache["x_test"]).copy()
        y_test = np.asarray(cache["y_test"], dtype=np.int64).copy()
    if mode == "quick":
        audit_samples = min(8, x_test.shape[0])
        budgets = (500, 1_000, 2_000)
    else:
        audit_samples = 128
        budgets = (30_000, 60_000, 120_000, 240_000, 480_000)

    phases_path = output_dir / f"{dataset}_locked_phases_v4.csv"
    curves_path = output_dir / f"{dataset}_locked_budget_curves_v4.csv"
    methods_path = output_dir / f"{dataset}_locked_methods_v4.csv"
    phase_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    if phases_path.exists() and curves_path.exists() and methods_path.exists():
        existing = pd.read_csv(methods_path)
        if (
            not existing.empty
            and set(existing["damping_candidate"].astype(str)) == {damping.key}
            and set(existing["tolerance_candidate"].astype(str)) == {tolerance.key}
            and set(existing["budget"].astype(int)) == {budgets[-1]}
        ):
            phase_rows = pd.read_csv(phases_path).to_dict(orient="records")
            curve_rows = pd.read_csv(curves_path).to_dict(orient="records")
            method_rows = existing.to_dict(orient="records")
    completed = {int(row["seed"]) for row in method_rows}
    available = np.asarray(sorted(set(range(x_test.shape[0])) - excluded), dtype=np.int64)
    if available.size < audit_samples:
        raise ValueError("Not enough unobserved test samples remain")
    for seed in seeds:
        if seed in completed:
            print(f"{dataset} seed={seed} validation already completed; skipping")
            continue
        rng = np.random.default_rng(seed + 1_400_000)
        indices = rng.choice(available, size=audit_samples, replace=False)
        phases, curves, method = run_locked_audit(
            seed,
            models[seed][0],
            models[seed][1],
            x_test[indices],
            y_test[indices],
            damping,
            tolerance,
            budgets,
            confirmatory_model_seed=True,
        )
        for row in phases:
            row["dataset"] = dataset
            row["audit_indices_json"] = json.dumps(indices.tolist())
            row["prior_excluded_indices_json"] = json.dumps(sorted(excluded))
            row["fresh_subset_disjoint_from_prior_audits"] = not bool(
                excluded.intersection(indices.tolist())
            )
        for row in curves:
            row["dataset"] = dataset
        method["dataset"] = dataset
        method["gold_lock_sha256"] = lock_hash
        method["n_prior_excluded_test_samples"] = len(excluded)
        method["fresh_subset_disjoint_from_prior_audits"] = not bool(
            excluded.intersection(indices.tolist())
        )
        phase_rows.extend(phases)
        curve_rows.extend(curves)
        method_rows.append(method)
        pd.DataFrame(phase_rows).to_csv(phases_path, index=False)
        pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
        pd.DataFrame(method_rows).to_csv(methods_path, index=False)

    curves_df = pd.DataFrame(curve_rows)
    methods_df = pd.DataFrame(method_rows)
    make_plot(curves_df, methods_df, dataset, output_dir)
    summary = {
        "version": "4.1-fixed-configuration-validation",
        "dataset": dataset,
        "trained_models_unchanged_during_audit": True,
        "damping_candidate": asdict(damping),
        "damping_candidate_key": damping.key,
        "tolerance_candidate": asdict(tolerance),
        "tolerance_candidate_key": tolerance.key,
        "gold_lock_sha256": lock_hash,
        "prior_phase_sha256": prior_hashes,
        "requested_seeds": list(seeds),
        "n_completed_seeds": int(methods_df.shape[0]),
        "all_strict_pass": bool(methods_df["strict_all_phases_converged"].all()),
        "all_finite_horizon_pass": bool(methods_df["finite_horizon_certificate_pass"].all()),
        "all_fresh_subsets_disjoint": bool(
            methods_df["fresh_subset_disjoint_from_prior_audits"].all()
        ),
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
    }
    (output_dir / f"{dataset}_locked_summary_v4.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "fashion_mnist"), default="mnist")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--gold-lock", type=Path, required=True)
    parser.add_argument("--seeds", default="59,71,97,131,193")
    parser.add_argument("--exclude-phases", type=Path, nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "paper"), default="paper")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.dataset,
        arguments.models_dir,
        arguments.cache_file,
        arguments.gold_lock,
        parse_seeds(arguments.seeds),
        tuple(arguments.exclude_phases),
        arguments.output_dir,
        arguments.mode,
    )
