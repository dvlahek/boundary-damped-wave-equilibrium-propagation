#!/usr/bin/env python3
"""Pre-register a boundary-damping schedule without using paper test data.

The calibration uses separate seeds and small synthetic batches.  Every
candidate is evaluated by endpoint convergence, state error relative to the
Newton equilibrium, centered-EqProp gradient error, and relaxation steps.  The
selected schedule is written to ``selected_damping.txt`` for the Windows paper
runner.  No classification test accuracy is used for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_rep_experiments import DynamicsConfig, relax_dynamics
from benchmark_suite import (
    CHAIN_REGIMES,
    DAMPING_SCHEDULES,
    chain_configuration,
    make_dataset,
)
from boundary_rep_learning import (
    flatten_gradients,
    implicit_gradients,
    initialize_parameters,
    make_features,
    solve_equilibrium,
    symmetric_eqprop_gradients,
)


CALIBRATION_SEEDS = (17, 29)
CALIBRATION_DATASETS = ("moons", "circles", "blobs")
CALIBRATION_SIZES = (7, 9)


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - reference)
        / max(np.linalg.norm(reference), 1e-12)
    )


def cosine_similarity(estimate: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(estimate) * np.linalg.norm(reference)
    if denominator <= 1e-14:
        return 1.0 if np.linalg.norm(estimate - reference) <= 1e-12 else 0.0
    return float(np.dot(estimate, reference) / denominator)


def calibration_problem(
    dataset: str,
    seed: int,
    n_samples: int,
    n_rbf: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x, targets = make_dataset(dataset, n_samples, rng)
    x = (x - np.mean(x, axis=0)) / np.maximum(np.std(x, axis=0), 1e-12)
    centers = x[rng.choice(n_samples, n_rbf, replace=False)]
    return make_features(x, centers, 0.72), targets


def evaluate_candidate(
    schedule: str,
    damping_trace: float,
    damping_scale: float,
    dataset: str,
    chain_size: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    features, targets = calibration_problem(
        dataset, seed, args.n_samples, args.n_rbf
    )
    config = chain_configuration(
        chain_size,
        seed,
        args.beta,
        schedule,
        damping_trace,
        damping_scale,
        args.chain_regime,
    )
    params = initialize_parameters(config, features.shape[1])
    strict_dynamics = DynamicsConfig(
        dt=args.dynamics_dt,
        max_steps=args.max_steps,
        gradient_tolerance=args.tolerance,
        velocity_tolerance=args.tolerance,
    )
    free_dynamics = DynamicsConfig(
        dt=args.dynamics_dt,
        max_steps=args.max_steps,
        gradient_tolerance=args.tolerance * args.free_tolerance_multiplier,
        velocity_tolerance=args.tolerance * args.free_tolerance_multiplier,
    )

    free = relax_dynamics(params, features, config, free_dynamics)
    plus = relax_dynamics(
        params,
        features,
        config,
        strict_dynamics,
        targets=targets,
        beta=args.beta,
        initial_q=np.asarray(free["q"]),
    )
    minus = relax_dynamics(
        params,
        features,
        config,
        strict_dynamics,
        targets=targets,
        beta=-args.beta,
        initial_q=np.asarray(free["q"]),
    )

    q_free, _, _ = solve_equilibrium(params, features, config)
    q_plus, _, _ = solve_equilibrium(
        params, features, config, targets, args.beta, initial=q_free
    )
    q_minus, _, _ = solve_equilibrium(
        params, features, config, targets, -args.beta, initial=q_free
    )
    implicit = flatten_gradients(
        implicit_gradients(q_free, params, features, targets, config)
    )
    dynamic_gradient = flatten_gradients(
        symmetric_eqprop_gradients(
            np.asarray(minus["q"]),
            np.asarray(plus["q"]),
            params,
            features,
            config,
            args.beta,
        )
    )
    exact_centered = flatten_gradients(
        symmetric_eqprop_gradients(
            q_minus, q_plus, params, features, config, args.beta
        )
    )
    endpoint_convergence = np.mean(
        [bool(free["converged"]), bool(plus["converged"]), bool(minus["converged"])]
    )
    return {
        "calibration_key": (
            f"{schedule}|trace={damping_trace:.4g}|scale={damping_scale:.4g}"
            f"|dataset={dataset}|n={chain_size}|seed={seed}"
            f"|samples={args.n_samples}|rbf={args.n_rbf}|steps={args.max_steps}"
            f"|tol={args.tolerance:.3g}|beta={args.beta:.4g}"
            f"|dt={args.dynamics_dt:.4g}"
            f"|freetolmult={args.free_tolerance_multiplier:.4g}"
            f"|regime={args.chain_regime}"
        ),
        "schedule": schedule,
        "candidate": (
            f"{schedule}:trace={damping_trace:.4g}:scale={damping_scale:.4g}"
        ),
        "damping_trace": damping_trace,
        "damping_scale": damping_scale,
        "chain_regime": args.chain_regime,
        "base_tolerance": args.tolerance,
        "dynamics_dt": args.dynamics_dt,
        "free_tolerance_multiplier": args.free_tolerance_multiplier,
        "free_tolerance": args.tolerance * args.free_tolerance_multiplier,
        "dataset": dataset,
        "chain_size": chain_size,
        "seed": seed,
        "endpoint_convergence_fraction": float(endpoint_convergence),
        "free_state_relative_error": relative_error(np.asarray(free["q"]), q_free),
        "plus_state_relative_error": relative_error(np.asarray(plus["q"]), q_plus),
        "minus_state_relative_error": relative_error(np.asarray(minus["q"]), q_minus),
        "gradient_vs_implicit_relative_error": relative_error(
            dynamic_gradient, implicit
        ),
        "gradient_vs_exact_centered_relative_error": relative_error(
            dynamic_gradient, exact_centered
        ),
        "gradient_vs_implicit_cosine": cosine_similarity(
            dynamic_gradient, implicit
        ),
        "gradient_vs_exact_centered_cosine": cosine_similarity(
            dynamic_gradient, exact_centered
        ),
        "total_steps": int(free["steps"])
        + int(plus["steps"])
        + int(minus["steps"]),
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    result = (
        rows.groupby(
            [
                "candidate",
                "schedule",
                "damping_trace",
                "damping_scale",
                "chain_regime",
                "free_tolerance_multiplier",
            ],
            as_index=False,
        )
        .agg(
            n_cases=("seed", "count"),
            convergence_fraction=("endpoint_convergence_fraction", "mean"),
            worst_case_convergence=("endpoint_convergence_fraction", "min"),
            median_gradient_error=("gradient_vs_implicit_relative_error", "median"),
            max_gradient_error=("gradient_vs_implicit_relative_error", "max"),
            median_free_state_error=("free_state_relative_error", "median"),
            max_free_state_error=("free_state_relative_error", "max"),
            median_centered_error=(
                "gradient_vs_exact_centered_relative_error",
                "median",
            ),
            max_centered_error=(
                "gradient_vs_exact_centered_relative_error",
                "max",
            ),
            p95_centered_error=(
                "gradient_vs_exact_centered_relative_error",
                lambda values: float(np.quantile(values, 0.95)),
            ),
            median_centered_cosine=(
                "gradient_vs_exact_centered_cosine",
                "median",
            ),
            min_centered_cosine=(
                "gradient_vs_exact_centered_cosine",
                "min",
            ),
            median_total_steps=("total_steps", "median"),
            max_total_steps=("total_steps", "max"),
        )
    )
    result["passes_gate"] = (
        (result["schedule"] != "legacy")
        # A calibration candidate is eligible only if every free and nudged
        # endpoint converged in every held-out calibration case.
        & (result["worst_case_convergence"] >= 1.0 - 1e-12)
        & (result["median_centered_error"] <= 0.01)
        & (result["max_centered_error"] <= 0.02)
        & (result["max_free_state_error"] <= 0.05)
        & (result["median_centered_cosine"] >= 0.999)
        & (result["min_centered_cosine"] >= 0.99)
    )
    # Once candidates pass the accuracy and convergence gates, select for
    # worst-case relaxation cost first.  This avoids choosing a much slower
    # schedule because of numerically immaterial cosine/error differences.
    result = result.sort_values(
        [
            "passes_gate",
            "worst_case_convergence",
            "max_total_steps",
            "median_total_steps",
            "max_centered_error",
            "median_centered_error",
        ],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    return result


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "damping_calibration.csv"
    rows: list[dict[str, object]] = (
        pd.read_csv(details_path).to_dict(orient="records")
        if args.resume and details_path.exists()
        else []
    )
    if rows and "calibration_key" not in rows[0]:
        raise ValueError(
            "The calibration directory contains a pre-v2 CSV without calibration keys. "
            "Use a new output directory for the corrected calibration."
        )
    if rows and (
        "free_tolerance_multiplier" not in rows[0]
        or not np.allclose(
            [float(row["free_tolerance_multiplier"]) for row in rows],
            args.free_tolerance_multiplier,
        )
    ):
        raise ValueError(
            "The calibration directory uses another free-phase tolerance rule. "
            "Use a new output directory."
        )
    if rows and (
        "dynamics_dt" not in rows[0]
        or not np.allclose(
            [float(row["dynamics_dt"]) for row in rows], args.dynamics_dt
        )
    ):
        raise ValueError(
            "The calibration directory uses another dynamics time step. "
            "Use a new output directory."
        )
    if rows and (
        "chain_regime" not in rows[0]
        or {str(row["chain_regime"]) for row in rows} != {args.chain_regime}
    ):
        raise ValueError(
            "The calibration directory belongs to another chain regime. "
            "Use a new output directory."
        )
    completed = {str(row["calibration_key"]) for row in rows}
    candidates: list[tuple[str, float, float]] = []
    if "legacy" in args.schedules:
        candidates.append(("legacy", 1.0, 1.0))
    for schedule in ("fixed_trace", "graded_trace"):
        if schedule in args.schedules:
            candidates.extend(
                (schedule, trace, 1.0) for trace in args.trace_values
            )
    if "impedance_terminal" in args.schedules:
        candidates.extend(
            ("impedance_terminal", 1.0, scale)
            for scale in args.impedance_scales
        )
    total = (
        len(candidates)
        * len(args.datasets)
        * len(args.chain_sizes)
        * len(args.seeds)
    )
    index = 0
    for schedule, damping_trace, damping_scale in candidates:
        for dataset in args.datasets:
            for chain_size in args.chain_sizes:
                for seed in args.seeds:
                    index += 1
                    key = (
                        f"{schedule}|trace={damping_trace:.4g}|scale={damping_scale:.4g}"
                        f"|dataset={dataset}|n={chain_size}|seed={seed}"
                        f"|samples={args.n_samples}|rbf={args.n_rbf}|steps={args.max_steps}"
                        f"|tol={args.tolerance:.3g}|beta={args.beta:.4g}"
                        f"|dt={args.dynamics_dt:.4g}"
                        f"|freetolmult={args.free_tolerance_multiplier:.4g}"
                        f"|regime={args.chain_regime}"
                    )
                    if key in completed:
                        print(f"[{index}/{total}] already completed {key}", flush=True)
                        continue
                    print(
                        f"[{index}/{total}] schedule={schedule} trace={damping_trace:g} "
                        f"scale={damping_scale:g} dataset={dataset} "
                        f"n={chain_size} seed={seed}",
                        flush=True,
                    )
                    rows.append(
                        evaluate_candidate(
                            schedule,
                            damping_trace,
                            damping_scale,
                            dataset,
                            chain_size,
                            seed,
                            args,
                        )
                    )
                    pd.DataFrame(rows).to_csv(details_path, index=False)

    details = pd.DataFrame(rows)
    table = summarize(details)
    table.to_csv(args.output_dir / "damping_calibration_summary.csv", index=False)
    best = table.iloc[0]
    selection = {
        "selected_schedule": str(best["schedule"]),
        "selected_damping_trace": float(best["damping_trace"]),
        "selected_damping_scale": float(best["damping_scale"]),
        "passes_gate": bool(best["passes_gate"]),
        "selection_uses_test_accuracy": False,
        "selection_rule": (
            "all-endpoint convergence and centered-gradient gate, then "
            "minimum worst-case and median relaxation steps"
        ),
        "chain_regime": args.chain_regime,
        "base_tolerance": args.tolerance,
        "dynamics_dt": args.dynamics_dt,
        "free_tolerance_multiplier": args.free_tolerance_multiplier,
        "free_tolerance": args.tolerance * args.free_tolerance_multiplier,
        "calibration_seeds": list(args.seeds),
        "calibration_datasets": list(args.datasets),
        "calibration_chain_sizes": list(args.chain_sizes),
        "convergence_fraction": float(best["convergence_fraction"]),
        "median_centered_gradient_error": float(best["median_centered_error"]),
        "max_centered_gradient_error": float(best["max_centered_error"]),
        "median_centered_gradient_cosine": float(best["median_centered_cosine"]),
        "minimum_centered_gradient_cosine": float(best["min_centered_cosine"]),
        "median_free_state_relative_error": float(best["median_free_state_error"]),
        "max_free_state_relative_error": float(best["max_free_state_error"]),
        "median_total_steps": float(best["median_total_steps"]),
    }
    (args.output_dir / "best_damping.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    (args.output_dir / "selected_damping.txt").write_text(
        str(best["schedule"]), encoding="utf-8"
    )
    (args.output_dir / "selected_damping_args.txt").write_text(
        f"{best['schedule']} {float(best['damping_trace']):.12g} "
        f"{float(best['damping_scale']):.12g}",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(json.dumps(selection, indent=2))
    if not bool(best["passes_gate"]) and not args.allow_failed_gate:
        raise SystemExit(
            "No damping schedule passed the pre-registered convergence and "
            "gradient gate. Inspect damping_calibration_summary.csv before the "
            "full benchmark."
        )


def csv_tuple(value: str, cast) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_damping_calibration_v2_3")
    )
    parser.add_argument(
        "--schedules",
        default=",".join(DAMPING_SCHEDULES),
        help="comma-separated damping schedules",
    )
    parser.add_argument(
        "--datasets", default=",".join(CALIBRATION_DATASETS)
    )
    parser.add_argument(
        "--chain-sizes",
        default=",".join(str(value) for value in CALIBRATION_SIZES),
    )
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in CALIBRATION_SEEDS)
    )
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--n-rbf", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=40_000)
    parser.add_argument("--tolerance", type=float, default=7e-6)
    parser.add_argument("--dynamics-dt", type=float, default=0.20)
    parser.add_argument(
        "--free-tolerance-multiplier",
        type=float,
        default=100.0,
        help="multiplier used only for the free calibration endpoint",
    )
    parser.add_argument("--beta", type=float, default=0.035)
    parser.add_argument(
        "--chain-regime", choices=CHAIN_REGIMES, default="propagating"
    )
    parser.add_argument(
        "--trace-values", default="0.35,0.50,0.70,1.00,1.50"
    )
    parser.add_argument(
        "--impedance-scales", default="0.50,0.75,1.00"
    )
    parser.add_argument("--allow-failed-gate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.schedules = csv_tuple(args.schedules, str)
    args.datasets = csv_tuple(args.datasets, str)
    args.chain_sizes = csv_tuple(args.chain_sizes, int)
    args.seeds = csv_tuple(args.seeds, int)
    args.trace_values = csv_tuple(args.trace_values, float)
    args.impedance_scales = csv_tuple(args.impedance_scales, float)
    unknown = set(args.schedules) - set(DAMPING_SCHEDULES)
    if unknown:
        parser.error(f"unknown schedules: {sorted(unknown)}")
    if args.free_tolerance_multiplier < 1.0:
        parser.error("--free-tolerance-multiplier must be at least 1")
    if args.dynamics_dt <= 0.0:
        parser.error("--dynamics-dt must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
