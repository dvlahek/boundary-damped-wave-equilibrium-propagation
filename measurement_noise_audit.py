#!/usr/bin/env python3
"""Audit the local centered gradient under noisy physical state readout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v5-noise")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graph_eqprop_core import (
    centered_gradients,
    cosine_similarity,
    flatten,
    implicit_gradients,
    initialize_parameters,
    make_config,
    relative_error,
    select_modal_boundary,
    solve_equilibrium,
)


PROFILES = {
    "quick": {
        "topologies": ("grid_3",),
        "seeds": (17,),
        "n_samples": 8,
        "replicates": 40,
        "noise_levels": (0.0, 1e-4, 1e-3, 1e-2),
    },
    "paper": {
        "topologies": ("grid_4", "grid_5", "sparse_16", "sparse_25"),
        "seeds": (17, 29, 43, 71, 97),
        "n_samples": 24,
        "replicates": 250,
        "noise_levels": (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    },
}


def run(profile: str, output_dir: Path) -> pd.DataFrame:
    settings = PROFILES[profile]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "measurement_noise_runs.csv"
    if result_path.exists():
        existing = pd.read_csv(result_path)
        rows: list[dict[str, object]] = existing.to_dict(orient="records")
        completed_pairs = {
            (str(topology), int(seed))
            for (topology, seed), part in existing.groupby(
                ["topology", "seed"], observed=True
            )
            if len(part)
            == 2 * len(settings["noise_levels"]) * settings["replicates"]
        }
        print(f"resuming from {len(existing)} completed noise rows", flush=True)
    else:
        rows = []
        completed_pairs: set[tuple[str, int]] = set()
    for topology in settings["topologies"]:
        for seed in settings["seeds"]:
            if (topology, int(seed)) in completed_pairs:
                print(f"topology={topology} seed={seed}: already completed", flush=True)
                continue
            print(f"topology={topology} seed={seed}: exact endpoint readouts", flush=True)
            config = make_config(topology, seed)
            rng_problem = np.random.default_rng(seed + 30_013)
            features = rng_problem.normal(size=(settings["n_samples"], 7))
            features[:, 0] = 1.0
            targets = np.where(
                features[:, 1] * features[:, 2] + 0.35 * features[:, 3] >= 0.0,
                1.0,
                -1.0,
            )
            params = initialize_parameters(config, features.shape[1], seed + 40_009)
            if topology.startswith("sparse_"):
                preliminary_free, _, _ = solve_equilibrium(params, features, config)
                config = select_modal_boundary(
                    config, params, q_states=preliminary_free
                )
            q_free, _, _ = solve_equilibrium(params, features, config)
            q_plus, _, _ = solve_equilibrium(
                params, features, config, targets, config.beta, initial=q_free
            )
            q_minus, _, _ = solve_equilibrium(
                params, features, config, targets, -config.beta, initial=q_free
            )
            exact_centered = flatten(
                centered_gradients(
                    q_minus, q_plus, params, features, config, config.beta
                )
            )
            implicit = flatten(
                implicit_gradients(q_free, params, features, targets, config)
            )
            signal_rms = max(
                float(np.sqrt(np.mean(np.concatenate((q_plus.ravel(), q_minus.ravel())) ** 2))),
                1e-12,
            )
            for noise_model in ("independent_phase", "shared_sensor_bias"):
                for noise_level in settings["noise_levels"]:
                    for replicate in range(settings["replicates"]):
                        rng = np.random.default_rng(
                            seed * 1_000_003
                            + replicate * 1009
                            + int(round(noise_level * 1e8))
                            + (0 if noise_model == "independent_phase" else 97)
                        )
                        standard_deviation = noise_level * signal_rms
                        if noise_model == "independent_phase":
                            noise_plus = rng.normal(scale=standard_deviation, size=q_plus.shape)
                            noise_minus = rng.normal(scale=standard_deviation, size=q_minus.shape)
                        else:
                            shared = rng.normal(scale=standard_deviation, size=q_plus.shape)
                            noise_plus = shared
                            noise_minus = shared
                        noisy = flatten(
                            centered_gradients(
                                q_minus + noise_minus,
                                q_plus + noise_plus,
                                params,
                                features,
                                config,
                                config.beta,
                            )
                        )
                        rows.append(
                            {
                                "profile": profile,
                                "topology": topology,
                                "seed": seed,
                                "noise_model": noise_model,
                                "relative_readout_noise": noise_level,
                                "absolute_noise_standard_deviation": standard_deviation,
                                "replicate": replicate,
                                "noisy_vs_exact_centered_gradient_error": relative_error(
                                    noisy, exact_centered
                                ),
                                "noisy_vs_implicit_gradient_error": relative_error(
                                    noisy, implicit
                                ),
                                "noisy_vs_implicit_cosine": cosine_similarity(
                                    noisy, implicit
                                ),
                                "exact_centered_vs_implicit_gradient_error": relative_error(
                                    exact_centered, implicit
                                ),
                            }
                        )
            pd.DataFrame(rows).to_csv(result_path, index=False)
            completed_pairs.add((topology, int(seed)))
    table = pd.DataFrame(rows)
    table.to_csv(result_path, index=False)
    summary = (
        table.groupby(
            ["topology", "noise_model", "relative_readout_noise"],
            as_index=False,
            observed=True,
        )
        .agg(
            median_gradient_error=("noisy_vs_implicit_gradient_error", "median"),
            q95_gradient_error=(
                "noisy_vs_implicit_gradient_error",
                lambda value: float(np.quantile(value, 0.95)),
            ),
            minimum_cosine=("noisy_vs_implicit_cosine", "min"),
            q05_cosine=(
                "noisy_vs_implicit_cosine",
                lambda value: float(np.quantile(value, 0.05)),
            ),
        )
    )
    summary.to_csv(output_dir / "measurement_noise_summary.csv", index=False)
    machine_summary = {
        "profile": profile,
        "noise_definition": "Gaussian state-readout standard deviation divided by the RMS exact plus/minus state",
        "noise_models": {
            "independent_phase": "independent sensor noise in positive and negative phase readouts",
            "shared_sensor_bias": "same additive sensor offset in both phase readouts",
        },
        "number_of_raw_rows": int(len(table)),
        "largest_noise_level": float(max(settings["noise_levels"])),
        "summary": summary.to_dict(orient="records"),
    }
    with (output_dir / "measurement_noise_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(machine_summary, handle, indent=2)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for (topology, noise_model), part in summary.groupby(
        ["topology", "noise_model"], observed=True
    ):
        positive = part[part["relative_readout_noise"] > 0.0]
        label = f"{topology}, {noise_model}"
        axes[0].plot(
            positive["relative_readout_noise"],
            positive["q95_gradient_error"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            positive["relative_readout_noise"],
            positive["q05_cosine"],
            marker="o",
            label=label,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("95th-percentile relative gradient error")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.05, 1.01)
    axes[1].set_ylabel("5th-percentile cosine similarity")
    for axis in axes:
        axis.set_xlabel("relative state-readout noise")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="lower left")
    figure.suptitle("Robustness of local phase-contrast measurements")
    figure.tight_layout()
    figure.savefig(output_dir / "measurement_noise_audit.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path(f"results_measurement_noise_{args.profile}")
    summary = run(args.profile, output)
    print(f"saved {len(summary)} summarized noise rows to {output}")


if __name__ == "__main__":
    main()
