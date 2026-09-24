#!/usr/bin/env python3
"""Train exact-centered EqProp image models for independent endpoint validation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep-v4-images")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from mnist_rep_benchmark import (
    PROFILES,
    evaluate,
    logistic_baseline,
    make_config,
    save_model,
    train_seed,
)


DATASETS = {
    "mnist": {"openml_id": 554, "title": "MNIST"},
    "fashion_mnist": {"openml_id": 40996, "title": "Fashion-MNIST"},
}


def parse_seeds(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("--seeds must contain unique comma-separated integers")
    return values


def cache_filename(dataset: str, pca_components: int) -> str:
    prefix = "mnist" if dataset == "mnist" else "fashion_mnist"
    return f"{prefix}_pca_{pca_components}.npz"


def load_image_dataset(
    dataset: str,
    cache_dir: Path,
    pca_components: int,
) -> tuple[np.ndarray, ...]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared = cache_dir / cache_filename(dataset, pca_components)
    if prepared.exists():
        # Copy arrays before leaving the context so Windows can release the
        # underlying .npz file immediately (important for temporary folders
        # and cache replacement).
        with np.load(prepared, allow_pickle=False) as stored:
            return (
                np.asarray(stored["x_train"]).copy(),
                np.asarray(stored["y_train"]).copy(),
                np.asarray(stored["x_test"]).copy(),
                np.asarray(stored["y_test"]).copy(),
            )
    specification = DATASETS[dataset]
    print(
        f"Downloading {specification['title']} from OpenML data_id={specification['openml_id']}. "
        "This occurs only once for each PCA cache."
    )
    fetched = fetch_openml(
        data_id=int(specification["openml_id"]),
        as_frame=False,
        parser="auto",
        data_home=cache_dir / "openml",
    )
    x = np.asarray(fetched.data, dtype=np.float32) / 255.0
    y = np.asarray(fetched.target, dtype=np.int64)
    if x.shape[0] != 70_000 or np.unique(y).size != 10:
        raise ValueError(
            f"Expected 70000 examples and 10 classes, received {x.shape[0]} and {np.unique(y).size}"
        )
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
        prepared,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        dataset=dataset,
        openml_id=int(specification["openml_id"]),
        pca_train_only=True,
    )
    return x_train, y_train, x_test, y_test


def make_plot(history: pd.DataFrame, runs: pd.DataFrame, baseline: dict[str, object], dataset: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for seed, block in history.groupby("seed"):
        axes[0].plot(block["epoch"], block["test_accuracy"], marker="o", label=f"seed={seed}")
    axes[0].axhline(
        float(baseline["test_accuracy"]), color="black", linestyle="--", label="PCA logistic"
    )
    axes[0].set(
        title=f"{DATASETS[dataset]['title']} exact-centered training",
        xlabel="epoch",
        ylabel="test accuracy",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(runs["seed"].astype(str), runs["test_accuracy"])
    axes[1].axhline(float(baseline["test_accuracy"]), color="black", linestyle="--")
    axes[1].set(title="Final accuracy by model seed", xlabel="seed", ylabel="test accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset}_v4_model_training.png", dpi=190)
    plt.close(fig)


def run(
    dataset: str,
    profile_name: str,
    seeds: tuple[int, ...],
    cache_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    profile = replace(PROFILES[profile_name], seeds=seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_image_dataset(
        dataset, cache_dir, profile.pca_components
    )
    baseline_path = output_dir / f"{dataset}_pca_logistic_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = logistic_baseline(x_train, y_train, x_test, y_test)
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    runs_path = output_dir / f"{dataset}_model_runs.csv"
    history_path = output_dir / f"{dataset}_model_history.csv"
    runs = pd.read_csv(runs_path).to_dict(orient="records") if runs_path.exists() else []
    history = pd.read_csv(history_path).to_dict(orient="records") if history_path.exists() else []
    completed = {int(row["seed"]) for row in runs}
    for seed in seeds:
        if seed in completed:
            print(f"{dataset} seed={seed} already completed; skipping")
            continue
        history = [row for row in history if int(row["seed"]) != seed]
        params, seed_history, result = train_seed(
            seed, profile, x_train, y_train, x_test, y_test
        )
        config = make_config(profile, x_train.shape[1])
        save_model(output_dir / f"mnist_model_seed_{seed}.npz", params, config)
        result["dataset"] = dataset
        result["training_endpoint_mechanism"] = "exact-centered EqProp"
        runs.append(result)
        history.extend(seed_history)
        pd.DataFrame(history).to_csv(history_path, index=False)
        pd.DataFrame(runs).to_csv(runs_path, index=False)
    runs_df = pd.DataFrame(runs)
    history_df = pd.DataFrame(history)
    make_plot(history_df, runs_df, baseline, dataset, output_dir)
    summary = {
        "version": "4.1",
        "dataset": dataset,
        "dataset_title": DATASETS[dataset]["title"],
        "openml_id": DATASETS[dataset]["openml_id"],
        "training_protocol": "exact-centered EqProp; physical endpoint validation is performed separately",
        "profile": profile_name,
        "profile_config": asdict(profile),
        "requested_seeds": list(seeds),
        "n_completed_seeds": int(runs_df.shape[0]),
        "mean_test_accuracy": float(runs_df["test_accuracy"].mean()),
        "std_test_accuracy": float(runs_df["test_accuracy"].std(ddof=1)) if runs_df.shape[0] > 1 else 0.0,
        "mean_test_balanced_accuracy": float(runs_df["test_balanced_accuracy"].mean()),
        "mean_test_macro_f1": float(runs_df["test_macro_f1"].mean()),
        "pca_logistic_test_accuracy": float(baseline["test_accuracy"]),
        "complete": set(runs_df["seed"].astype(int)) == set(seeds),
    }
    (output_dir / f"{dataset}_model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="mnist")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="paper")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--cache-dir", type=Path, default=Path("image_cache_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_v4_image_models"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.dataset,
        arguments.profile,
        parse_seeds(arguments.seeds),
        arguments.cache_dir,
        arguments.output_dir,
    )
