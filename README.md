# Boundary-Damped Wave Equilibrium Propagation

Reproducibility code for boundary-damped wave equilibrium propagation. The repository implements the theoretical audits, controlled synthetic
benchmark, locked MNIST and Fashion-MNIST experiments, block-aware statistical analysis, non-chain graph audit, dark-mode control, and measurement-noise audit
reported in the accompanying study.

The implementation uses a dimensionless damped wave model. It does not claim that a generic gravitational wave is a neural-network gradient. The tested
claim: localized boundary dissipation can relax a wave-mediated system toward equilibrium, while centered equilibrium perturbations estimate
the parameter gradient with a controlled finite-time error.

Every command writes its outputs under `results/`.

## Installation

Requirements:

- Python 3.12
- internet access during the first MNIST or Fashion-MNIST run

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all deterministic unit tests:

```bash
python -m unittest discover -s tests -v
```

## Quick verification

Quick profiles verify the installation and execution paths. They are smoke
tests and are not the reported paper results.

### Synthetic benchmark

```bash
python benchmark_suite.py --profile quick --damping-schedule fixed_trace --damping-trace 0.7 --damping-scale 1.0 --phase-protocol centered_pair_cache --dynamics-dt 0.20 --learning-rate-u 0.0025 --learning-rate-structure 0.0003 --free-tolerance-multiplier 100 --endpoint-step-multiplier 10 --workers 2 --output-dir results/quick_synthetic
```

### Theory audits

```bash
python audit_energy_observability.py --profile quick --output-dir results/quick_energy
python audit_gradient_scaling.py --profile quick --output-dir results/quick_gradient
python audit_finite_time_gradient.py --profile quick --output-dir results/quick_finite_time
python audit_chain_scaling.py --profile quick --output-dir results/quick_chain_scaling
```

### Statistics, topology, dark-mode, and noise audits

The statistics command uses the `benchmark_runs.csv` created by the preceding
synthetic command.

```bash
python statistical_refinement.py --input results/quick_synthetic/benchmark_runs.csv --output results/quick_statistics --bootstrap 2000 --seed 20260813
python graph_topology_audit.py --profile quick --output results/quick_graph_topology
python measurement_noise_audit.py --profile quick --output results/quick_measurement_noise
```

### MNIST exact-centered training and locked dynamic audit

```bash
python v4_train_image_models.py --dataset mnist --profile quick --seeds 59 --cache-dir image_cache --output-dir results/quick_mnist_model
python v4_locked_replication.py --dataset mnist --mode quick --seeds 59 --models-dir results/quick_mnist_model --cache-file image_cache/mnist_pca_16.npz --gold-lock locks/v4_gold_solver_lock.json --exclude-phases locks/prior_mnist_audit_v33_phases.csv locks/prior_mnist_audit_v34_phases.csv --output-dir results/quick_mnist_audit
```

### MNIST end-to-end dynamic training

```bash
python v4_dynamic_training.py --dataset mnist --profile quick --seeds 17 --cache-dir image_cache --gold-lock locks/v4_gold_solver_lock.json --output-dir results/quick_mnist_dynamic_model
python v4_locked_replication.py --dataset mnist --mode quick --seeds 17 --models-dir results/quick_mnist_dynamic_model --cache-file image_cache/mnist_pca_16.npz --gold-lock locks/v4_gold_solver_lock.json --exclude-phases locks/prior_mnist_audit_v33_phases.csv locks/prior_mnist_audit_v34_phases.csv --output-dir results/quick_mnist_dynamic_audit
```

## Reproduce the reported experiments

Paper profiles are computationally expensive. Long-running commands save
intermediate output and resume when the same output directory is reused.

### Controlled synthetic benchmark: 400 runs

The command covers five datasets, four chain sizes, five seeds, and four
gradient mechanisms. Its fixed settings are also recorded in
`locks/synthetic_paper_config.json`.

```bash
python benchmark_suite.py --profile paper --damping-schedule fixed_trace --damping-trace 0.7 --damping-scale 1.0 --phase-protocol centered_pair_cache --dynamics-dt 0.20 --learning-rate-u 0.0025 --learning-rate-structure 0.0003 --free-tolerance-multiplier 100 --endpoint-step-multiplier 10 --workers 4 --resume --output-dir results/synthetic_paper
```

### Block-aware equivalence analysis

This analysis forms 100 paired configurations, retains all chain sizes inside
25 dataset-seed blocks, performs 50,000 cluster-bootstrap resamples, and applies
Holm correction to the equivalence tests.

```bash
python statistical_refinement.py --input results/synthetic_paper/benchmark_runs.csv --output results/statistical_refinement --margin 0.02 --bootstrap 50000 --seed 20260813
```

### Theoretical chain audits

```bash
python audit_energy_observability.py --profile paper --output-dir results/theory_energy
python audit_gradient_scaling.py --profile paper --output-dir results/theory_gradient
python audit_finite_time_gradient.py --profile paper --output-dir results/theory_finite_time
python audit_chain_scaling.py --profile paper --output-dir results/theory_chain_scaling
```

### Grid, sparse-graph, and dark-mode audits

The graph command evaluates boundary and matched-trace uniform damping on
4x4 and 5x5 grids and connected sparse graphs with 16 and 25 nodes. It also
creates the symmetric-star dark-mode control.

```bash
python graph_topology_audit.py --profile paper --output results/graph_topology_paper
```

### Measurement-noise audit: 80,000 rows

```bash
python measurement_noise_audit.py --profile paper --output results/measurement_noise_paper
```

### Combined V5 summary figure

Run this after the paper statistics, graph, and noise commands.

```bash
python make_v5_summary_figure.py --statistics results/statistical_refinement/block_equivalence_summary.csv --graph results/graph_topology_paper/graph_topology_audit.csv --noise results/measurement_noise_paper/measurement_noise_runs.csv --output results/figure_v5_statistics_topology_noise.png
```

### Five-seed MNIST exact-centered replication

```bash
python v4_train_image_models.py --dataset mnist --profile paper --seeds 59,71,97,131,193 --cache-dir image_cache --output-dir results/mnist_exact_models
python v4_locked_replication.py --dataset mnist --mode paper --seeds 59,71,97,131,193 --models-dir results/mnist_exact_models --cache-file image_cache/mnist_pca_32.npz --gold-lock locks/v4_gold_solver_lock.json --exclude-phases locks/prior_mnist_audit_v33_phases.csv locks/prior_mnist_audit_v34_phases.csv --output-dir results/mnist_exact_audit
```

### Three-seed MNIST end-to-end dynamic training

```bash
python v4_dynamic_training.py --dataset mnist --profile paper --seeds 17,29,43 --cache-dir image_cache --gold-lock locks/v4_gold_solver_lock.json --output-dir results/mnist_dynamic_models
python v4_locked_replication.py --dataset mnist --mode paper --seeds 17,29,43 --models-dir results/mnist_dynamic_models --cache-file image_cache/mnist_pca_32.npz --gold-lock locks/v4_gold_solver_lock.json --exclude-phases locks/prior_mnist_audit_v33_phases.csv locks/prior_mnist_audit_v34_phases.csv --output-dir results/mnist_dynamic_audit
```

### Three-seed Fashion-MNIST transfer

```bash
python v4_train_image_models.py --dataset fashion_mnist --profile paper --seeds 17,29,43 --cache-dir image_cache --output-dir results/fashion_models
python v4_locked_replication.py --dataset fashion_mnist --mode paper --seeds 17,29,43 --models-dir results/fashion_models --cache-file image_cache/fashion_mnist_pca_32.npz --gold-lock locks/v4_gold_solver_lock.json --output-dir results/fashion_audit
```

## Registered configurations

- `locks/synthetic_paper_config.json` records the controlled synthetic setup.
- `locks/v4_gold_solver_lock.json` records the image-audit solver settings.
- `locks/prior_mnist_audit_v33_phases.csv` and
  `locks/prior_mnist_audit_v34_phases.csv` keep confirmatory samples separate
  from the earlier calibration audits.
- `locks/v5_experiment_config.json` records the block statistics, topology,
  seeds, damping, and noise settings added in release `v1.2.0-paper`.

Generated results, downloaded datasets, caches, and model checkpoints are excluded from the repository. They can be reproduced using the documented commands, fixed configurations, and deterministic seeds.

## License

MIT License. See `LICENSE`.
