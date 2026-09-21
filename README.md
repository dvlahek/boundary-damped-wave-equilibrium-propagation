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

### Theory and relaxation audits

```bash
python audit_energy_observability.py --profile quick --output-dir results/quick_energy
python audit_gradient_scaling.py --profile quick --output-dir results/quick_gradient
python audit_finite_time_gradient.py --profile quick --output-dir results/quick_finite_time
python audit_chain_scaling.py --profile quick --output-dir results/quick_chain_scaling
python audit_fixed_resource_damping.py --profile quick --output-dir results/quick_fixed_resource
python audit_modal_relaxation.py --profile quick --output-dir results/quick_modal_relaxation
python audit_graph_placement_generalization.py --profile quick --output-dir results/quick_graph_placement
python audit_timestep_refinement.py --profile quick --output-dir results/quick_timestep
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

### Higher-capacity gradient-usability control: 75 runs

This auxiliary control uses a 17-node chain and recomputes the exact free
equilibrium at every epoch. Boundary dynamics is still used for both
gradient-forming nudged phases. The 60-run base control covers two moons,
concentric circles, XOR, and intertwined spirals with 48 radial-basis
features. The one-factor extension adds 15 spiral runs with 96 features while
keeping all remaining settings fixed.

```bash
python high_capacity_control.py --workers 4
python spirals_capacity_control.py --workers 4
python spirals_96rbf_control.py --workers 4
```

The complete fixed configuration is recorded in
`locks/high_capacity_gradient_usability_config.json`. A compact summary of
the reported runs is stored in
`reported_results/high_capacity_gradient_usability_summary.csv`. Full
endpoint histories are regenerated under `results/` and remain excluded from
Git.

### Theoretical chain audits

```bash
python audit_energy_observability.py --profile paper --output-dir results/theory_energy
python audit_gradient_scaling.py --profile paper --output-dir results/theory_gradient
python audit_finite_time_gradient.py --profile paper --output-dir results/theory_finite_time
python audit_chain_scaling.py --profile paper --output-dir results/theory_chain_scaling
```

### Fixed-resource damping-support audit

This experiment fixes the 17-node chain, total damping trace, conservative
parameters, nudging strength, and free-state initialization while varying the
number of damped nodes over (m=1,2,4,6,8,17).

```bash
python audit_fixed_resource_damping.py --profile paper_refined --output-dir results/fixed_resource_damping
```

### Weak-damping modal-relaxation audit

```bash
python audit_modal_relaxation.py --profile paper --output-dir results/modal_relaxation
```

### Calibration-to-unseen sparse-graph placement

The modal damping support is selected on calibration states, frozen, and then
evaluated on unseen states against topology-only and matched-size random
supports.

```bash
python audit_graph_placement_generalization.py --profile paper --output-dir results/graph_placement
```

### Timestep-refinement control

```bash
python audit_timestep_refinement.py --profile paper --output-dir results/timestep_refinement
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

## Locked numerical results and manuscript figures

The numerical tables used for the reported finite-relaxation, fixed-resource
damping, modal-relaxation, graph-placement, and timestep-refinement results are
stored under `reported_results/`. These are locked outputs from the paper-scale
runs, not recomputed summaries.

The manuscript figures associated with those tables can be regenerated with:

```bash
python make_paper_figures.py \
  --support-dir reported_results/fixed_resource_damping \
  --graph-dir reported_results/graph_placement \
  --modal-dir reported_results/modal_relaxation \
  --finite-time-dir reported_results/finite_time \
  --timestep-dir reported_results/timestep_refinement \
  --output-dir reported_results/figures
```

The generated `reported_results/figures/figure_provenance.json` records the
SHA-256 hash of every numerical input used by each generated figure.
`reported_results/manifest.json` records SHA-256 hashes and sizes for the
locked result files retained in the repository.

## Registered configurations

- `locks/synthetic_paper_config.json` records the controlled synthetic setup.
- `locks/v4_gold_solver_lock.json` records the image-audit solver settings.
- `locks/prior_mnist_audit_v33_phases.csv` and
  `locks/prior_mnist_audit_v34_phases.csv` keep confirmatory samples separate
  from the earlier calibration audits.
- `locks/v5_experiment_config.json` records the block statistics, topology,
  seeds, damping, and noise settings added in release `v1.2.0-paper`.

Locked numerical tables and compact reported summaries are retained under
`reported_results/`. Downloaded datasets, caches, model checkpoints, and
per-step endpoint histories that are not needed to reproduce the reported
tables remain excluded from Git. They can be regenerated using the documented
commands, fixed configurations, and deterministic seeds.

## License

MIT License. See `LICENSE`.
