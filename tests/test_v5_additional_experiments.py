#!/usr/bin/env python3
"""Small deterministic tests for the V5 add-on experiments."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from graph_eqprop_core import (
    centered_gradients,
    flatten,
    implicit_gradients,
    initialize_parameters,
    make_config,
    relative_error,
    solve_equilibrium,
    state_hessian,
)
from statistical_refinement import analyze


class GraphCoreTests(unittest.TestCase):
    def test_hessian_is_symmetric_positive(self) -> None:
        config = make_config("grid_3", 17)
        params = initialize_parameters(config, 4, 29)
        q = np.zeros((2, config.n_nodes))
        hessian = state_hessian(q, params, config)
        self.assertTrue(np.allclose(hessian, np.swapaxes(hessian, 1, 2)))
        self.assertGreater(float(np.min(np.linalg.eigvalsh(hessian))), 0.0)

    def test_centered_gradient_matches_implicit_for_small_beta(self) -> None:
        config = make_config("sparse_9", 17, beta=1e-3)
        params = initialize_parameters(config, 5, 31)
        rng = np.random.default_rng(41)
        features = rng.normal(size=(6, 5))
        targets = np.where(features[:, 0] > 0.0, 1.0, -1.0)
        q_free, _, _ = solve_equilibrium(params, features, config)
        q_plus, _, _ = solve_equilibrium(
            params, features, config, targets, config.beta, initial=q_free
        )
        q_minus, _, _ = solve_equilibrium(
            params, features, config, targets, -config.beta, initial=q_free
        )
        centered = flatten(
            centered_gradients(q_minus, q_plus, params, features, config, config.beta)
        )
        implicit = flatten(implicit_gradients(q_free, params, features, targets, config))
        self.assertLess(relative_error(centered, implicit), 1e-5)


class StatisticalTests(unittest.TestCase):
    def test_block_equivalence_pipeline(self) -> None:
        rows = []
        for dataset in ("a", "b"):
            for seed in (1, 2, 3):
                for chain_size in (3, 5):
                    for method, offset in (
                        ("boundary_dynamic", 0.0),
                        ("implicit_id", 0.001),
                        ("standard_eqprop", 0.0),
                        ("uniform_dynamic", -0.001),
                    ):
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "chain_size": chain_size,
                                "method": method,
                                "test_accuracy": 0.8 + offset,
                                "wall_time_seconds": 2.0 if method == "boundary_dynamic" else 1.0,
                            }
                        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "runs.csv"
            output = Path(temporary) / "out"
            pd.DataFrame(rows).to_csv(source, index=False)
            summary = analyze(source, output, n_bootstrap=200, seed=5)
            self.assertEqual(len(summary), 3)
            self.assertTrue(summary["tost_equivalent_holm_0_05"].all())


if __name__ == "__main__":
    unittest.main()
