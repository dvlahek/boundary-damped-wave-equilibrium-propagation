import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mnist_boundary_calibration_v33 import DampingCandidate
from mnist_rep_benchmark import ModelConfig, initialize_parameters
from mnist_tolerance_audit_v34 import (
    ToleranceCandidate,
    calibration_selection_key,
    gradient_change,
    load_locked_damping,
    load_v33_audit_indices,
    relax_phase,
    run_locked_audit,
    tolerance_grid,
)


class MnistToleranceV34Tests(unittest.TestCase):
    def test_registered_tolerance_grid_is_monotone(self) -> None:
        grid = tolerance_grid()
        self.assertEqual(grid[0], ToleranceCandidate(1e-4, 2e-5))
        self.assertTrue(all(a.free >= b.free for a, b in zip(grid, grid[1:])))
        self.assertTrue(all(a.nudged > b.nudged for a, b in zip(grid, grid[1:])))

    def test_v33_lock_is_required_and_validated(self) -> None:
        payload = {
            "locked_before_test_audit": True,
            "selected_candidate": {"width": 3, "trace": 5.0, "power": 1.0},
            "selected_candidate_key": "width=3|trace=5|power=1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            candidate, lock, digest = load_locked_damping(path)
            self.assertEqual(candidate, DampingCandidate(3, 5.0, 1.0))
            self.assertTrue(lock["locked_before_test_audit"])
            self.assertEqual(len(digest), 64)
            payload["locked_before_test_audit"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_locked_damping(path)

    def test_gradient_change_is_scale_aware(self) -> None:
        previous = np.asarray([1.0, 2.0, 3.0])
        self.assertEqual(gradient_change(previous, previous), 0.0)
        current = previous * 1.001
        self.assertAlmostEqual(gradient_change(current, previous), 0.001 / 1.001)

    def test_v33_audit_indices_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phases.csv"
            path.write_text(
                'seed,audit_indices_json\n17,"[1, 2, 3]"\n17,"[1, 2, 3]"\n29,"[4, 5]"\n',
                encoding="utf-8",
            )
            indices, digest = load_v33_audit_indices(path)
            self.assertEqual(indices, {17: {1, 2, 3}, 29: {4, 5}})
            self.assertEqual(len(digest), 64)

    def test_relaxation_accepts_explicit_tolerance_and_records_budgets(self) -> None:
        config = ModelConfig(
            n_nodes=5,
            n_features=4,
            n_classes=3,
            n_input_nodes=2,
            boundary_width=2,
        )
        params = initialize_parameters(config, 51)
        rng = np.random.default_rng(51)
        features = rng.normal(scale=0.04, size=(3, config.n_features))
        result = relax_phase(
            params,
            features,
            config,
            DampingCandidate(2, 2.0, 1.0),
            (10, 20, 30),
            1e-5,
        )
        self.assertEqual(set(result["snapshots"]), {10, 20, 30})
        self.assertEqual(result["tolerance"], 1e-5)

    def test_audit_reports_gradient_stability_after_first_budget(self) -> None:
        config = ModelConfig(
            n_nodes=5,
            n_features=4,
            n_classes=3,
            n_input_nodes=2,
            boundary_width=2,
        )
        params = initialize_parameters(config, 61)
        rng = np.random.default_rng(61)
        features = rng.normal(scale=0.03, size=(2, config.n_features))
        labels = np.asarray([0, 2])
        phases, curves, method = run_locked_audit(
            61,
            params,
            config,
            features,
            labels,
            DampingCandidate(2, 2.0, 1.0),
            ToleranceCandidate(1e-3, 2e-4),
            (20, 40),
            confirmatory_model_seed=False,
        )
        self.assertEqual(len(phases), 6)
        self.assertTrue(np.isnan(curves[0]["gradient_change_from_previous_budget"]))
        self.assertTrue(np.isfinite(curves[1]["gradient_change_from_previous_budget"]))
        self.assertEqual(method["budget"], 40)

    def test_selection_prefers_complete_registered_pass(self) -> None:
        passed = {
            "finite_horizon_certificate_pass": True,
            "strict_all_phases_converged": True,
            "maximum_state_relative_error": 8e-5,
            "gradient_vs_exact_centered_relative_error": 8e-3,
            "gradient_vs_implicit_cosine": 0.9995,
            "gradient_change_from_previous_budget": 1e-3,
            "minimum_phase_convergence_fraction": 1.0,
            "total_active_channel_steps": 1000,
        }
        failed = dict(passed)
        failed.update(
            finite_horizon_certificate_pass=False,
            gradient_vs_exact_centered_relative_error=0.011,
        )
        self.assertLess(calibration_selection_key(passed), calibration_selection_key(failed))


if __name__ == "__main__":
    unittest.main()
