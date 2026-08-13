import unittest

import numpy as np

from mnist_boundary_calibration_v33 import (
    DampingCandidate,
    calibration_selection_key,
    candidate_grid,
    damping_vector,
    pair_gradient,
    relax_phase,
)
from mnist_rep_benchmark import (
    ModelConfig,
    initialize_parameters,
    simplex_targets,
    state_gradient,
)


class MnistBoundaryV33Tests(unittest.TestCase):
    def test_candidates_are_boundary_local_and_trace_matched(self) -> None:
        for candidate in candidate_grid(17):
            damping = damping_vector(candidate, 17)
            self.assertAlmostEqual(float(np.sum(damping)), candidate.trace)
            self.assertEqual(np.count_nonzero(damping), candidate.width)
            self.assertEqual(float(np.sum(damping[: -candidate.width])), 0.0)

    def test_pair_gradient_matches_full_tensor_gradient(self) -> None:
        config = ModelConfig(
            n_nodes=7,
            n_features=5,
            n_classes=4,
            n_input_nodes=2,
            boundary_width=3,
        )
        params = initialize_parameters(config, 31)
        rng = np.random.default_rng(31)
        features = rng.normal(scale=0.2, size=(6, config.n_features))
        labels = rng.integers(0, config.n_classes, size=features.shape[0])
        targets = simplex_targets(labels, config.n_classes)
        q = rng.normal(scale=0.1, size=(6, config.n_classes, config.n_nodes))
        full = state_gradient(q, params, features, config, targets, config.beta)
        sample_indices = np.asarray([0, 1, 1, 4, 5])
        class_indices = np.asarray([2, 0, 3, 1, 2])
        selected = q[sample_indices, class_indices]
        paired = pair_gradient(
            selected,
            sample_indices,
            class_indices,
            params,
            features,
            config,
            targets,
            config.beta,
        )
        np.testing.assert_allclose(
            paired, full[sample_indices, class_indices], rtol=1e-13, atol=1e-13
        )

    def test_relaxation_records_every_budget(self) -> None:
        config = ModelConfig(
            n_nodes=5,
            n_features=4,
            n_classes=3,
            n_input_nodes=2,
            boundary_width=2,
        )
        params = initialize_parameters(config, 41)
        rng = np.random.default_rng(41)
        features = rng.normal(scale=0.05, size=(3, config.n_features))
        result = relax_phase(
            params,
            features,
            config,
            DampingCandidate(2, 2.0, 1.0),
            (10, 20, 30),
        )
        self.assertEqual(set(result["snapshots"]), {10, 20, 30})
        for snapshot in result["snapshots"].values():
            self.assertEqual(snapshot["q"].shape, (3, 3, 5))
            self.assertEqual(snapshot["converged"].shape, (3, 3))

    def test_selection_prefers_registered_pass(self) -> None:
        passed = {
            "finite_horizon_certificate_pass": True,
            "maximum_state_relative_error": 8e-5,
            "gradient_vs_exact_centered_relative_error": 8e-3,
            "gradient_vs_implicit_cosine": 0.9995,
            "minimum_phase_convergence_fraction": 0.95,
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
