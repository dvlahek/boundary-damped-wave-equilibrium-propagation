import unittest

import numpy as np

from mnist_rep_benchmark import (
    ModelConfig,
    centered_physical_gradients,
    flatten_physical,
    implicit_physical_gradients,
    initialize_parameters,
    simplex_targets,
    solve_equilibrium,
    state_gradient,
)


class MnistRepTests(unittest.TestCase):
    def test_simplex_targets_are_centered_and_unique(self) -> None:
        labels = np.arange(5)
        targets = simplex_targets(labels, 5)
        np.testing.assert_allclose(np.mean(targets, axis=1), 0.0, atol=1e-14)
        np.testing.assert_array_equal(np.argmax(targets, axis=1), labels)

    def test_multiclass_state_gradient_matches_finite_difference(self) -> None:
        config = ModelConfig(
            n_nodes=5,
            n_features=4,
            n_classes=3,
            n_input_nodes=2,
            boundary_width=2,
            beta=0.01,
        )
        params = initialize_parameters(config, seed=7)
        rng = np.random.default_rng(7)
        features = rng.normal(scale=0.2, size=(2, config.n_features))
        targets = simplex_targets(np.asarray([0, 2]), config.n_classes)
        q = rng.normal(scale=0.1, size=(2, config.n_classes, config.n_nodes))

        def potential(value: np.ndarray) -> float:
            a = np.logaddexp(0.0, params.log_a) + config.stiffness_floor
            w = np.logaddexp(0.0, params.log_w) + config.edge_floor
            difference = value[:, :, 1:] - value[:, :, :-1]
            drive = np.einsum("bf,kif->bki", features, params.u)
            physical = (
                0.5 * np.sum(a[None, None, :] * value**2)
                + 0.25 * config.alpha * np.sum(value**4)
                + 0.5 * np.sum(w[None, None, :] * difference**2)
                - np.sum(value[:, :, : config.n_input_nodes] * drive)
            )
            cost = 0.5 * np.sum((value[:, :, -1] - targets) ** 2)
            return float(physical + config.beta * cost)

        analytic = state_gradient(q, params, features, config, targets, config.beta)
        numeric = np.zeros_like(q)
        step = 2e-6
        for sample in range(q.shape[0]):
            for channel in range(q.shape[1]):
                for node in range(q.shape[2]):
                    plus = q.copy()
                    minus = q.copy()
                    plus[sample, channel, node] += step
                    minus[sample, channel, node] -= step
                    numeric[sample, channel, node] = (
                        potential(plus) - potential(minus)
                    ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=3e-7, atol=3e-8)

    def test_centered_multiclass_gradient_matches_implicit_gradient(self) -> None:
        config = ModelConfig(
            n_nodes=7,
            n_features=6,
            n_classes=4,
            n_input_nodes=2,
            boundary_width=3,
            beta=0.002,
        )
        params = initialize_parameters(config, seed=19)
        rng = np.random.default_rng(19)
        features = rng.normal(scale=0.2, size=(12, config.n_features))
        labels = rng.integers(0, config.n_classes, size=features.shape[0])
        targets = simplex_targets(labels, config.n_classes)
        q_free, _, _ = solve_equilibrium(params, features, config)
        q_plus, _, _ = solve_equilibrium(
            params, features, config, targets, config.beta, q_free
        )
        q_minus, _, _ = solve_equilibrium(
            params, features, config, targets, -config.beta, q_free
        )
        centered = flatten_physical(
            centered_physical_gradients(q_minus, q_plus, params, features, config.beta)
        )
        implicit = flatten_physical(
            implicit_physical_gradients(q_free, params, features, targets, config)
        )
        relative_error = np.linalg.norm(centered - implicit) / np.linalg.norm(implicit)
        cosine = float(
            np.dot(centered, implicit)
            / (np.linalg.norm(centered) * np.linalg.norm(implicit))
        )
        self.assertLess(relative_error, 2e-4)
        self.assertGreater(cosine, 0.99999)


if __name__ == "__main__":
    unittest.main()
