import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mnist_boundary_calibration_v33 import DampingCandidate
from mnist_rep_benchmark import ModelConfig, initialize_parameters, save_model
from mnist_tolerance_audit_v34 import ToleranceCandidate
from v4_dynamic_training import DynamicProfile, clip_gradients, train_seed_dynamic
from v4_locked_replication import load_gold_lock, prior_indices, run as run_locked_replication
from v4_train_image_models import cache_filename, load_image_dataset, parse_seeds


class Version4Tests(unittest.TestCase):
    def test_seed_parser_and_cache_names(self) -> None:
        self.assertEqual(parse_seeds("59,71,97"), (59, 71, 97))
        with self.assertRaises(ValueError):
            parse_seeds("17,17")
        self.assertEqual(cache_filename("mnist", 32), "mnist_pca_32.npz")
        self.assertEqual(
            cache_filename("fashion_mnist", 32), "fashion_mnist_pca_32.npz"
        )

    def test_prepared_cache_loads_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x_train = np.ones((12, 4))
            y_train = np.arange(12) % 10
            x_test = np.ones((8, 4))
            y_test = np.arange(8) % 10
            np.savez_compressed(
                root / "fashion_mnist_pca_3.npz",
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            loaded = load_image_dataset("fashion_mnist", root, 3)
            np.testing.assert_array_equal(loaded[0], x_train)
            np.testing.assert_array_equal(loaded[3], y_test)

    def test_gold_lock_and_prior_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "locked_before_v34_test_audit": True,
                        "locked_damping_candidate": {
                            "width": 2,
                            "trace": 3.0,
                            "power": 1.0,
                        },
                        "selected_tolerance": {"free": 1e-5, "nudged": 1e-6},
                    }
                ),
                encoding="utf-8",
            )
            damping, tolerance, digest = load_gold_lock(lock_path)
            self.assertEqual(damping, DampingCandidate(2, 3.0, 1.0))
            self.assertEqual(tolerance, ToleranceCandidate(1e-5, 1e-6))
            self.assertEqual(len(digest), 64)
            phase_path = root / "phases.csv"
            phase_path.write_text(
                'audit_indices_json\n"[1, 2, 3]"\n"[3, 4]"\n', encoding="utf-8"
            )
            excluded, hashes = prior_indices((phase_path,))
            self.assertEqual(excluded, {1, 2, 3, 4})
            self.assertIn("phases.csv", hashes)

    def test_gradient_clipping_preserves_direction(self) -> None:
        gradients = (np.asarray([3.0]), np.asarray([4.0]), np.asarray([0.0]))
        clipped, norm = clip_gradients(gradients, 2.5)
        self.assertEqual(norm, 5.0)
        np.testing.assert_allclose(clipped[0], np.asarray([1.5]))
        np.testing.assert_allclose(clipped[1], np.asarray([2.0]))

    def test_dynamic_training_and_locked_audit_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            rng = np.random.default_rng(81)
            x_train = rng.normal(scale=0.04, size=(20, 4))
            y_train = np.arange(20) % 10
            x_test = rng.normal(scale=0.04, size=(10, 4))
            y_test = np.arange(10)
            profile = DynamicProfile(5, 3, 1, 1, 4, 40, 20, 1e-4, 1e-2, 5.0)
            damping = DampingCandidate(2, 2.0, 1.0)
            tolerance = ToleranceCandidate(1e-3, 2e-4)
            params, history, result = train_seed_dynamic(
                81,
                profile,
                x_train,
                y_train,
                x_test,
                y_test,
                damping,
                tolerance,
                models,
            )
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(result["test_accuracy"]))
            config = ModelConfig(
                n_nodes=5,
                n_features=4,
                n_classes=10,
                n_input_nodes=2,
                boundary_width=2,
            )
            save_model(models / "mnist_model_seed_81.npz", params, config)
            cache = root / "cache.npz"
            np.savez_compressed(
                cache,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            lock = root / "gold.json"
            lock.write_text(
                json.dumps(
                    {
                        "locked_before_v34_test_audit": True,
                        "locked_damping_candidate": {
                            "width": 2,
                            "trace": 2.0,
                            "power": 1.0,
                        },
                        "selected_tolerance": {"free": 1e-3, "nudged": 2e-4},
                    }
                ),
                encoding="utf-8",
            )
            summary = run_locked_replication(
                "mnist",
                models,
                cache,
                lock,
                (81,),
                (),
                root / "audit",
                "quick",
            )
            self.assertEqual(summary["n_completed_seeds"], 1)
            self.assertTrue(summary["all_fresh_subsets_disjoint"])


if __name__ == "__main__":
    unittest.main()
