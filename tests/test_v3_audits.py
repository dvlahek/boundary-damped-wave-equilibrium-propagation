import unittest

import numpy as np

from advanced_rep_experiments import damping_vector
from audit_energy_observability import blind_mode_control
from boundary_rep_learning import (
    flatten_gradients,
    parameter_energy_gradients,
    solve_equilibrium,
    state_hessian,
)
from theory_audit_core import (
    exact_gradient_triplet,
    fit_loglog_slope,
    integrate_energy_trajectory,
    make_audit_problem,
    modal_observability,
    parameter_map_segment_lipschitz_bound,
)


class REPv3TheoremAuditTests(unittest.TestCase):
    def test_single_terminal_observes_every_chain_mode(self) -> None:
        config, params, features, _ = make_audit_problem(
            9, 17, n_samples=6, boundary_variant="single_terminal"
        )
        equilibrium, _, _ = solve_equilibrium(params, features[:1], config)
        hessian = state_hessian(equilibrium, params, config)[0]
        damping = damping_vector(config, "boundary", params)
        modal = modal_observability(hessian, damping)
        self.assertEqual(np.count_nonzero(damping), 1)
        self.assertTrue(modal["observable"])
        self.assertTrue(modal["linearly_exponentially_stable"])

    def test_dark_mode_control_is_detected(self) -> None:
        controls, _ = blind_mode_control(t_end=20.0)
        blind = controls.loc[controls["control"] == "blind_center"].iloc[0]
        visible = controls.loc[controls["control"] == "visible_endpoint"].iloc[0]
        self.assertFalse(bool(blind["observable"]))
        self.assertGreater(float(blind["final_energy_fraction"]), 0.999)
        self.assertTrue(bool(visible["observable"]))
        self.assertLess(
            float(visible["final_energy_fraction"]),
            float(blind["final_energy_fraction"]),
        )

    def test_energy_balance_is_boundary_flux_identity(self) -> None:
        config, params, features, _ = make_audit_problem(
            3, 19, n_samples=5, boundary_variant="single_terminal"
        )
        result = integrate_energy_trajectory(
            params, features[:1], config, t_end=12.0, n_times=121
        )
        self.assertEqual(result["monotonicity_violations"], 0)
        self.assertLess(result["relative_balance_error"], 1e-6)

    def test_centered_gradient_has_second_order_scaling(self) -> None:
        config, params, features, targets = make_audit_problem(
            5, 23, n_samples=8
        )
        betas = np.asarray([0.08, 0.04, 0.02, 0.01])
        errors = np.asarray(
            [
                exact_gradient_triplet(
                    params, features, targets, config, float(beta)
                )["relative_error"]
                for beta in betas
            ]
        )
        slope, r_squared = fit_loglog_slope(betas, errors)
        self.assertGreater(slope, 1.7)
        self.assertLess(slope, 2.3)
        self.assertGreater(r_squared, 0.98)

    def test_parameter_map_segment_lipschitz_bound(self) -> None:
        config, params, features, _ = make_audit_problem(5, 31, n_samples=7)
        rng = np.random.default_rng(31)
        first = rng.normal(scale=0.3, size=(features.shape[0], config.n_nodes))
        second = rng.normal(scale=0.3, size=first.shape)
        first_map = flatten_gradients(
            parameter_energy_gradients(first, params, features, config)
        )
        second_map = flatten_gradients(
            parameter_energy_gradients(second, params, features, config)
        )
        lipschitz = parameter_map_segment_lipschitz_bound(
            params, features, first, second
        )
        self.assertLessEqual(
            np.linalg.norm(first_map - second_map),
            lipschitz * np.linalg.norm(first - second) + 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
