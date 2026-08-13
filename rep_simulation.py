#!/usr/bin/env python3
"""Numerical proof-of-concept for Radiative Equilibrium Propagation (REP).

The script runs two complementary experiments:

1. A Robinson--Trautman-inspired spherical-harmonic relaxation model.  The
   first-order model is an exact gradient flow, while its inertial extension
   carries a news-like velocity field and obeys an augmented energy balance.
2. A local damped-wave model on a periodic one-dimensional domain.  It contrasts
   propagation and relaxation with ordinary diffusion and with an undamped
   wave, which cannot generically converge.

This is an effective mathematical model, not a numerical-relativity solver.
All quantities are dimensionless and the angular operator is normalized by its
ell=2 eigenvalue.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-rep"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import sph_harm_y


@dataclass(frozen=True)
class SphericalConfig:
    seed: int = 2026
    l_max: int = 8
    tau: float = 0.25
    kappa: float = 1.0
    t_end: float = 10.0
    n_times: int = 801


@dataclass(frozen=True)
class WaveConfig:
    n_x: int = 192
    c: float = 1.0
    gamma: float = 0.60
    mass: float = 0.35
    width: float = 0.22
    t_end: float = 18.0
    n_times: int = 601


def rt_eigenvalue(ell: int) -> float:
    """Positive spin-2/Calabi operator eigenvalue, normalized later."""
    if ell < 2:
        return 0.0
    return float((ell - 1) * ell * (ell + 1) * (ell + 2))


def real_spherical_harmonic(ell: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """A real orthonormal basis constructed from complex spherical harmonics."""
    if m == 0:
        return sph_harm_y(ell, 0, theta, phi).real
    y = sph_harm_y(ell, abs(m), theta, phi)
    if m > 0:
        return np.sqrt(2.0) * ((-1) ** m) * y.real
    return np.sqrt(2.0) * ((-1) ** abs(m)) * y.imag


def make_modes(config: SphericalConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ell, m, initial_amplitude) with deterministic random phases."""
    rng = np.random.default_rng(config.seed)
    ells: list[int] = []
    ms: list[int] = []
    amplitudes: list[float] = []
    for ell in range(2, config.l_max + 1):
        for m in range(-ell, ell + 1):
            ells.append(ell)
            ms.append(m)
            amplitudes.append(rng.normal() / (ell ** 1.5))
    a0 = np.asarray(amplitudes, dtype=float)
    a0 /= np.linalg.norm(a0)
    return np.asarray(ells), np.asarray(ms), a0


def spherical_experiment(config: SphericalConfig) -> dict[str, object]:
    """Run the exact gradient flow and inertial radiative relaxation."""
    ells, ms, a0 = make_modes(config)
    lam = np.asarray([rt_eigenvalue(int(ell)) for ell in ells]) / rt_eigenvalue(2)
    t = np.linspace(0.0, config.t_end, config.n_times)

    # Exact first-order Robinson--Trautman/Calabi gradient flow.
    a_gradient = a0[:, None] * np.exp(-config.kappa * lam[:, None] * t[None, :])
    j_gradient = 0.5 * config.kappa * np.sum(lam[:, None] * a_gradient**2, axis=0)
    gradient_rate = -config.kappa**2 * np.sum((lam[:, None] * a_gradient) ** 2, axis=0)

    # Inertial radiative extension: tau*a_ddot + a_dot + kappa*A*a = 0.
    n_modes = a0.size

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        a = y[:n_modes]
        news = y[n_modes : 2 * n_modes]
        d_news = (-news - config.kappa * lam * a) / config.tau
        d_dissipated = np.asarray([np.sum(news**2)])
        return np.concatenate((news, d_news, d_dissipated))

    y0 = np.concatenate((a0, np.zeros_like(a0), np.zeros(1)))
    solution = solve_ivp(
        rhs,
        (0.0, config.t_end),
        y0,
        t_eval=t,
        method="DOP853",
        rtol=2e-10,
        atol=2e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    a_rep = solution.y[:n_modes]
    news_rep = solution.y[n_modes : 2 * n_modes]
    potential_rep = 0.5 * config.kappa * np.sum(lam[:, None] * a_rep**2, axis=0)
    kinetic_rep = 0.5 * config.tau * np.sum(news_rep**2, axis=0)
    h_rep = potential_rep + kinetic_rep
    news_norm_sq = np.sum(news_rep**2, axis=0)
    dissipated = solution.y[-1]
    balance_curve = h_rep + dissipated - h_rep[0]
    relative_balance_error = float(np.max(np.abs(balance_curve)) / h_rep[0])
    monotonic_violations = int(np.count_nonzero(np.diff(h_rep) > 2e-10 * h_rep[0]))

    def first_threshold_time(curve: np.ndarray, fraction: float) -> float | None:
        indices = np.flatnonzero(curve <= fraction * curve[0])
        return float(t[indices[0]]) if indices.size else None

    metrics = {
        "n_modes": int(n_modes),
        "initial_augmented_energy": float(h_rep[0]),
        "final_augmented_energy": float(h_rep[-1]),
        "final_energy_fraction": float(h_rep[-1] / h_rep[0]),
        "relative_energy_balance_error": relative_balance_error,
        "monotonicity_violations": monotonic_violations,
        "rep_time_to_1e-4_energy": first_threshold_time(h_rep, 1e-4),
        "gradient_time_to_1e-4_energy": first_threshold_time(j_gradient, 1e-4),
        "final_news_norm": float(np.linalg.norm(news_rep[:, -1])),
        "final_rep_state_norm": float(np.linalg.norm(a_rep[:, -1])),
        "final_gradient_state_norm": float(np.linalg.norm(a_gradient[:, -1])),
        "gradient_identity_max_abs_error": float(
            np.max(
                np.abs(
                    np.sum(
                        config.kappa
                        * lam[:, None]
                        * a_gradient
                        * (-config.kappa * lam[:, None] * a_gradient),
                        axis=0,
                    )
                    - gradient_rate
                )
            )
        ),
    }

    return {
        "config": config,
        "t": t,
        "ells": ells,
        "ms": ms,
        "lam": lam,
        "a0": a0,
        "a_gradient": a_gradient,
        "j_gradient": j_gradient,
        "a_rep": a_rep,
        "news_rep": news_rep,
        "potential_rep": potential_rep,
        "kinetic_rep": kinetic_rep,
        "h_rep": h_rep,
        "news_norm_sq": news_norm_sq,
        "balance_curve": balance_curve,
        "metrics": metrics,
    }


def spectral_second_derivative(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.fft.ifft(-(k**2) * np.fft.fft(q)).real


def wave_experiment(config: WaveConfig) -> dict[str, object]:
    """Compare diffusion, damped propagation, and conservative propagation."""
    length = 2.0 * np.pi
    x = np.linspace(-np.pi, np.pi, config.n_x, endpoint=False)
    dx = length / config.n_x
    k = 2.0 * np.pi * np.fft.fftfreq(config.n_x, d=dx)
    t = np.linspace(0.0, config.t_end, config.n_times)
    q0 = np.exp(-0.5 * (x / config.width) ** 2)
    q0 -= 0.15 * np.exp(-0.5 * ((x - 0.65) / (1.3 * config.width)) ** 2)
    v0 = np.zeros_like(q0)

    stiffness = config.c**2 * k**2 + config.mass**2
    q_hat0 = np.fft.fft(q0)
    diffusion = np.asarray(
        [np.fft.ifft(q_hat0 * np.exp(-stiffness * ti)).real for ti in t]
    )

    def solve_wave(gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def rhs(_: float, y: np.ndarray) -> np.ndarray:
            q = y[: config.n_x]
            v = y[config.n_x : 2 * config.n_x]
            acceleration = (
                config.c**2 * spectral_second_derivative(q, k)
                - config.mass**2 * q
                - gamma * v
            )
            d_dissipated = np.asarray([gamma * np.mean(v**2)])
            return np.concatenate((v, acceleration, d_dissipated))

        sol = solve_ivp(
            rhs,
            (0.0, config.t_end),
            np.concatenate((q0, v0, np.zeros(1))),
            t_eval=t,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return (
            sol.y[: config.n_x].T,
            sol.y[config.n_x : 2 * config.n_x].T,
            sol.y[-1],
        )

    q_damped, v_damped, dissipated_damped = solve_wave(config.gamma)
    q_free, v_free, _ = solve_wave(0.0)

    def field_energy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        energy = []
        for qi, vi in zip(q, v, strict=True):
            q_x = np.fft.ifft(1j * k * np.fft.fft(qi)).real
            energy.append(
                0.5
                * np.mean(vi**2 + config.c**2 * q_x**2 + config.mass**2 * qi**2)
            )
        return np.asarray(energy)

    e_damped = field_energy(q_damped, v_damped)
    e_free = field_energy(q_free, v_free)
    balance_curve = e_damped + dissipated_damped - e_damped[0]

    metrics = {
        "damped_final_energy_fraction": float(e_damped[-1] / e_damped[0]),
        "undamped_final_energy_fraction": float(e_free[-1] / e_free[0]),
        "damped_relative_balance_error": float(
            np.max(np.abs(balance_curve)) / e_damped[0]
        ),
        "damped_monotonicity_violations": int(
            np.count_nonzero(np.diff(e_damped) > 2e-8 * e_damped[0])
        ),
        "diffusion_final_state_norm_fraction": float(
            np.linalg.norm(diffusion[-1]) / np.linalg.norm(diffusion[0])
        ),
        "damped_final_state_norm_fraction": float(
            np.linalg.norm(q_damped[-1]) / np.linalg.norm(q_damped[0])
        ),
        "undamped_final_state_norm_fraction": float(
            np.linalg.norm(q_free[-1]) / np.linalg.norm(q_free[0])
        ),
    }
    return {
        "config": config,
        "x": x,
        "t": t,
        "diffusion": diffusion,
        "q_damped": q_damped,
        "v_damped": v_damped,
        "q_free": q_free,
        "v_free": v_free,
        "e_damped": e_damped,
        "e_free": e_free,
        "balance_curve": balance_curve,
        "metrics": metrics,
    }


def reconstruct_field(
    amplitudes: np.ndarray,
    ells: np.ndarray,
    ms: np.ndarray,
    n_lon: int = 220,
    n_lat: int = 111,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.linspace(-np.pi, np.pi, n_lon)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    theta = np.pi / 2.0 - lat_grid
    field = np.zeros_like(theta)
    for coefficient, ell, m in zip(amplitudes, ells, ms, strict=True):
        field += coefficient * real_spherical_harmonic(int(ell), int(m), theta, lon_grid)
    return lon_grid, lat_grid, field


def plot_spherical_results(result: dict[str, object], output_dir: Path) -> None:
    t = result["t"]
    h_rep = result["h_rep"]
    j_gradient = result["j_gradient"]
    potential = result["potential_rep"]
    kinetic = result["kinetic_rep"]
    news_norm_sq = result["news_norm_sq"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].semilogy(t, j_gradient / j_gradient[0], label="RT gradient flow", lw=2.2)
    axes[0].semilogy(t, h_rep / h_rep[0], label="REP augmented energy", lw=2.2)
    axes[0].semilogy(t, potential / h_rep[0], label="REP potential", alpha=0.75)
    axes[0].set(xlabel="retarded/relaxation time", ylabel="normalized energy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].semilogy(t, np.maximum(kinetic / h_rep[0], 1e-16), label="wave kinetic term")
    axes[1].semilogy(
        t,
        np.maximum(news_norm_sq / h_rep[0], 1e-16),
        label=r"news norm $\|N\|^2$",
    )
    axes[1].set(xlabel="retarded/relaxation time", ylabel="normalized quantity")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Spherical radiative relaxation")
    fig.tight_layout()
    fig.savefig(output_dir / "spherical_energy_descent.png", dpi=190)
    plt.close(fig)

    ells = result["ells"]
    unique_ells = np.unique(ells)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ell in unique_ells:
        select = ells == ell
        rms_g = np.sqrt(np.mean(result["a_gradient"][select] ** 2, axis=0))
        rms_r = np.sqrt(np.mean(result["a_rep"][select] ** 2, axis=0))
        axes[0].semilogy(t, rms_g, label=fr"$\ell={ell}$")
        axes[1].semilogy(t, rms_r, label=fr"$\ell={ell}$")
    axes[0].set_title("RT gradient flow")
    axes[1].set_title("Radiative inertial flow")
    for ax in axes:
        ax.set_xlabel("time")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("modal RMS amplitude")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "spherical_mode_decay.png", dpi=190)
    plt.close(fig)

    snapshot_times = [0.0, 0.35, 1.5, 6.0]
    indices = [int(np.argmin(np.abs(t - value))) for value in snapshot_times]
    fields: list[np.ndarray] = []
    coordinates: tuple[np.ndarray, np.ndarray] | None = None
    for model_key in ("a_gradient", "a_rep"):
        for index in indices:
            lon, lat, field = reconstruct_field(
                result[model_key][:, index], result["ells"], result["ms"]
            )
            coordinates = (lon, lat)
            fields.append(field)
    limit = max(float(np.max(np.abs(field))) for field in fields)
    assert coordinates is not None
    lon, lat = coordinates
    fig, axes = plt.subplots(
        2, 4, figsize=(14.0, 6.2), subplot_kw={"projection": "mollweide"}
    )
    image = None
    for row, label in enumerate(("RT gradient", "REP wave-like")):
        for col, (index, time_value) in enumerate(zip(indices, snapshot_times, strict=True)):
            field = fields[row * 4 + col]
            image = axes[row, col].pcolormesh(
                lon, lat, field, shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit
            )
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_xticklabels([])
            axes[row, col].set_yticklabels([])
            axes[row, col].set_title(f"{label}\nt={time_value:g}", fontsize=9)
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.04, label="field amplitude")
    fig.suptitle("Angular deformation relaxing toward the zero-radiation equilibrium")
    fig.savefig(output_dir / "spherical_field_snapshots.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_wave_results(result: dict[str, object], output_dir: Path) -> None:
    t = result["t"]
    x = result["x"]
    heatmap_mask = t <= 8.0
    heatmap_t = t[heatmap_mask]
    extent = [float(x[0]), float(x[-1]), float(heatmap_t[-1]), float(heatmap_t[0])]
    limit = max(
        float(np.max(np.abs(result["diffusion"]))),
        float(np.max(np.abs(result["q_damped"]))),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    im0 = axes[0].imshow(
        result["diffusion"][heatmap_mask],
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[0].set_title("Gradient flow (diffusion)")
    axes[1].imshow(
        result["q_damped"][heatmap_mask],
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[1].set_title("REP (damped propagation)")
    axes[2].semilogy(t, result["e_damped"] / result["e_damped"][0], label="damped wave")
    axes[2].semilogy(t, result["e_free"] / result["e_free"][0], label="undamped control")
    axes[2].set(xlabel="time", ylabel="normalized field energy", title="Radiative loss is required")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    for ax in axes[:2]:
        ax.set(xlabel="position", ylabel="time")
    fig.colorbar(im0, ax=axes[:2].ravel().tolist(), shrink=0.78, pad=0.03, label="field amplitude")
    fig.savefig(output_dir / "local_propagation_and_relaxation.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_tables(spherical: dict[str, object], wave: dict[str, object], output_dir: Path) -> None:
    t = spherical["t"]
    spherical_table = pd.DataFrame(
        {
            "time": t,
            "rt_gradient_energy": spherical["j_gradient"],
            "rep_potential": spherical["potential_rep"],
            "rep_kinetic": spherical["kinetic_rep"],
            "rep_augmented_energy": spherical["h_rep"],
            "rep_news_norm_squared": spherical["news_norm_sq"],
            "rep_balance_residual": spherical["balance_curve"],
        }
    )
    spherical_table.to_csv(output_dir / "spherical_trajectory.csv", index=False)

    wave_table = pd.DataFrame(
        {
            "time": wave["t"],
            "damped_wave_energy": wave["e_damped"],
            "undamped_wave_energy": wave["e_free"],
            "damped_balance_residual": wave["balance_curve"],
        }
    )
    wave_table.to_csv(output_dir / "wave_energy_trajectory.csv", index=False)


def run(output_dir: Path, quick: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spherical_config = SphericalConfig(n_times=401 if quick else 801)
    wave_config = WaveConfig(n_x=96 if quick else 192, n_times=301 if quick else 601)
    spherical = spherical_experiment(spherical_config)
    wave = wave_experiment(wave_config)
    plot_spherical_results(spherical, output_dir)
    plot_wave_results(wave, output_dir)
    save_tables(spherical, wave, output_dir)
    summary = {
        "model_status": "effective proof-of-concept; not a numerical-relativity solver",
        "spherical_config": asdict(spherical_config),
        "wave_config": asdict(wave_config),
        "spherical_metrics": spherical["metrics"],
        "wave_metrics": wave["metrics"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="Use a smaller spatial/time grid")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result_summary = run(arguments.output_dir, quick=arguments.quick)
    print(json.dumps(result_summary, indent=2))
