#!/usr/bin/env python3
"""Shared graph-valued equilibrium-propagation utilities.

The original experiments use one-dimensional chains.  This module implements
the same strictly convex onsite-plus-edge energy on an arbitrary undirected
graph so that the boundary-damping claim can be audited without relying on a
chain topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np
from scipy.linalg import eigh


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


@dataclass(frozen=True)
class GraphConfig:
    n_nodes: int
    edges: np.ndarray
    input_nodes: np.ndarray
    output_nodes: np.ndarray
    alpha: float = 0.08
    stiffness_floor: float = 0.25
    edge_floor: float = 0.35
    # A light matched trace avoids both the weakly damped and overdamped
    # branches of the graph spectrum in the registered topology audit.
    damping_trace: float = 0.10
    beta: float = 0.035


@dataclass
class GraphParameters:
    log_a: np.ndarray
    log_w: np.ndarray
    u: np.ndarray


def grid_graph(side: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Square grid with the left column as input and right column as output."""
    if side < 2:
        raise ValueError("side must be at least 2")
    edges: list[tuple[int, int]] = []
    for row in range(side):
        for column in range(side):
            node = row * side + column
            if column + 1 < side:
                edges.append((node, node + 1))
            if row + 1 < side:
                edges.append((node, node + side))
    inputs = np.arange(0, side * side, side, dtype=int)
    outputs = inputs + side - 1
    return side * side, np.asarray(edges, dtype=int), inputs, outputs


def sparse_connected_graph(
    n_nodes: int, seed: int, extra_edge_probability: float = 0.16
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic connected sparse graph with separated input/output sets."""
    if n_nodes < 6:
        raise ValueError("n_nodes must be at least 6")
    rng = np.random.default_rng(seed)
    edge_set: set[tuple[int, int]] = set()
    # A random recursive tree guarantees connectivity.
    order = rng.permutation(n_nodes)
    for position in range(1, n_nodes):
        child = int(order[position])
        parent = int(order[rng.integers(0, position)])
        edge_set.add(tuple(sorted((child, parent))))
    for first in range(n_nodes):
        for second in range(first + 1, n_nodes):
            if (first, second) not in edge_set and rng.random() < extra_edge_probability:
                edge_set.add((first, second))

    edges = np.asarray(sorted(edge_set), dtype=int)
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for first, second in edges:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))

    root = int(order[0])
    distance = np.full(n_nodes, -1, dtype=int)
    distance[root] = 0
    queue = [root]
    for node in queue:
        for neighbor in adjacency[node]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    # A graph boundary should remain low-rank but must not be reduced to an
    # accidentally invisible single vertex.  The square-root rule matches the
    # side length used as the output boundary of an equally sized 2-D grid.
    width = max(3, int(np.ceil(np.sqrt(n_nodes))))
    inputs = np.argsort(distance, kind="stable")[:width]
    outputs = np.argsort(distance, kind="stable")[-width:]
    if np.intersect1d(inputs, outputs).size:
        raise RuntimeError("input and output node sets must be disjoint")
    return n_nodes, edges, inputs.astype(int), outputs.astype(int)


def make_config(topology: str, seed: int, beta: float = 0.035) -> GraphConfig:
    if topology.startswith("grid_"):
        side = int(topology.split("_")[1])
        n_nodes, edges, inputs, outputs = grid_graph(side)
    elif topology.startswith("sparse_"):
        n_nodes = int(topology.split("_")[1])
        n_nodes, edges, inputs, outputs = sparse_connected_graph(n_nodes, seed)
    else:
        raise ValueError(f"unknown topology: {topology}")
    return GraphConfig(n_nodes, edges, inputs, outputs, beta=beta)


def initialize_parameters(
    config: GraphConfig, n_features: int, seed: int
) -> GraphParameters:
    rng = np.random.default_rng(seed)
    log_a = np.full(
        config.n_nodes, inverse_softplus(0.90 - config.stiffness_floor)
    )
    log_a += rng.normal(scale=0.07, size=config.n_nodes)
    log_w = np.full(
        len(config.edges), inverse_softplus(0.70 - config.edge_floor)
    )
    # Small heterogeneous edge weights prevent accidental graph symmetries.
    log_w += rng.normal(scale=0.09, size=len(config.edges))
    u = rng.normal(scale=0.18, size=(len(config.input_nodes), n_features))
    return GraphParameters(log_a, log_w, u)


def select_modal_boundary(
    config: GraphConfig,
    params: GraphParameters,
    n_boundary_nodes: int | None = None,
    q_states: np.ndarray | None = None,
) -> GraphConfig:
    """Select a low-rank output boundary that maximizes worst modal coverage.

    A sparse graph need not have a unique geometric outer face.  We therefore
    define its accessible output boundary by a deterministic sensor-placement
    problem on the conservative linearization.  Grid outputs remain geometric
    and do not use this helper.
    """
    width = (
        len(config.output_nodes)
        if n_boundary_nodes is None
        else int(n_boundary_nodes)
    )
    candidates = [
        node for node in range(config.n_nodes) if node not in set(config.input_nodes)
    ]
    if width <= 0 or width > len(candidates):
        raise ValueError("invalid number of boundary nodes")
    audited_states = (
        np.zeros((1, config.n_nodes))
        if q_states is None
        else np.asarray(q_states, dtype=float)
    )
    hessians = state_hessian(audited_states, params, config)
    # squared[sample, node, mode]
    squared = np.stack(
        [np.linalg.eigh(hessian)[1] ** 2 for hessian in hessians], axis=0
    )

    selected: list[int] = []
    coverage = np.zeros((len(audited_states), config.n_nodes))
    for _ in range(width):
        best_node = max(
            (node for node in candidates if node not in selected),
            key=lambda node: (
                float(np.min(coverage + squared[:, node, :])),
                -node,
            ),
        )
        selected.append(best_node)
        coverage += squared[:, best_node, :]

    # Greedy placement is followed by deterministic single-node swaps until
    # no swap improves the minimum modal visibility.
    improved = True
    while improved:
        improved = False
        current_objective = float(
            np.min(np.sum(squared[:, selected, :], axis=1))
        )
        for position, old_node in enumerate(tuple(selected)):
            for new_node in candidates:
                if new_node in selected:
                    continue
                proposal = selected.copy()
                proposal[position] = new_node
                objective = float(
                    np.min(np.sum(squared[:, proposal, :], axis=1))
                )
                if objective > current_objective + 1e-15:
                    selected = proposal
                    current_objective = objective
                    improved = True
    return replace(config, output_nodes=np.asarray(sorted(selected), dtype=int))


def physical_coefficients(
    params: GraphParameters, config: GraphConfig
) -> tuple[np.ndarray, np.ndarray]:
    return (
        softplus(params.log_a) + config.stiffness_floor,
        softplus(params.log_w) + config.edge_floor,
    )


def input_force(
    params: GraphParameters, features: np.ndarray, config: GraphConfig
) -> np.ndarray:
    force = np.zeros((features.shape[0], config.n_nodes))
    force[:, config.input_nodes] = features @ params.u.T
    return force


def state_gradient(
    q: np.ndarray,
    params: GraphParameters,
    features: np.ndarray,
    config: GraphConfig,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
) -> np.ndarray:
    onsite, edge_weight = physical_coefficients(params, config)
    gradient = onsite[None, :] * q + config.alpha * q**3
    gradient -= input_force(params, features, config)
    first = config.edges[:, 0]
    second = config.edges[:, 1]
    difference = q[:, first] - q[:, second]
    for edge_index, (node_first, node_second) in enumerate(config.edges):
        contribution = edge_weight[edge_index] * difference[:, edge_index]
        gradient[:, node_first] += contribution
        gradient[:, node_second] -= contribution
    if beta != 0.0:
        if targets is None:
            raise ValueError("targets are required for a nudged equilibrium")
        scale = beta / len(config.output_nodes)
        gradient[:, config.output_nodes] += scale * (
            q[:, config.output_nodes] - targets[:, None]
        )
    return gradient


def state_hessian(
    q: np.ndarray,
    params: GraphParameters,
    config: GraphConfig,
    beta: float = 0.0,
) -> np.ndarray:
    onsite, edge_weight = physical_coefficients(params, config)
    batch = q.shape[0]
    hessian = np.zeros((batch, config.n_nodes, config.n_nodes))
    diagonal = onsite[None, :] + 3.0 * config.alpha * q**2
    if beta != 0.0:
        diagonal[:, config.output_nodes] += beta / len(config.output_nodes)
    indices = np.arange(config.n_nodes)
    hessian[:, indices, indices] = diagonal
    for edge_index, (node_first, node_second) in enumerate(config.edges):
        weight = edge_weight[edge_index]
        hessian[:, node_first, node_first] += weight
        hessian[:, node_second, node_second] += weight
        hessian[:, node_first, node_second] -= weight
        hessian[:, node_second, node_first] -= weight
    return hessian


def solve_equilibrium(
    params: GraphParameters,
    features: np.ndarray,
    config: GraphConfig,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    initial: np.ndarray | None = None,
    tolerance: float = 2e-11,
    max_iterations: int = 70,
) -> tuple[np.ndarray, int, float]:
    q = (
        np.zeros((features.shape[0], config.n_nodes))
        if initial is None
        else initial.copy()
    )
    for iteration in range(1, max_iterations + 1):
        gradient = state_gradient(q, params, features, config, targets, beta)
        residual = float(np.max(np.linalg.norm(gradient, axis=1)))
        if residual <= tolerance:
            return q, iteration - 1, residual
        hessian = state_hessian(q, params, config, beta)
        step = np.linalg.solve(hessian, gradient[..., None])[..., 0]
        q -= (0.70 if iteration <= 3 else 1.0) * step
    raise RuntimeError(f"equilibrium solver failed; residual={residual:.3e}")


def parameter_energy_gradients(
    q: np.ndarray,
    params: GraphParameters,
    features: np.ndarray,
    config: GraphConfig,
) -> GraphParameters:
    first = config.edges[:, 0]
    second = config.edges[:, 1]
    difference = q[:, first] - q[:, second]
    return GraphParameters(
        sigmoid(params.log_a) * np.mean(0.5 * q**2, axis=0),
        sigmoid(params.log_w) * np.mean(0.5 * difference**2, axis=0),
        -np.mean(q[:, config.input_nodes, None] * features[:, None, :], axis=0),
    )


def centered_gradients(
    q_minus: np.ndarray,
    q_plus: np.ndarray,
    params: GraphParameters,
    features: np.ndarray,
    config: GraphConfig,
    beta: float,
) -> GraphParameters:
    minus = parameter_energy_gradients(q_minus, params, features, config)
    plus = parameter_energy_gradients(q_plus, params, features, config)
    return GraphParameters(
        (plus.log_a - minus.log_a) / (2.0 * beta),
        (plus.log_w - minus.log_w) / (2.0 * beta),
        (plus.u - minus.u) / (2.0 * beta),
    )


def implicit_gradients(
    q_free: np.ndarray,
    params: GraphParameters,
    features: np.ndarray,
    targets: np.ndarray,
    config: GraphConfig,
) -> GraphParameters:
    hessian = state_hessian(q_free, params, config)
    cost_gradient = np.zeros_like(q_free)
    cost_gradient[:, config.output_nodes] = (
        q_free[:, config.output_nodes] - targets[:, None]
    ) / len(config.output_nodes)
    adjoint = np.linalg.solve(hessian, cost_gradient[..., None])[..., 0]
    first = config.edges[:, 0]
    second = config.edges[:, 1]
    q_difference = q_free[:, first] - q_free[:, second]
    adjoint_difference = adjoint[:, first] - adjoint[:, second]
    return GraphParameters(
        -sigmoid(params.log_a) * np.mean(adjoint * q_free, axis=0),
        -sigmoid(params.log_w)
        * np.mean(adjoint_difference * q_difference, axis=0),
        np.mean(
            adjoint[:, config.input_nodes, None] * features[:, None, :], axis=0
        ),
    )


def flatten(gradient: GraphParameters) -> np.ndarray:
    return np.concatenate(
        (gradient.log_a.ravel(), gradient.log_w.ravel(), gradient.u.ravel())
    )


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - reference)
        / max(float(np.linalg.norm(reference)), 1e-14)
    )


def cosine_similarity(estimate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(estimate) * np.linalg.norm(reference))
    if denominator <= 1e-14:
        return 1.0 if np.linalg.norm(estimate - reference) <= 1e-12 else 0.0
    return float(np.dot(estimate, reference) / denominator)


def damping_vector(config: GraphConfig, mode: str) -> np.ndarray:
    boundary = np.zeros(config.n_nodes)
    boundary[config.output_nodes] = config.damping_trace / len(config.output_nodes)
    if mode == "boundary":
        return boundary
    if mode == "uniform_trace":
        return np.full(config.n_nodes, config.damping_trace / config.n_nodes)
    raise ValueError(f"unknown damping mode: {mode}")


def relax_dynamics(
    params: GraphParameters,
    features: np.ndarray,
    config: GraphConfig,
    *,
    targets: np.ndarray | None = None,
    beta: float = 0.0,
    damping_mode: str = "boundary",
    initial_q: np.ndarray | None = None,
    dt: float = 0.12,
    max_steps: int = 80_000,
    tolerance: float = 1e-6,
    check_every: int = 25,
    consecutive_checks: int = 3,
) -> dict[str, object]:
    batch = features.shape[0]
    q = (
        np.zeros((batch, config.n_nodes))
        if initial_q is None
        else initial_q.copy()
    )
    velocity = np.zeros_like(q)
    damping = damping_vector(config, damping_mode)
    half_decay = np.exp(-0.5 * dt * damping)[None, :]
    consecutive = np.zeros(batch, dtype=int)
    converged_samples = np.zeros(batch, dtype=bool)
    sample_steps = np.zeros(batch, dtype=int)
    active_sample_steps = 0

    for step in range(1, max_steps + 1):
        active_indices = np.flatnonzero(~converged_samples)
        if active_indices.size == 0:
            break
        q_active = q[active_indices]
        velocity_active = velocity[active_indices] * half_decay
        features_active = features[active_indices]
        targets_active = None if targets is None else targets[active_indices]
        gradient = state_gradient(
            q_active, params, features_active, config, targets_active, beta
        )
        velocity_active -= 0.5 * dt * gradient
        q_active += dt * velocity_active
        gradient_new = state_gradient(
            q_active, params, features_active, config, targets_active, beta
        )
        velocity_active -= 0.5 * dt * gradient_new
        velocity_active *= half_decay
        q[active_indices] = q_active
        velocity[active_indices] = velocity_active
        sample_steps[active_indices] += 1
        active_sample_steps += int(active_indices.size)
        if step % check_every == 0:
            local = (
                (np.linalg.norm(gradient_new, axis=1) <= tolerance)
                & (np.linalg.norm(velocity_active, axis=1) <= tolerance)
            )
            consecutive[active_indices] = np.where(
                local, consecutive[active_indices] + 1, 0
            )
            newly_converged = active_indices[
                consecutive[active_indices] >= consecutive_checks
            ]
            converged_samples[newly_converged] = True

    final_gradient = state_gradient(q, params, features, config, targets, beta)
    return {
        "q": q,
        "velocity": velocity,
        "steps": int(step),
        "sample_steps": sample_steps,
        "active_sample_steps": int(active_sample_steps),
        "converged": bool(np.all(converged_samples)),
        "convergence_fraction": float(np.mean(converged_samples)),
        "residual": float(np.max(np.linalg.norm(final_gradient, axis=1))),
        "velocity_norm": float(np.max(np.linalg.norm(velocity, axis=1))),
    }


def modal_observability(hessian: np.ndarray, damping: np.ndarray) -> dict[str, float | bool]:
    eigenvalues, eigenvectors = eigh(hessian)
    selector = np.zeros((int(np.count_nonzero(damping)), len(damping)))
    selected = np.flatnonzero(damping > 0.0)
    selector[np.arange(len(selected)), selected] = 1.0
    margins: list[float] = []
    start = 0
    while start < len(eigenvalues):
        stop = start + 1
        scale = max(1.0, abs(float(eigenvalues[start])))
        while (
            stop < len(eigenvalues)
            and abs(float(eigenvalues[stop] - eigenvalues[start])) <= 1e-8 * scale
        ):
            stop += 1
        block = selector @ eigenvectors[:, start:stop]
        if block.shape[0] < block.shape[1]:
            margin = 0.0
        else:
            singular_values = np.linalg.svd(block, compute_uv=False)
            margin = float(np.min(singular_values)) if singular_values.size else 0.0
        margins.append(margin)
        start = stop
    minimum_margin = float(min(margins))
    tolerance = float(100.0 * np.finfo(float).eps * max(1.0, np.sqrt(len(damping))))
    system = np.block(
        [
            [np.zeros_like(hessian), np.eye(len(damping))],
            [-hessian, -np.diag(damping)],
        ]
    )
    spectral_abscissa = float(np.max(np.linalg.eigvals(system).real))
    return {
        "minimum_eigenspace_visibility": minimum_margin,
        "observable": bool(minimum_margin > tolerance),
        "minimum_hessian_eigenvalue": float(np.min(eigenvalues)),
        "spectral_abscissa": spectral_abscissa,
        "spectral_decay_rate": max(0.0, -spectral_abscissa),
    }
