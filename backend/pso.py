"""
pso.py
======
Particle Swarm Optimization engine for traffic signal optimization using pso-final.py logic.

Algorithm Overview:
    Each particle encodes candidate signal plan factors for all network edges:
        position = [s_1, s_2, ..., s_E] ∈ [0, 1]^E

    Velocities:
        v_i = 0.5 * v_i + 1.5 * r1 * (pbest_i - position_i) + 1.5 * r2 * (gbest - position_i)
        position_i = clip(position_i + v_i, 0, 1)

    Traffic Redistribution (pso-final.py):
        redistributed = base_traffic * (1 - 0.3 * position)
        conserved_flow = (redistributed / sum(redistributed)) * sum(base_traffic)

    Fitness (pso-final.py):
        occs = conserved_flow / capacities
        peak_congestion = max(occs) * 100.0
        penalty = sum(max(occs - threshold_factor, 0))
        fitness = peak_congestion + penalty
"""

import numpy as np
from typing import Dict, List, Tuple

from graph import TrafficGraph
from predictions import (
    INITIAL_CONGESTION,
    INITIAL_CYCLE_TIMES,
)


class Particle:
    """A single particle in the PSO swarm representing signal plan factors."""

    def __init__(self, n_dims: int, rng: np.random.Generator) -> None:
        self.position: np.ndarray = rng.uniform(0.0, 1.0, n_dims)
        self.velocity: np.ndarray = np.zeros(n_dims)
        self.best_position: np.ndarray = self.position.copy()
        self.best_fitness: float = float("inf")

    def update_velocity(
        self, global_best_position: np.ndarray, w: float, c1: float, c2: float, rng: np.random.Generator
    ) -> None:
        n = len(self.position)
        r1 = rng.uniform(0.0, 1.0, n)
        r2 = rng.uniform(0.0, 1.0, n)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self) -> None:
        self.position = np.clip(self.position + self.velocity, 0.0, 1.0)


class PSO:
    def __init__(
        self,
        n_particles: int = 30,
        max_iter: int = 20,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        threshold_factor: float = 0.6,
        seed: int = 42,
    ) -> None:
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.threshold_factor = threshold_factor
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        graph: TrafficGraph,
        initial_congestion: Dict[str, float] | None = None,
        initial_cycle_times: Dict[str, float] | None = None,
    ) -> dict:

        if initial_congestion is None:
            initial_congestion = graph.get_initial_predictions()
        if initial_cycle_times is None:
            initial_cycle_times = INITIAL_CYCLE_TIMES.copy()

        edges = graph.edges
        num_edges = len(edges)

        base_traffic = np.array([initial_congestion.get(e.id, 0.0) for e in edges], dtype=float)
        capacities = np.array([e.capacity for e in edges], dtype=float)
        total_traffic = float(np.sum(base_traffic))

        def evaluate_particle(pos: np.ndarray) -> Tuple[float, Dict[str, float], Dict[str, float]]:
            redistributed = base_traffic * (1.0 - 0.3 * pos)
            if np.sum(redistributed) > 0 and total_traffic > 0:
                redistributed = (redistributed / np.sum(redistributed)) * total_traffic
            else:
                redistributed = base_traffic.copy()

            caps_safe = np.where(capacities > 0, capacities, 1.0)
            occs = np.where(capacities > 0, redistributed / caps_safe, 0.0)

            peak_congestion = float(np.max(occs)) * 100.0 if len(occs) > 0 else 0.0
            penalty = float(np.sum(np.maximum(occs - self.threshold_factor, 0.0)))
            fit = peak_congestion + penalty

            opt_congestion = {edges[i].id: float(redistributed[i]) for i in range(num_edges)}

            # Derive node cycle times bounded in [30, 120] seconds from signal plan
            opt_cycle_times = {}
            for node_id in graph.nodes:
                incident_indices = [
                    i for i, e in enumerate(edges) if e.target == node_id or e.source == node_id
                ]
                if incident_indices:
                    avg_signal = float(np.mean(pos[incident_indices]))
                else:
                    avg_signal = 0.5
                
                # Base cycle time modified by signal plan
                base_ct = initial_cycle_times.get(node_id, 60.0)
                ct_val = base_ct * (1.0 + 0.3 * (avg_signal - 0.5))
                opt_cycle_times[node_id] = float(np.clip(ct_val, 30.0, 120.0))

            return fit, opt_cycle_times, opt_congestion

        # Evaluate initial baseline fitness (signal plan = zeros or baseline)
        initial_fitness, _, _ = evaluate_particle(np.zeros(num_edges))

        # Initialise Swarm
        particles: List[Particle] = [Particle(num_edges, self.rng) for _ in range(self.n_particles)]

        global_best_fitness: float = float("inf")
        global_best_position: np.ndarray = particles[0].position.copy()

        # Initial evaluation
        for p in particles:
            fit, _, _ = evaluate_particle(p.position)
            p.best_fitness = fit
            p.best_position = p.position.copy()
            if fit < global_best_fitness:
                global_best_fitness = fit
                global_best_position = p.position.copy()

        fitness_history: List[float] = [initial_fitness]

        # PSO Main Loop
        for _iter in range(self.max_iter):
            for p in particles:
                p.update_velocity(global_best_position, self.w, self.c1, self.c2, self.rng)
                p.update_position()

                fit, _, _ = evaluate_particle(p.position)

                if fit < p.best_fitness:
                    p.best_fitness = fit
                    p.best_position = p.position.copy()

                if fit < global_best_fitness:
                    global_best_fitness = fit
                    global_best_position = p.position.copy()

            fitness_history.append(global_best_fitness)

        # Extract Best Results
        final_fitness, optimized_cycle_times, optimized_congestion = evaluate_particle(global_best_position)

        non_ref_edges = graph.get_non_reference_edges()
        desired_congestion = (
            sum(e.threshold for e in non_ref_edges) / len(non_ref_edges) if non_ref_edges else 0.0
        )

        return {
            "optimized_cycle_times": optimized_cycle_times,
            "optimized_congestion":  optimized_congestion,
            "fitness_history":       fitness_history,
            "initial_fitness":       initial_fitness,
            "final_fitness":         final_fitness,
            "iterations":            self.max_iter,
            "desired_congestion":    desired_congestion,
            "gbest_signal_plan":     global_best_position.tolist(),
        }

