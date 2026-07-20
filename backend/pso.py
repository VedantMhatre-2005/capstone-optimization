"""
pso.py
======
Particle Swarm Optimization engine for traffic signal cycle time optimization.

Algorithm Overview:
    Each particle encodes a candidate signal timing plan:
        position = [C_A, C_B, C_C, C_D, C_E]  (cycle times in seconds)

    The PSO searches the bounded space [30 s, 120 s]^5 to find cycle times
    that minimize the objective function J (weighted congestion variance).
"""

import numpy as np
from typing import Dict, List, Tuple

from graph import TrafficGraph
from objective import compute_fitness
from traffic_redistributor import redistribute_traffic
from predictions import (
    INITIAL_CONGESTION,
    INITIAL_CYCLE_TIMES,
    CYCLE_TIME_MIN,
    CYCLE_TIME_MAX,
)

# ---------------------------------------------------------------------------
# PSO Particle Definition
# ---------------------------------------------------------------------------

class Particle:
    """A single particle in the PSO swarm representing signal timings."""

    def __init__(self, n_dims: int, bounds: Tuple[float, float], rng: np.random.Generator) -> None:
        lo, hi = bounds
        self.position: np.ndarray = rng.uniform(lo, hi, n_dims)
        span = hi - lo
        self.velocity: np.ndarray = rng.uniform(-span * 0.1, span * 0.1, n_dims)
        self.best_position: np.ndarray = self.position.copy()
        self.best_fitness: float = float("inf")

    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float, rng: np.random.Generator) -> None:
        n = len(self.position)
        r1 = rng.uniform(0.0, 1.0, n)
        r2 = rng.uniform(0.0, 1.0, n)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social    = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds: Tuple[float, float]) -> None:
        self.position = self.position + self.velocity
        lo, hi = bounds
        for i in range(len(self.position)):
            if self.position[i] < lo:
                self.position[i] = lo
                self.velocity[i] = abs(self.velocity[i])  # bounce inward
            elif self.position[i] > hi:
                self.position[i] = hi
                self.velocity[i] = -abs(self.velocity[i])  # bounce inward


# ---------------------------------------------------------------------------
# PSO Optimizer
# ---------------------------------------------------------------------------

class PSO:
    def __init__(
        self,
        n_particles: int = 100,
        max_iter: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
    ) -> None:
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.w           = w
        self.c1          = c1
        self.c2          = c2
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        graph: TrafficGraph,
        initial_congestion: Dict[str, float] | None = None,
        initial_cycle_times: Dict[str, float] | None = None,
    ) -> dict:
        
        if initial_congestion is None:
            initial_congestion = INITIAL_CONGESTION.copy()
        if initial_cycle_times is None:
            initial_cycle_times = INITIAL_CYCLE_TIMES.copy()

        node_order: List[str] = list(graph.nodes.keys())
        n_dims = len(node_order)
        bounds: Tuple[float, float] = (CYCLE_TIME_MIN, CYCLE_TIME_MAX)

        # Baseline threshold & capacities dicts for redistributor
        thresholds = {e.id: e.threshold for e in graph.edges}
        capacities = {e.id: e.capacity for e in graph.edges}

        # Evaluate initial fitness
        initial_fitness = compute_fitness(initial_congestion, graph)

        # ------------------------------------------------------------------
        # Initialise Swarm
        # ------------------------------------------------------------------        # Initialise Swarm
        particles: List[Particle] = []
        init_pos = np.array([initial_cycle_times[n] for n in node_order])
        for i in range(self.n_particles):
            p = Particle(n_dims, bounds, self.rng)
            if i == 0:
                p.position = init_pos.copy()
            else:
                # Gaussian perturbation around initial position for swarm dispersion
                noise = self.rng.normal(0, 15.0, n_dims)
                p.position = np.clip(init_pos + noise, bounds[0], bounds[1])
            p.best_position = p.position.copy()
            particles.append(p)

        global_best_fitness: float = float("inf")
        global_best_position: np.ndarray = np.array([initial_cycle_times[n] for n in node_order])
        
        # ------------------------------------------------------------------
        # Helper for particle evaluation
        # ------------------------------------------------------------------
        def evaluate_particle(pos: np.ndarray) -> Tuple[float, Dict[str, float], Dict[str, float]]:
            proposed_ct = dict(zip(node_order, pos))
            # Redistribute traffic
            q_new = redistribute_traffic(graph, initial_congestion, proposed_ct, thresholds, capacities)
            # Compute fitness
            fit = compute_fitness(q_new, graph)
            return fit, proposed_ct, q_new

        # Initial evaluation
        for p in particles:
            fit, c_new, q_new = evaluate_particle(p.position)
            p.best_fitness = fit
            p.best_position = p.position.copy()
            if fit < global_best_fitness:
                global_best_fitness = fit
                global_best_position = p.position.copy()

        # ------------------------------------------------------------------
        # PSO Main Loop
        # ------------------------------------------------------------------
        fitness_history: List[float] = [initial_fitness]

        for _iter in range(self.max_iter):
            for p in particles:
                p.update_velocity(global_best_position, self.w, self.c1, self.c2, self.rng)
                p.update_position(bounds)

                fit, _, _ = evaluate_particle(p.position)

                if fit < p.best_fitness:
                    p.best_fitness = fit
                    p.best_position = p.position.copy()

                if fit < global_best_fitness:
                    global_best_fitness = fit
                    global_best_position = p.position.copy()

            fitness_history.append(global_best_fitness)

        # ------------------------------------------------------------------
        # Extract Best Results
        # ------------------------------------------------------------------
        _, optimized_cycle_times, optimized_congestion = evaluate_particle(global_best_position)

        # Compute desired_congestion (Q*)
        non_ref_edges = graph.get_non_reference_edges()
        desired_congestion = sum(e.threshold for e in non_ref_edges) / len(non_ref_edges) if non_ref_edges else 0.0

        return {
            "optimized_cycle_times": optimized_cycle_times,
            "optimized_congestion":  optimized_congestion,
            "fitness_history":       fitness_history,
            "initial_fitness":       initial_fitness,
            "final_fitness":         global_best_fitness,
            "iterations":            self.max_iter,
            "desired_congestion":    desired_congestion,
        }
