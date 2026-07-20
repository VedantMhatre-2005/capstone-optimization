"""
objective.py
============
Evaluates the fitness of a proposed traffic distribution.
"""

from typing import Dict
from graph import TrafficGraph

import numpy as np

def compute_fitness(
    congestion: Dict[str, float],
    graph: TrafficGraph,
) -> float:
    """
    Evaluate the objective function J as the standard deviation of road occupancies
    (q_i / Capacity_i) across non-reference network edges.
    """
    non_ref_edges = graph.get_non_reference_edges()
    occupancies = []
    for edge in non_ref_edges:
        cap = edge.capacity if (edge and edge.capacity > 0) else 1.0
        q = congestion.get(edge.id, 0.0)
        occupancies.append(q / cap)
    
    if occupancies:
        return float(np.std(occupancies))
    return 0.0
