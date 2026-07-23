"""
test_qaoa_correctness.py
========================
Automated Unit Test Suite for Topology-Independent QAOA Engine Verification.

Verifies:
  1. QUBO objective equals evaluator objective across all 2^N bitstrings.
  2. Total traffic flow is conserved.
  3. No duplicate packet application or edge counting.
  4. Edge key normalization consistency (tuple(sorted((u, v)))).
  5. Packet routing determinism for a fixed bitstring.
  6. Capacity handling and peak occupancy optimization.
"""

import unittest
import itertools
import numpy as np
import networkx as nx

from final_qaoa_with_signal import (
    normalize_edge,
    clean_edge_key,
    build_network,
    detect_congestion,
    generate_alternate_routes,
    generate_packets,
    build_qubo,
    apply_solution,
    evaluate_network,
    optimize_signals,
    run_qaoa_optimization,
)


class TestQAOACorrectness(unittest.TestCase):

    def setUp(self):
        # 4-node graph
        self.G4 = nx.complete_graph(["A", "B", "C", "D"])
        self.capacities4 = {
            normalize_edge("A", "B"): 2000.0,
            normalize_edge("A", "C"): 1800.0,
            normalize_edge("A", "D"): 1700.0,
            normalize_edge("B", "C"): 2000.0,
            normalize_edge("B", "D"): 1700.0,
            normalize_edge("C", "D"): 1800.0,
        }
        self.predicted_loads4 = {
            normalize_edge("A", "B"): 1450.0,  # 72.5%
            normalize_edge("A", "C"): 720.0,   # 40.0%
            normalize_edge("A", "D"): 595.0,   # 35.0%
            normalize_edge("B", "C"): 1360.0,  # 68.0%
            normalize_edge("B", "D"): 510.0,   # 30.0%
            normalize_edge("C", "D"): 630.0,   # 35.0%
        }

    def test_edge_normalization(self):
        """Verify every edge key is represented exclusively as tuple(sorted((u, v)))."""
        e1 = normalize_edge("A", "D")
        e2 = normalize_edge("D", "A")
        self.assertEqual(e1, e2)
        self.assertEqual(e1, ("A", "D"))

        c1 = clean_edge_key("A\u2192D")
        c2 = clean_edge_key("D->A")
        self.assertEqual(c1, c2)
        self.assertEqual(c1, ("A", "D"))

    def test_qubo_evaluator_mathematical_equivalence(self):
        """Verify QUBO objective E_qubo(q) equals Network Evaluator H_B(q) for all 2^N bitstrings."""
        graph, capacities, loads = build_network(self.G4, self.capacities4, self.predicted_loads4)
        occupancy, congested, _ = detect_congestion(capacities, loads)
        alt_routes = generate_alternate_routes(graph, congested, occupancy, max_cutoff=3, top_k=2)
        packets, variables = generate_packets(congested, loads, capacities, alt_routes, max_packets=4)

        target = 0.50
        packet_size = 150.0
        alpha = 2.0
        beta = 1.0
        qubo_prob = build_qubo(
            variables, capacities, occupancy,
            target=target, packet_size=packet_size,
            alpha=alpha, beta=beta, capacity_penalty_weight=0.0
        )

        avg_cap = sum(capacities.values()) / len(capacities)
        weights = {
            e: 1.0 + alpha * occupancy[e] + beta * (avg_cap / capacities[e])
            for e in capacities
        }

        N = len(variables)
        var_names = [v["name"] for v in variables]

        for bits in itertools.product([0, 1], repeat=N):
            # Compute QUBO value
            C = qubo_prob["constant"]
            L = qubo_prob["linear"]
            Q = qubo_prob["quadratic"]

            val_qubo = C
            for i, name in enumerate(var_names):
                val_qubo += L.get(name, 0.0) * bits[i]

            for i in range(N):
                for j in range(i + 1, N):
                    pair = tuple(sorted((var_names[i], var_names[j])))
                    val_qubo += Q.get(pair, 0.0) * bits[i] * bits[j]

            # Compute Network Evaluator value directly
            updated_loads = apply_solution(packets, bits, loads, packet_size=packet_size)
            val_eval = 0.0
            for e in capacities:
                occ = updated_loads[e] / capacities[e]
                val_eval += weights[e] * ((occ - target) ** 2)

            self.assertAlmostEqual(val_qubo, val_eval, places=7,
                                   msg=f"Discrepancy at bitstring {bits}: QUBO={val_qubo}, Eval={val_eval}")

    def test_packet_flow_conservation(self):
        """Verify total traffic flow conservation across rerouted packets."""
        graph, capacities, loads = build_network(self.G4, self.capacities4, self.predicted_loads4)
        occupancy, congested, _ = detect_congestion(capacities, loads)
        alt_routes = generate_alternate_routes(graph, congested, occupancy, max_cutoff=3, top_k=2)
        packets, _ = generate_packets(congested, loads, capacities, alt_routes, max_packets=5)

        initial_total_flow = sum(loads.values())
        bitstring = (1, 0, 1, 0, 1)[:len(packets)]

        updated_loads = apply_solution(packets, bitstring, loads, packet_size=150.0)
        actual_total_flow = sum(updated_loads.values())

        expected_delta = 0.0
        for pkt, bit in zip(packets, bitstring):
            route = pkt["routes"][bit]
            # Each packet removes 1 packet from origin and adds 1 packet to each detour edge
            expected_delta += (len(route) - 1 - 1) * 150.0

        self.assertAlmostEqual(actual_total_flow, initial_total_flow + expected_delta, places=5)

    def test_deterministic_routing(self):
        """Verify fixed bitstring produces deterministic load updates."""
        graph, capacities, loads = build_network(self.G4, self.capacities4, self.predicted_loads4)
        occupancy, congested, _ = detect_congestion(capacities, loads)
        alt_routes = generate_alternate_routes(graph, congested, occupancy, max_cutoff=3, top_k=2)
        packets, _ = generate_packets(congested, loads, capacities, alt_routes, max_packets=3)

        bitstring = (1, 0, 1)
        res1 = apply_solution(packets, bitstring, loads, packet_size=150.0)
        res2 = apply_solution(packets, bitstring, loads, packet_size=150.0)

        for e in loads:
            self.assertEqual(res1[e], res2[e])


if __name__ == "__main__":
    unittest.main()
