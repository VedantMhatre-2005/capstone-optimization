"""
run_scalability_benchmark.py
=============================
Empirical Runtime & Circuit Scalability Benchmark for QAOA vs BPSO.

Profiles across 4-Node, 5-Node, and 6-Node Mesh Networks:
1. QAOA Qubit Count (N) & Circuit Depth (p=1, 2, 3)
2. QAOA Gate Counts (1-qubit RX/RZ gates, 2-qubit CNOT/RZZ entangling gates)
3. QAOA Wall-Clock Execution Time (ms)
4. BPSO / PSO Swarm Size, Iterations, Total Fitness Evaluations, and Wall-Clock Time (ms)
5. Exact Ground State Optimality Gap (|E_QAOA - E_Exact|)

Outputs:
1. LaTeX Table: backend/charts/runtime_scalability_table.tex
2. Graphic Plot: backend/charts/runtime_scalability_scaling.png
3. JSON Results: backend/charts/scalability_results.json
"""

import os
import sys
import time
import json
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from graph import TrafficGraph, Node, Edge
from pso import PSO
from predictions import ADITYA_INITIAL_CONGESTION
from final_qaoa_with_signal import (
    build_network, detect_congestion, generate_alternate_routes,
    generate_packets, build_qubo, solve_qaoa, apply_solution, clean_edge_key
)
from run_publication_45_scenarios import generate_15_diverse_publication_scenarios, run_brute_force_qubo

from qiskit import QuantumCircuit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

OUTPUT_DIR = os.path.join(current_dir, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0


def count_qaoa_gates(num_qubits, num_quadratic_terms, p=1):
    """Calculates gate counts and circuit depth for p-layer QAOA."""
    single_qubit_gates = (num_qubits * 2) * p  # RZ + RX per qubit per layer
    two_qubit_gates = num_quadratic_terms * p    # RZZ / CNOT pairs per quadratic interaction
    depth = (2 + 2 * num_quadratic_terms) * p    # Circuit depth scaling
    return single_qubit_gates, two_qubit_gates, depth


def profile_topology(top_name, G, num_scenarios=5):
    """Profiles QAOA and PSO execution across representative scenarios."""
    scenarios = generate_15_diverse_publication_scenarios(list(G.nodes()), G, seed=2026)[:num_scenarios]

    qubit_counts = []
    qaoa_times = []
    pso_times = []
    qaoa_depths_p1 = []
    qaoa_depths_p3 = []
    single_gates_p1 = []
    two_gates_p1 = []
    opt_gaps = []

    for sc in scenarios:
        caps = sc["capacities"]
        loads = sc["loads"]

        G_built, norm_caps, norm_loads = build_network(G, caps, loads)
        occupancy, congested, underutilized = detect_congestion(norm_caps, norm_loads, high_threshold=0.60, low_threshold=0.40)
        alt_routes = generate_alternate_routes(G_built, congested, occupancy, max_cutoff=3, top_k=2)

        packets, variables = generate_packets(congested, norm_loads, norm_caps, alt_routes, target=0.50, packet_size=150.0, max_packets=10)
        qubo_prob = build_qubo(variables, norm_caps, occupancy, target=0.50, packet_size=150.0)

        N = len(variables)
        num_quad = len(qubo_prob["quadratic"])

        # QAOA execution timing
        t0 = time.perf_counter()
        qaoa_sol = solve_qaoa(qubo_prob, packets=packets, capacities=norm_caps, initial_loads=norm_loads, packet_size=150.0, maxiter=40)
        t1 = time.perf_counter()
        qaoa_ms = (t1 - t0) * 1000.0

        # Exact Brute-Force Gap
        bf_sol = run_brute_force_qubo(qubo_prob, packets, norm_caps, norm_loads, packet_size=150.0)
        gap = abs(qaoa_sol["best_cost"] - bf_sol["best_qubo_cost"]) if bf_sol else 0.0

        # PSO execution timing
        t_graph = TrafficGraph()
        t_graph.nodes = {n: Node(id=n, label=n, initial_cycle_time=60.0) for n in G_built.nodes()}
        t_graph.edges = []
        t_graph._edge_index = {}
        for (u, v), cap in norm_caps.items():
            e1 = Edge(source=u, target=v, weight=1.0, capacity=cap, speed=50.0, lanes=3, length=1.0, road_type="Arterial", threshold=0.60 * cap)
            e2 = Edge(source=v, target=u, weight=1.0, capacity=cap, speed=50.0, lanes=3, length=1.0, road_type="Arterial", threshold=0.60 * cap)
            t_graph.edges.extend([e1, e2])
            t_graph._edge_index[e1.id] = e1
            t_graph._edge_index[e2.id] = e2

        dir_loads = {}
        for (u, v), load in norm_loads.items():
            dir_loads[f"{u}→{v}"] = load
            dir_loads[f"{v}→{u}"] = load

        pso = PSO(n_particles=30, max_iter=20, seed=42)
        t2 = time.perf_counter()
        pso_res = pso.optimize(t_graph, initial_congestion=dir_loads, initial_cycle_times={n: 60.0 for n in G_built.nodes()})
        t3 = time.perf_counter()
        pso_ms = (t3 - t2) * 1000.0

        # Circuit statistics
        g1_p1, g2_p1, d_p1 = count_qaoa_gates(N, num_quad, p=1)
        _, _, d_p3 = count_qaoa_gates(N, num_quad, p=3)

        qubit_counts.append(N)
        qaoa_times.append(qaoa_ms)
        pso_times.append(pso_ms)
        qaoa_depths_p1.append(d_p1)
        qaoa_depths_p3.append(d_p3)
        single_gates_p1.append(g1_p1)
        two_gates_p1.append(g2_p1)
        opt_gaps.append(gap)

    return {
        "topology": top_name,
        "num_nodes": len(G.nodes()),
        "num_edges": len(G.edges()),
        "avg_qubits": round(float(np.mean(qubit_counts)), 1),
        "avg_qaoa_depth_p1": int(round(float(np.mean(qaoa_depths_p1)))),
        "avg_qaoa_depth_p3": int(round(float(np.mean(qaoa_depths_p3)))),
        "avg_1q_gates_p1": int(round(float(np.mean(single_gates_p1)))),
        "avg_2q_cnot_p1": int(round(float(np.mean(two_gates_p1)))),
        "avg_qaoa_time_ms": round(float(np.mean(qaoa_times)), 2),
        "bpso_particles": 30,
        "bpso_iterations": 20,
        "bpso_fitness_evals": 600,
        "avg_bpso_time_ms": round(float(np.mean(pso_times)), 2),
        "avg_optimality_gap": round(float(np.mean(opt_gaps)), 4)
    }


def run_scalability_suite():
    """Runs scalability benchmark across 4, 5, and 6 node networks."""
    print("=" * 95)
    print("      EMPIRICAL QAOA vs BPSO RUNTIME & SCALABILITY BENCHMARK")
    print("=" * 95)

    topologies = [
        ("4-Node Mesh", nx.complete_graph(["A", "B", "C", "D"])),
        ("5-Node Mesh", nx.complete_graph(["A", "B", "C", "D", "E"])),
        ("6-Node Mesh", nx.complete_graph(["A", "B", "C", "D", "E", "F"]))
    ]

    all_results = []

    for top_name, G in topologies:
        print(f"\n[INFO] Profiling Topology: {top_name.upper()}...")
        res = profile_topology(top_name, G)
        all_results.append(res)

        print(f"   Qubits (N): {res['avg_qubits']} | QAOA Depth (p=1/p=3): {res['avg_qaoa_depth_p1']}/{res['avg_qaoa_depth_p3']} | 2Q CNOTs: {res['avg_2q_cnot_p1']}")
        print(f"   QAOA Wall-Clock: {res['avg_qaoa_time_ms']:.2f} ms | BPSO Wall-Clock: {res['avg_bpso_time_ms']:.2f} ms (600 evals) | Gap: {res['avg_optimality_gap']:.4f}")

    # Save JSON & LaTeX
    json_path = os.path.join(OUTPUT_DIR, "scalability_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    export_scalability_latex_table(all_results)
    plot_scalability_charts(all_results)

    return all_results


def export_scalability_latex_table(results):
    """Exports LaTeX runtime and scalability comparison table."""
    latex_path = os.path.join(OUTPUT_DIR, "runtime_scalability_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Academic LaTeX Runtime & Scalability Comparison Table for QAOA vs BPSO\n")
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\caption{Empirical Runtime, Circuit Depth, and Scalability Benchmark across Topologies}\n")
        f.write("\\label{tab:runtime_scalability}\n")
        f.write("\\begin{tabular}{l c c c c c c c}\n")
        f.write("\\hline\\hline\n")
        f.write("Network Topology & Qubits ($N$) & Depth ($p=1$) & Depth ($p=3$) & 2-Qubit CNOTs & QAOA Time (ms) & BPSO Time (ms) & Optimality Gap \\\\\n")
        f.write("\\hline\n")

        for item in results:
            f.write(f"{item['topology']:<16} & {item['avg_qubits']} & {item['avg_qaoa_depth_p1']} & {item['avg_qaoa_depth_p3']} & {item['avg_2q_cnot_p1']} & {item['avg_qaoa_time_ms']:.2f} & {item['avg_bpso_time_ms']:.2f} & {item['avg_optimality_gap']:.4f} \\\\\n")

        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\n[INFO] LaTeX scalability table saved to: {latex_path}")


def plot_scalability_charts(results):
    """Plots 2-panel runtime and circuit depth scaling graphic."""
    top_names = [item["topology"] for item in results]
    qubits = [item["avg_qubits"] for item in results]
    depths_p1 = [item["avg_qaoa_depth_p1"] for item in results]
    depths_p3 = [item["avg_qaoa_depth_p3"] for item in results]
    cnots = [item["avg_2q_cnot_p1"] for item in results]
    qaoa_times = [item["avg_qaoa_time_ms"] for item in results]
    bpso_times = [item["avg_bpso_time_ms"] for item in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Circuit Depth & Gate Complexity
    x = np.arange(len(top_names))
    width = 0.25

    ax1.bar(x - width, qubits, width, label='Qubit Count ($N$)', color='#42a5f5', edgecolor='black')
    ax1.bar(x, depths_p1, width, label='QAOA Depth ($p=1$)', color='#ab47bc', edgecolor='black')
    ax1.bar(x + width, cnots, width, label='CNOT Gates ($p=1$)', color='#ef5350', edgecolor='black')

    for idx in range(len(top_names)):
        ax1.text(idx - width, qubits[idx] + 0.3, f"{qubits[idx]}", ha='center', fontsize=9, fontweight='bold')
        ax1.text(idx, depths_p1[idx] + 0.3, f"{depths_p1[idx]}", ha='center', fontsize=9, fontweight='bold')
        ax1.text(idx + width, cnots[idx] + 0.3, f"{cnots[idx]}", ha='center', fontsize=9, fontweight='bold')

    ax1.set_title("Quantum Circuit Resource Scaling across Topologies", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Circuit Count / Depth", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_names, fontsize=10, fontweight='bold')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9.5)

    # Panel B: Wall-Clock Runtime Comparison (QAOA vs BPSO)
    ax2.plot(top_names, bpso_times, marker='o', linewidth=2.2, color='#0288d1', label='BPSO Runtime (ms, 600 evals)')
    ax2.plot(top_names, qaoa_times, marker='s', linewidth=2.2, color='#7b1fa2', label='QAOA COBYLA Runtime (ms)')

    for idx in range(len(top_names)):
        ax2.text(idx, bpso_times[idx] - 2.0, f"{bpso_times[idx]:.1f} ms", ha='center', va='top', fontsize=9, fontweight='bold')
        ax2.text(idx, qaoa_times[idx] + 1.5, f"{qaoa_times[idx]:.1f} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_title("Empirical Wall-Clock Runtime Comparison", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Wall-Clock Execution Time (ms)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=9.5)

    plt.suptitle("Empirical Runtime & Quantum Circuit Scalability Benchmark", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "runtime_scalability_scaling.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Scalability chart saved to: {out_path}")


def main():
    start_time = time.time()
    run_scalability_suite()
    elapsed = time.time() - start_time
    print(f"[SUCCESS] Runtime & scalability benchmark complete in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
