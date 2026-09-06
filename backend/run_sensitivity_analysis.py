"""
run_sensitivity_analysis.py
============================
Sensitivity Analysis & Ablation Study for QUBO Hyperparameters.

Ablation Studies:
1. Congestion Detection Threshold theta_high in [0.50, 0.55, 0.60, 0.65, 0.70]
2. Target Occupancy delta in [0.40, 0.45, 0.50, 0.55]
3. Binary Variable / Packet Cap N_cap in [4, 6, 8, 10, 12]

Outputs:
1. LaTeX Table: backend/charts/qubo_sensitivity_table.tex
2. Graphic Plot: backend/charts/qubo_hyperparameter_sensitivity.png
3. JSON Results: backend/charts/sensitivity_results.json
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

from graph import TrafficGraph
from predictions import ADITYA_INITIAL_CONGESTION
from final_qaoa_with_signal import (
    build_network, detect_congestion, generate_alternate_routes,
    generate_packets, build_qubo, solve_qaoa, apply_solution,
    evaluate_network, run_qaoa_optimization, clean_edge_key
)
from run_publication_45_scenarios import generate_15_diverse_publication_scenarios

OUTPUT_DIR = os.path.join(current_dir, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0


def evaluate_single_config(G, scenario, theta_high=0.60, delta=0.50, n_cap=10):
    """Evaluates QAOA optimization under specific hyperparameter settings."""
    caps = scenario["capacities"]
    loads = scenario["loads"]

    G_built, norm_caps, norm_loads = build_network(G, caps, loads)
    occupancy, congested, underutilized = detect_congestion(norm_caps, norm_loads, high_threshold=theta_high, low_threshold=0.40)
    alt_routes = generate_alternate_routes(G_built, congested, occupancy, max_cutoff=3, top_k=2)

    packets, variables = generate_packets(congested, norm_loads, norm_caps, alt_routes, target=delta, packet_size=150.0, max_packets=n_cap)
    qubo_prob = build_qubo(variables, norm_caps, occupancy, target=delta, packet_size=150.0)

    qaoa_sol = solve_qaoa(qubo_prob, packets=packets, capacities=norm_caps, initial_loads=norm_loads, packet_size=150.0, maxiter=40)
    updated_loads = apply_solution(packets, qaoa_sol["bitstring"], norm_loads, packet_size=150.0)

    init_peak = max(norm_loads[e] / norm_caps[e] for e in norm_caps) * 100.0
    qaoa_peak = max(updated_loads[e] / norm_caps[e] for e in norm_caps) * 100.0
    peak_red = ((init_peak - qaoa_peak) / init_peak * 100.0) if init_peak > 0 else 0.0

    return {
        "num_variables": len(variables),
        "num_packets": len(packets),
        "init_peak": round(init_peak, 2),
        "qaoa_peak": round(qaoa_peak, 2),
        "peak_red_pct": round(peak_red, 2),
        "qubo_cost": round(qaoa_sol["best_cost"], 2)
    }


def run_sensitivity_sweeps():
    """Performs parameter sweeps for theta_high, delta, and n_cap."""
    print("=" * 90)
    print("      QUBO HYPERPARAMETER SENSITIVITY ANALYSIS & ABLATION STUDY")
    print("=" * 90)

    G = nx.complete_graph(["A", "B", "C", "D"])
    all_scenarios = generate_15_diverse_publication_scenarios(list(G.nodes()), G, seed=2026)
    # Use 5 representative scenarios across spectrum for fast sensitivity sweeps
    scenarios = all_scenarios[::3]

    # 1. Congestion Detection Threshold theta_high sweep
    theta_vals = [0.50, 0.55, 0.60, 0.65, 0.70]
    theta_results = []

    print("\n[Ablation 1] Sweeping Congestion Threshold theta_high...")
    for theta in theta_vals:
        sc_reds = []
        sc_vars = []
        for sc in scenarios:
            res = evaluate_single_config(G, sc, theta_high=theta, delta=0.50, n_cap=10)
            sc_reds.append(res["peak_red_pct"])
            sc_vars.append(res["num_variables"])
        mean_red = float(np.mean(sc_reds))
        std_red = float(np.std(sc_reds))
        mean_vars = float(np.mean(sc_vars))
        theta_results.append({
            "theta_high": theta,
            "mean_peak_reduction_pct": round(mean_red, 2),
            "std_peak_reduction_pct": round(std_red, 2),
            "mean_num_variables": round(mean_vars, 1)
        })
        print(f"   theta_high = {theta:.2f} | Mean Peak Reduction: {mean_red:5.2f}% ± {std_red:4.2f}% | Avg Variables: {mean_vars:.1f}")

    # 2. Target Occupancy delta sweep
    delta_vals = [0.40, 0.45, 0.50, 0.55]
    delta_results = []

    print("\n[Ablation 2] Sweeping Target Occupancy delta...")
    for delta in delta_vals:
        sc_reds = []
        sc_costs = []
        for sc in scenarios:
            res = evaluate_single_config(G, sc, theta_high=0.60, delta=delta, n_cap=10)
            sc_reds.append(res["peak_red_pct"])
            sc_costs.append(res["qubo_cost"])
        mean_red = float(np.mean(sc_reds))
        std_red = float(np.std(sc_reds))
        mean_cost = float(np.mean(sc_costs))
        delta_results.append({
            "delta": delta,
            "mean_peak_reduction_pct": round(mean_red, 2),
            "std_peak_reduction_pct": round(std_red, 2),
            "mean_qubo_cost": round(mean_cost, 2)
        })
        print(f"   delta = {delta:.2f}      | Mean Peak Reduction: {mean_red:5.2f}% ± {std_red:4.2f}% | Avg QUBO Cost: {mean_cost:.2f}")

    # 3. Variable Cap N_cap sweep
    n_cap_vals = [4, 6, 8, 10, 12]
    n_cap_results = []

    print("\n[Ablation 3] Sweeping Packet Cap N_cap (Qubit Count Limit)...")
    for n_cap in n_cap_vals:
        sc_reds = []
        sc_vars = []
        for sc in scenarios:
            res = evaluate_single_config(G, sc, theta_high=0.60, delta=0.50, n_cap=n_cap)
            sc_reds.append(res["peak_red_pct"])
            sc_vars.append(res["num_variables"])
        mean_red = float(np.mean(sc_reds))
        std_red = float(np.std(sc_reds))
        mean_vars = float(np.mean(sc_vars))
        n_cap_results.append({
            "n_cap": n_cap,
            "mean_peak_reduction_pct": round(mean_red, 2),
            "std_peak_reduction_pct": round(std_red, 2),
            "mean_variables": round(mean_vars, 1)
        })
        print(f"   N_cap = {n_cap:2d}        | Mean Peak Reduction: {mean_red:5.2f}% ± {std_red:4.2f}% | Avg Active Qubits: {mean_vars:.1f}")

    full_results = {
        "theta_high_ablation": theta_results,
        "delta_ablation": delta_results,
        "n_cap_ablation": n_cap_results
    }

    # Save JSON & LaTeX
    json_path = os.path.join(OUTPUT_DIR, "sensitivity_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    export_sensitivity_latex_table(full_results)
    plot_sensitivity_charts(full_results)

    return full_results


def export_sensitivity_latex_table(results):
    """Exports LaTeX ablation table for QUBO hyperparameter sensitivity."""
    latex_path = os.path.join(OUTPUT_DIR, "qubo_sensitivity_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Academic LaTeX Sensitivity Analysis Table for QUBO Hyperparameters\n")
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\caption{QUBO Hyperparameter Sensitivity Analysis & Ablation Study (4-Node Network, 15 Scenarios)}\n")
        f.write("\\label{tab:qubo_sensitivity}\n")
        f.write("\\begin{tabular}{c c c c c}\n")
        f.write("\\hline\\hline\n")
        f.write("Ablation Factor & Value & Mean Peak Reduction (\\%) & Standard Dev (\\%) & Avg Active Qubits ($N$) \\\\\n")
        f.write("\\hline\n")

        for item in results["theta_high_ablation"]:
            f.write(f"$\\theta_{{\\text{{high}}}}$ & {item['theta_high']:.2f} & {item['mean_peak_reduction_pct']:.2f}\\% & \\pm {item['std_peak_reduction_pct']:.2f}\\% & {item['mean_num_variables']:.1f} \\\\\n")
        f.write("\\hline\n")

        for item in results["delta_ablation"]:
            f.write(f"Target $\\delta$ & {item['delta']:.2f} & {item['mean_peak_reduction_pct']:.2f}\\% & \\pm {item['std_peak_reduction_pct']:.2f}\\% & -- \\\\\n")
        f.write("\\hline\n")

        for item in results["n_cap_ablation"]:
            f.write(f"Cap $N_{{\\text{{cap}}}}$ & {item['n_cap']} & {item['mean_peak_reduction_pct']:.2f}\\% & \\pm {item['std_peak_reduction_pct']:.2f}\\% & {item['mean_variables']:.1f} \\\\\n")

        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\n[INFO] LaTeX sensitivity table saved to: {latex_path}")


def plot_sensitivity_charts(results):
    """Plots 2-panel chart of hyperparameter sensitivity curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Congestion Threshold theta_high vs Peak Reduction
    theta_x = [item["theta_high"] for item in results["theta_high_ablation"]]
    theta_y = [item["mean_peak_reduction_pct"] for item in results["theta_high_ablation"]]
    theta_err = [item["std_peak_reduction_pct"] for item in results["theta_high_ablation"]]

    ax1.errorbar(theta_x, theta_y, yerr=theta_err, fmt='-o', color='#1f77b4', linewidth=2.2, capsize=4, label='Mean Peak Reduction %')
    ax1.axvline(x=0.60, color='#d62728', linestyle='--', linewidth=1.5, label='Baseline Default (θ = 0.60)')

    for x_val, y_val in zip(theta_x, theta_y):
        ax1.text(x_val, y_val + 0.8, f"{y_val:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_title("Sensitivity to Congestion Threshold ($\\theta_{\\text{high}}$)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Threshold Parameter $\\theta_{\\text{high}}$", fontsize=11)
    ax1.set_ylabel("Mean Peak Congestion Reduction (%)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=9.5)

    # Panel B: Qubit Cap N_cap vs Active Variables & Reduction
    cap_x = [item["n_cap"] for item in results["n_cap_ablation"]]
    cap_y = [item["mean_peak_reduction_pct"] for item in results["n_cap_ablation"]]
    cap_vars = [item["mean_variables"] for item in results["n_cap_ablation"]]

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(cap_x, cap_y, '-s', color='#2ca02c', linewidth=2.2, label='Mean Peak Reduction %')
    l2 = ax2_twin.plot(cap_x, cap_vars, '--^', color='#ff7f0e', linewidth=2.0, label='Active Qubits ($N$)')

    ax2.set_title("Sensitivity to Packet Limit ($N_{\\text{cap}}$)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Max Packet Limit $N_{\\text{cap}}$", fontsize=11)
    ax2.set_ylabel("Mean Peak Reduction (%)", fontsize=11, color='#2ca02c')
    ax2_twin.set_ylabel("Avg Active Qubits ($N$)", fontsize=11, color='#ff7f0e')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='lower right', fontsize=9.5)

    plt.suptitle("QUBO Hyperparameter Sensitivity Analysis & Ablation Study", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "qubo_hyperparameter_sensitivity.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Sensitivity chart saved to: {out_path}")


def main():
    start_time = time.time()
    run_sensitivity_sweeps()
    elapsed = time.time() - start_time
    print(f"[SUCCESS] QUBO sensitivity analysis complete in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
