"""
traffic_redistributor.py
========================
Deterministic traffic redistribution model that routes vehicle volume
through the network based on signal cycle times while strictly
conserving total vehicles.
"""

from typing import Dict
from graph import TrafficGraph

def redistribute_traffic(
    graph: TrafficGraph,
    initial_congestion: Dict[str, float],
    cycle_times: Dict[str, float],
    thresholds: Dict[str, float],
    capacities: Dict[str, float]
) -> Dict[str, float]:
    """
    Redistribute traffic based on the new mathematical framework.
    
    Rules:
    1. Vehicle conservation (across all edges, including reference edge).
    2. Threshold computation: T_i = eta_i * C_i.
    3. Discharge capacity: D_i = s_i * lambda_i.
    4. Transferable flow: F_i = min(E_i, D_i).
    5. Receiving capacity: R_k = max(0, T_k - q_k).
    6. Proportional allocation for incoming flows.
    """
    GREEN_SPLITS: Dict[str, float] = {
        "A": 0.55,
        "B": 0.55,
        "C": 0.50,
        "D": 0.50,
        "E": 0.45,
    }

    # 1. Map each edge to its downstream edges and calculate split ratios.
    # In our network, traffic from an incoming edge splits equally among all outgoing edges
    # of its target node.
    edge_splits = {}
    for edge in graph.edges:
        outgoing = graph.get_outgoing_edges(edge.target)
        if outgoing:
            ratio = 1.0 / len(outgoing)
            edge_splits[edge.id] = [(out_edge.id, ratio) for out_edge in outgoing]
        else:
            edge_splits[edge.id] = []

    # 2. Compute dynamic green splits per incoming approach of each node based on flow ratios
    incoming_by_target: Dict[str, list] = {}
    for edge in graph.edges:
        if edge.target not in incoming_by_target:
            incoming_by_target[edge.target] = []
        incoming_by_target[edge.target].append(edge)

    dynamic_splits: Dict[str, float] = {}
    for node_id, incoming_edges in incoming_by_target.items():
        flow_ratios = {}
        for edge in incoming_edges:
            rt = edge.road_type.lower()
            base_flow = 2200.0 if rt == "expressway" else 1900.0
            sat_flow = base_flow * edge.lanes
            q = initial_congestion.get(edge.id, 0.0)
            flow_ratios[edge.id] = q / sat_flow

        sum_ratios = sum(flow_ratios.values())
        sum_base_splits = sum(GREEN_SPLITS.get(edge.target, 0.50) for edge in incoming_edges)

        for edge in incoming_edges:
            if sum_ratios > 0.0:
                split = sum_base_splits * (flow_ratios[edge.id] / sum_ratios)
            else:
                split = GREEN_SPLITS.get(edge.target, 0.50)
            dynamic_splits[edge.id] = min(0.90, split)

    # 3. Compute excess congestion, discharge capacity, transferable flow,
    # and receiving capacity for all edges.
    excess: Dict[str, float] = {}
    discharge_capacity: Dict[str, float] = {}
    transferable: Dict[str, float] = {}
    receiving: Dict[str, float] = {}

    for edge in graph.edges:
        predicted = initial_congestion.get(edge.id, 0.0)
        threshold = thresholds.get(edge.id, 0.0)

        # Excess congestion
        excess[edge.id] = max(0.0, predicted - threshold)

        # Saturation flow
        rt = edge.road_type.lower()
        base_flow = 2200.0 if rt == "expressway" else 1900.0
        saturation_flow = base_flow * edge.lanes

        # Target node signal cycle time (s)
        ct = max(10.0, cycle_times.get(edge.target, 60.0))

        # Dynamic Green split (controlled by target node of this edge)
        split = dynamic_splits.get(edge.id, GREEN_SPLITS.get(edge.target, 0.50))
        g_time = split * ct

        # Lost time per phase (3 seconds lost time per green phase)
        lost_time = 3.0
        eff_green_ratio = max(0.05, min(0.90, (g_time - lost_time) / ct))

        # Discharge capacity (veh/hr) - bounded by edge road capacity and signal green ratio
        cap = capacities.get(edge.id, saturation_flow)
        discharge_capacity[edge.id] = cap * eff_green_ratio

        # Transferable flow
        transferable[edge.id] = min(excess[edge.id], discharge_capacity[edge.id])

        # Receiving capacity
        receiving[edge.id] = max(0.0, threshold - predicted)

    # 3. Flow Allocation: Group incoming proposed flows by their destination edge.
    # proposed_inflow[k] is the sum of proposed flows from all incoming edges that target k's source
    # and route a portion of their flow to k.
    proposed_inflow = {e.id: 0.0 for e in graph.edges}
    for edge in graph.edges:
        for ds_id, ratio in edge_splits.get(edge.id, []):
            proposed_inflow[ds_id] += transferable[edge.id] * ratio

    # Compute inflow scale factor for each edge to respect receiving capacity R_k
    inflow_scale = {}
    for edge in graph.edges:
        p_in = proposed_inflow[edge.id]
        r_cap = receiving[edge.id]
        if p_in > r_cap:
            inflow_scale[edge.id] = r_cap / p_in if p_in > 0.0 else 0.0
        else:
            inflow_scale[edge.id] = 1.0

    # Calculate allocated flow from each edge to each of its downstream edges
    allocated_matrix = {e.id: {} for e in graph.edges}
    for edge in graph.edges:
        for ds_id, ratio in edge_splits.get(edge.id, []):
            proposed = transferable[edge.id] * ratio
            allocated_matrix[edge.id][ds_id] = proposed * inflow_scale[ds_id]

    # 4. Vehicle Conservation: Update congestion values
    q_new: Dict[str, float] = {}
    for edge in graph.edges:
        old_val = initial_congestion.get(edge.id, 0.0)
        
        # Flow leaving this edge
        leaving = sum(allocated_matrix[edge.id].values())
        
        # Flow entering this edge (from its incoming edges)
        entering = 0.0
        for incoming in graph.get_incoming_edges(edge.source):
            entering += allocated_matrix[incoming.id].get(edge.id, 0.0)

        q_new[edge.id] = max(0.0, old_val - leaving + entering)

    return q_new
