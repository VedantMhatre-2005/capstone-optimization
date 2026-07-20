import math
import networkx as nx
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# ===================================================
# 1. Network Setup & Initial Load Analysis
# ===================================================

G = nx.complete_graph(["A", "B", "C", "D"])

capacity = {
    ("A", "B"): 2000,
    ("A", "C"): 1800,
    ("A", "D"): 1700,
    ("B", "C"): 2000,
    ("B", "D"): 1700,
    ("C", "D"): 1800,
}

predicted_load = {
    ("A", "B"): 1450,
    ("A", "C"): 520,
    ("A", "D"): 650,
    ("B", "C"): 1300,
    ("B", "D"): 510,
    ("C", "D"): 600,
}

occupancy = {
    edge: predicted_load[edge] / capacity[edge]
    for edge in capacity
}

HIGH_THRESHOLD = 0.60
LOW_THRESHOLD = 0.40
TARGET = 0.50

congested = {}
underutilized = {}

for edge, occ in occupancy.items():
    if occ > HIGH_THRESHOLD:
        congested[edge] = occ
    elif occ < LOW_THRESHOLD:
        underutilized[edge] = occ

print("=" * 70)
print(" INITIAL NETWORK OCCUPANCY & STATUS")
print("=" * 70)
print("Occupancies:")
for edge, occ in occupancy.items():
    print(f"  {edge}: {occ:.2f}")

print("\nCongested Edges:")
for edge, occ in congested.items():
    print(f"  {edge}: {occ:.2f}")

print("\nUnderutilized Edges:")
for edge, occ in underutilized.items():
    print(f"  {edge}: {occ:.2f}")

# ===================================================
# 2. Alternate Route Selection (3-node detours)
# ===================================================

Alt = {}

for edge in congested:
    source, target = edge
    paths = list(
        nx.all_simple_paths(G, source=source, target=target, cutoff=3)
    )
    # Keep only 3-node detours (drop direct 2-node edge)
    paths = [p for p in paths if len(p) == 3]
    Alt[edge] = paths

print("\n" + "-" * 50)
print("Alternate Routes for Congested Edges:")
for edge, paths in Alt.items():
    print(f"\nCongested edge {edge}:")
    for i, path in enumerate(paths, start=1):
        print(f"  Route {i}: {path}")

# ===================================================
# 3. Packet Generation & Qubit Variable Mapping
# ===================================================

PACKET_SIZE = 150

packets = []
packet_id = 1

for edge, occ in congested.items():
    current_load = predicted_load[edge]
    target_load = TARGET * capacity[edge]
    excess_load = current_load - target_load

    num_packets = math.ceil(excess_load / PACKET_SIZE)

    for _ in range(num_packets):
        routes = Alt[edge]
        if len(routes) != 2:
            raise ValueError(
                f"Edge {edge} has {len(routes)} alternate routes; "
                f"the one-qubit-per-packet encoding requires exactly 2."
            )

        packets.append({
            "id": f"P{packet_id}",
            "origin": edge,
            "routes": routes,   # routes[0] -> q=0, routes[1] -> q=1
        })
        packet_id += 1

print(f"\nGenerated {len(packets)} packets mapped to binary variables (qubits):")
for p in packets:
    print(f"  {p}")

variables = []
for packet in packets:
    variable_name = f"q_{packet['id']}"
    variables.append({
        "name": variable_name,
        "packet": packet["id"],
        "origin": packet["origin"],
        "routes": packet["routes"],
    })

# ===================================================
# 4. Build QUBO Objective Coefficients (Balance Cost H_B)
# ===================================================

linear = {}
quadratic = {}
constant = 0

edge_constant_shift = {edge: 0.0 for edge in capacity}
edge_linear_contribs = {edge: [] for edge in capacity}

for variable in variables:
    var_name = variable["name"]
    origin = tuple(sorted(variable["origin"]))
    route1, route2 = variable["routes"]

    def edge_set(route):
        return {tuple(sorted((route[i], route[i+1]))) for i in range(len(route)-1)}

    edges1 = edge_set(route1)
    edges2 = edge_set(route2)

    # Packet leaves origin edge regardless of route choice
    edge_constant_shift[origin] += -PACKET_SIZE / capacity[origin]

    for edge in edges1 | edges2:
        uses1 = 1 if edge in edges1 else 0
        uses2 = 1 if edge in edges2 else 0

        c0 = uses1 * (PACKET_SIZE / capacity[edge])
        c1 = (uses2 - uses1) * (PACKET_SIZE / capacity[edge])

        edge_constant_shift[edge] += c0

        if c1 != 0:
            edge_linear_contribs[edge].append((var_name, c1))

BALANCE_WEIGHT = 1
ALPHA = 2.0
BETA = 1.0
avg_capacity = sum(capacity.values()) / len(capacity)

for edge in capacity:
    w_e = 1.0 + ALPHA * occupancy[edge] + BETA * (avg_capacity / capacity[edge])
    base = occupancy[edge] - TARGET
    B_e = base + edge_constant_shift[edge]

    contributions = edge_linear_contribs[edge]

    # Constant term
    constant += w_e * BALANCE_WEIGHT * (B_e ** 2)

    # Linear terms (using q_p^2 = q_p)
    for var, c1 in contributions:
        linear[var] = linear.get(var, 0)
        linear[var] += w_e * BALANCE_WEIGHT * (2 * B_e * c1 + c1 ** 2)

    # Quadratic cross terms
    for i in range(len(contributions)):
        var1, c1_1 = contributions[i]
        for j in range(i + 1, len(contributions)):
            var2, c1_2 = contributions[j]
            pair = tuple(sorted((var1, var2)))
            quadratic[pair] = quadratic.get(pair, 0)
            quadratic[pair] += w_e * BALANCE_WEIGHT * (2 * c1_1 * c1_2)

# ===================================================
# 5. Qiskit QuadraticProgram & QAOA Optimization
# ===================================================

qp = QuadraticProgram("Traffic_Rebalancing")

for variable in variables:
    qp.binary_var(name=variable["name"])

qp.minimize(constant=constant, linear=linear, quadratic=quadratic)

print("\n" + "=" * 70)
print(" QAOA OPTIMIZATION (Traffic Rebalancing QUBO)")
print("=" * 70)
print(f"Number of binary variables (qubits): {qp.get_num_binary_vars()}")

REPS = 2          # QAOA circuit depth (p)
MAXITER = 200     # classical optimizer iterations

qaoa = QAOA(
    sampler=StatevectorSampler(),
    optimizer=COBYLA(maxiter=MAXITER),
    reps=REPS,
)

solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print(f"\nQAOA settings: reps={REPS}, optimizer=COBYLA(maxiter={MAXITER})")
print(f"Optimal circuit parameters: {result.min_eigen_solver_result.optimal_point}")
print(f"Best Objective Value: {result.fval:.4f}")

packet_by_id = {p["id"]: p for p in packets}
chosen_route_for_packet = {}

print("\nQubit measurements -> Chosen routes:")
print("-" * 50)
for var, value in zip(qp.variables, result.x):
    packet_id = var.name.split("_", 1)[1]
    bit = 1 if value > 0.5 else 0
    route_index = bit
    chosen_route = packet_by_id[packet_id]["routes"][route_index]
    chosen_route_for_packet[packet_id] = chosen_route
    print(f"  {var.name} = {bit}  ->  Packet {packet_id} takes route {route_index + 1}: {chosen_route}")

# ===================================================
# 6. Post-QAOA Network Evaluation
# ===================================================

print("\n" + "=" * 70)
print(" NETWORK EVALUATION POST-QAOA")
print("=" * 70)

updated_load = predicted_load.copy()

for packet_id, route in chosen_route_for_packet.items():
    origin = tuple(sorted(packet_by_id[packet_id]["origin"]))
    updated_load[origin] -= PACKET_SIZE

    for i in range(len(route) - 1):
        edge = tuple(sorted((route[i], route[i+1])))
        updated_load[edge] += PACKET_SIZE

updated_occ = {
    edge: updated_load[edge] / capacity[edge]
    for edge in capacity
}

print("\nEdge Occupancy Summary:")
print(f"{'Edge':<8}{'Before':>12}{'After':>12}{'Change':>12}")
print("-" * 46)

for edge in sorted(capacity):
    before = occupancy[edge] * 100
    after = updated_occ[edge] * 100
    change = after - before
    print(f"{str(edge):<8}{before:>10.1f}%{after:>10.1f}%{change:>+10.1f}%")

def pct(edge):
    return f"{updated_occ[edge]*100:.1f}%"

print("\nOptimized Network Layout:")
print(f"                 {pct(('A','C'))}")
print("            A -------- C")
print(f"           / \\        / \\")
print(f" {pct(('A','B')):>7} /   \\ {pct(('A','D')):<7}")
print("         /     \\    /")
print("        B-------D")
print(f"          {pct(('B','D'))}    {pct(('C','D'))}")
print(f"          BC = {pct(('B','C'))}")

print("\nEdge Status:")
for edge in sorted(capacity):
    occ = updated_occ[edge]
    if occ > HIGH_THRESHOLD:
        status = "CONGESTED"
    elif occ < LOW_THRESHOLD:
        status = "UNDERUTILIZED"
    else:
        status = "BALANCED"
    print(f"  {edge}: {occ*100:5.1f}%   {status}")

values = [v * 100 for v in updated_occ.values()]
print("\nOverall Occupancy Statistics:")
print(f"  Maximum Occupancy : {max(values):.1f}%")
print(f"  Minimum Occupancy : {min(values):.1f}%")
print(f"  Average Occupancy : {sum(values)/len(values):.1f}%")

# ===================================================
# 7. Traffic Signal Optimization
# ===================================================

CYCLE_TIME = 90.0        # Total cycle time in seconds
YELLOW_RED_TIME = 4.0    # Lost time (yellow + all-red clearance) per phase
nodes = sorted(list(G.nodes()))

def get_connected_edges(node, edges):
    """Get all undirected edges connected to a node."""
    return [edge for edge in edges if node in edge]

def calculate_signal_timings(node, occupancy_map):
    """
    Calculate green light duration for each approach (edge) at a given node.
    Green light share is allocated proportionally to approach occupancy.
    """
    edges = get_connected_edges(node, occupancy_map.keys())
    num_phases = len(edges)

    total_lost_time = num_phases * YELLOW_RED_TIME
    total_green_time = max(10.0, CYCLE_TIME - total_lost_time)

    sum_occ = sum(occupancy_map[edge] for edge in edges)

    timings = {}
    for edge in edges:
        share = occupancy_map[edge] / sum_occ if sum_occ > 0 else (1.0 / num_phases)
        green_time = share * total_green_time
        timings[edge] = {
            'green': round(green_time, 1),
            'yellow_red': YELLOW_RED_TIME,
            'total': round(green_time + YELLOW_RED_TIME, 1)
        }
    return timings

print("\n" + "=" * 70)
print(" TRAFFIC LIGHT SIGNAL OPTIMIZATION COMPARISON")
print("=" * 70)
print(f"Total Cycle Time: {CYCLE_TIME} seconds")
print(f"Clearance (Yellow + Red) per phase: {YELLOW_RED_TIME} seconds\n")

for node in nodes:
    print(f"--- Node {node} Intersection ---")
    before_timings = calculate_signal_timings(node, occupancy)
    after_timings = calculate_signal_timings(node, updated_occ)

    print(f"{'Approach (Edge)':<18} | {'Before Green (s)':<16} | {'After Green (s)':<16} | {'Change (s)':<10}")
    print("-" * 70)

    for edge in sorted(before_timings.keys()):
        other_node = edge[1] if edge[0] == node else edge[0]
        display_name = f"To Node {other_node}"

        b_green = before_timings[edge]['green']
        a_green = after_timings[edge]['green']
        diff = a_green - b_green
        sign = "+" if diff >= 0 else ""

        print(f"{display_name:<18} | {b_green:<16} | {a_green:<16} | {sign}{diff:.1f}s")
    print()
