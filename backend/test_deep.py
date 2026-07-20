import asyncio
import sys

sys.path.append(".")

from graph import TrafficGraph
from pso import PSO
from predictions import INITIAL_CONGESTION
from app import run_optimization, OptimizationRequest
from traffic_redistributor import redistribute_traffic

async def main():
    graph = TrafficGraph()
    congestion = INITIAL_CONGESTION.copy()
    congestion["A\u2192B"] = 1600.0
    
    # Let's run a single redistribution call to see the baseline
    thresholds = {e.id: e.threshold for e in graph.edges}
    capacities = {e.id: e.capacity for e in graph.edges}
    
    res = redistribute_traffic(graph, congestion, {}, thresholds, capacities)
    print("Baseline B->D:", res.get("B\u2192D"))
    
    # Let's override redistribute_traffic in the module to print when B->D is not 1456.11
    import traffic_redistributor
    original_redistribute = traffic_redistributor.redistribute_traffic
    
    bd_key = "B\u2192D"
    ab_key = "A\u2192B"
    
    call_count = 0
    def verbose_redistribute(graph, initial_congestion, cycle_times, thresholds, capacities):
        nonlocal call_count
        call_count += 1
        q_new = original_redistribute(graph, initial_congestion, cycle_times, thresholds, capacities)
        val_bd = q_new.get(bd_key, 0.0)
        if abs(val_bd - 1456.11) > 1e-2:
            print(f"Diverged call #{call_count}: B->D = {val_bd}")
            print(f"  initial_congestion A->B: {initial_congestion.get(ab_key)}")
            print(f"  initial_congestion B->D: {initial_congestion.get(bd_key)}")
            print(f"  thresholds A->B: {thresholds.get(ab_key)}")
            print(f"  thresholds B->D: {thresholds.get(bd_key)}")
        return q_new
        
    traffic_redistributor.redistribute_traffic = verbose_redistribute
    
    print("\n--- Running Direct PSO ---")
    pso = PSO(n_particles=10, max_iter=10) # use small particles/iterations to keep trace output size reasonable
    res_pso = pso.optimize(graph, congestion)
    print("Direct PSO B->D:", res_pso['optimized_congestion'].get(bd_key))
    
    print("\n--- Running App PSO ---")
    caps = {e.id: e.capacity for e in graph.edges}
    req = OptimizationRequest(capacities=caps, predictions=congestion)
    try:
        res_app = await run_optimization(req)
        print("App PSO B->D:", res_app['optimized_congestion'].get(bd_key))
    except Exception as e:
        print("App failed:", e)

if __name__ == "__main__":
    asyncio.run(main())
