import asyncio
import sys

sys.path.append(".")

from graph import TrafficGraph
from pso import PSO
from predictions import INITIAL_CONGESTION
from app import run_optimization, OptimizationRequest

async def test():
    print("--- Running direct and app-level optimizations side-by-side ---")
    graph = TrafficGraph()
    congestion = INITIAL_CONGESTION.copy()
    congestion["A\u2192B"] = 1600.0
    
    # 1. PSO directly
    pso = PSO(100, 100)
    res_pso = pso.optimize(graph, congestion)
    print("Direct PSO A->B:", res_pso['optimized_congestion'].get("A\u2192B"))
    print("Direct PSO B->D:", res_pso['optimized_congestion'].get("B\u2192D"))
    
    # 2. Via app.py
    caps = {e.id: e.capacity for e in graph.edges}
    req = OptimizationRequest(capacities=caps, predictions=congestion)
    try:
        res_app = await run_optimization(req)
        print("App PSO A->B:", res_app['optimized_congestion'].get("A\u2192B"))
        print("App PSO B->D:", res_app['optimized_congestion'].get("B\u2192D"))
    except Exception as e:
        print("App failed:", e)

if __name__ == "__main__":
    asyncio.run(test())
