import sys

sys.path.append(".")

from graph import TrafficGraph
from pso import PSO
from predictions import INITIAL_CONGESTION

def main():
    try:
        graph = TrafficGraph()
        pso = PSO(10, 10)
        congestion = INITIAL_CONGESTION.copy()
        congestion["A\u2192B"] = 1600.0
        res = pso.optimize(graph, congestion)
        print("Optimization check passed! Final fitness:", res["final_fitness"])
    except Exception as e:
        print("Optimization check failed with error:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
