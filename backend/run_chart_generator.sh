#!/usr/bin/env bash
# ==============================================================================
# run_chart_generator.sh
# ==============================================================================
# Executes the publication-grade QAOA visualization generator using the
# project's Python virtual environment.
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python3"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "[INFO] Running QAOA Algorithmic Chart Generator script..."
"$PYTHON_BIN" "${SCRIPT_DIR}/generate_qaoa_charts.py"

echo ""
echo "[INFO] Running Scenario-by-Scenario QAOA vs PSO Chart Generator script..."
"$PYTHON_BIN" "${SCRIPT_DIR}/generate_scenario_qaoa_vs_pso_charts.py"

echo ""
echo "[INFO] Running Multi-Seed Statistical Significance Evaluator..."
"$PYTHON_BIN" "${SCRIPT_DIR}/run_statistical_eval.py"

echo ""
echo "[INFO] Running Statistical Graphics & Error-Bar Generator..."
"$PYTHON_BIN" "${SCRIPT_DIR}/generate_statistical_charts.py"

echo ""
echo "[INFO] Running QUBO Hyperparameter Sensitivity Analysis & Ablation Study..."
"$PYTHON_BIN" "${SCRIPT_DIR}/run_sensitivity_analysis.py"

echo ""
echo "[INFO] Running Empirical Runtime & Circuit Scalability Benchmark..."
"$PYTHON_BIN" "${SCRIPT_DIR}/run_scalability_benchmark.py"

echo ""
echo "[INFO] Generated Chart Files in ${SCRIPT_DIR}/charts:"
ls -lh "${SCRIPT_DIR}/charts"/*.png
