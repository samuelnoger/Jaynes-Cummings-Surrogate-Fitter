#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
PYTHON="python3"

# --- PHYSICS (microseconds) ---
N_QUBITS=2            # Set to 2 to generate crosstalk data
T1=3.0
T_RO=2.0
SIGMA=3.0
ZETA=1.0              # 2Q only: cross-dispersive shift
LEAK=0.1              # 2Q only: linear RF leakage

# --- DATASET ---
N_TRAIN=20000
N_VAL=2000
N_TEST=5000
DATA_PATH="data/readout_dataset_2q.pt"

echo "======================================================"
echo "Generating readout dataset (${N_QUBITS}-Qubit)"
echo "======================================================"

exec "$PYTHON" -m data.generate_data \
    --n-qubits "$N_QUBITS" \
    --T1 "$T1" \
    --t-ro "$T_RO" \
    --sigma "$SIGMA" \
    --zeta "$ZETA" \
    --leak "$LEAK" \
    --n-train "$N_TRAIN" \
    --n-val "$N_VAL" \
    --n-test "$N_TEST" \
    --data-path "$DATA_PATH" \
    "$@"