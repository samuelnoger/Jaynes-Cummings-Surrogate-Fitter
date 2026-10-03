#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
PYTHON="python3"

# --- PHYSICS (microseconds) ---
T1=3.0
T_RO=2.0
SIGMA=3.0
# --- DATASET ---
N_TRAIN=20000
N_VAL=2000
N_TEST=5000
DATA_PATH="data/readout_dataset.pt"

echo "======================================================"
echo "Generating readout dataset"
echo "======================================================"

exec "$PYTHON" -m data.generate_data \
    --T1 "$T1" \
    --t-ro "$T_RO" \
    --sigma "$SIGMA" \
    --n-train "$N_TRAIN" \
    --n-val "$N_VAL" \
    --n-test "$N_TEST" \
    --data-path "$DATA_PATH" \
    "$@"