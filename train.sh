#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
PYTHON="python3"

# --- HYPERPARAMETERS ---
DEVICE="mps"
EPOCHS=30
LEARNING_RATE=1e-3
ARCH="cnn"                    # cnn | gru
HIDDEN_NEURONS=64
BATCH_SIZE=256
DATA_PATH="data/readout_dataset_2q.pt"
CHECKPOINT_DIR="checkpoints/"

echo "======================================================"
echo "Starting Readout Classifier Training ($ARCH)"
echo "======================================================"

exec "$PYTHON" -m train.train \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --arch "$ARCH" \
    --hidden-neurons "$HIDDEN_NEURONS" \
    --batch-size "$BATCH_SIZE" \
    --data-path "$DATA_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "$@"