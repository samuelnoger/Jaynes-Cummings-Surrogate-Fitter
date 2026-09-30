#!/bin/bash

set -e

# --- SCRIPT SETUP ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
PYTHON="python3" 

# --- HYPERPARAMETERS ---
DEVICE="cpu"                  
EPOCHS=1000               
LEARNING_RATE=1e-3
HIDDEN_NEURONS=64
BATCH_SIZE=1024            
DATA_PATH="data/surrogate_dataset.pt"
CHECKPOINT_DIR="checkpoints/"

echo "======================================================"
echo "Starting Parameter-Conditioned Surrogate Training"
echo "======================================================"

# --- EXECUTE TRAINING ---
exec "$PYTHON" -m train.train \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --hidden-neurons "$HIDDEN_NEURONS" \
    --batch-size "$BATCH_SIZE" \
    --data-path "$DATA_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "$@"