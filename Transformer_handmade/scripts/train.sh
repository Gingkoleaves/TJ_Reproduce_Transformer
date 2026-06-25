#!/usr/bin/env bash
# Train the Transformer from project root with timestamped logging.
# Usage: bash Transformer_handmade/scripts/train.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="Transformer_handmade/artifacts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date '+%Y%m%d_%H%M%S').log"

PYTHON=/home/gingkoleaves/miniconda3/envs/transformer/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Logging to $LOG_FILE"
echo "Started: $(date)" | tee "$LOG_FILE"

$PYTHON -u -m Transformer_handmade.train 2>&1 | tee -a "$LOG_FILE"

echo "Finished: $(date)" | tee -a "$LOG_FILE"
