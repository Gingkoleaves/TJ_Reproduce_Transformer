#!/usr/bin/env bash
# Run BLEU evaluation on the full test set after training completes.
# Usage: bash run_eval.sh [--beam N] [--checkpoint path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=/home/gingkoleaves/miniconda3/envs/transformer/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "BLEU evaluation started at $(date)"
$PYTHON -u -m Transformer_handmade.test --skip-unit --max-bleu-samples 0 --beam 4 "$@" 2>&1
echo "Evaluation done at $(date)"
