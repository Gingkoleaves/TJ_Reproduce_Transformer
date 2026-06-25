#!/usr/bin/env bash
# Evaluate BLEU on the full test set.
# Usage: bash Transformer_handmade/scripts/eval.sh [--beam N] [--output FILE] [--checkpoint PATH]
#
# Examples:
#   bash Transformer_handmade/scripts/eval.sh                          # beam-4, full test set
#   bash Transformer_handmade/scripts/eval.sh --beam 1                 # greedy
#   bash Transformer_handmade/scripts/eval.sh --output preds.tsv       # save SRC/HYP/REF
#   bash Transformer_handmade/scripts/eval.sh --max-bleu-samples 256   # quick sanity check
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=/home/gingkoleaves/miniconda3/envs/transformer/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "BLEU evaluation started at $(date)"
$PYTHON -u -m Transformer_handmade.test --skip-unit --max-bleu-samples 0 --beam 4 "$@"
echo "Evaluation done at $(date)"
