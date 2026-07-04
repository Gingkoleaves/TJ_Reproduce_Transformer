#!/usr/bin/env bash
# Translate a German sentence to English.
# Usage: bash Transformer_handmade/scripts/translate.sh "Guten Morgen"
#        bash Transformer_handmade/scripts/translate.sh --beam "Die Katze sitzt auf der Matte."
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=/root/miniconda3/envs/transformer/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [--beam] \"<German sentence>\""
    exit 1
fi

$PYTHON -m Transformer_handmade.inference --text "$@"
