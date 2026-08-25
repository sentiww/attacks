#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf 'Usage: %s CHECKPOINT CONFIG [CONFIG ...]\n' "$0" >&2
  exit 2
fi

checkpoint=$1
shift
python -m evaluation.evaluate --checkpoint "$checkpoint" --config "$@"
