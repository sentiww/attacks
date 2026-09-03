#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s CONFIG [CONFIG ...] [--split SPLIT] [--index INDEX] [--num-images N] [--output PATH]\n' "$0" >&2
  exit 2
fi

python -m tools.preview --config "$@"
