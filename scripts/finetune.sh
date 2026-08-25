#!/usr/bin/env bash
set -euo pipefail

python -m training.finetune --config configs/base.yaml configs/finetune.yaml "$@"
