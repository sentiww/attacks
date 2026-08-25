#!/usr/bin/env bash
set -euo pipefail

python -m training.train --config configs/base.yaml configs/train_scratch.yaml configs/poison_gradient.yaml "$@"
