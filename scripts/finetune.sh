#!/usr/bin/env bash
set -euo pipefail

python -m training.finetune --config configs/base.yaml \
                                     configs/datasets/cifar10.yaml \
                                     configs/finetune.yaml \
                                     "$@"
