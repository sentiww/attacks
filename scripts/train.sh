#!/usr/bin/env bash
set -euo pipefail

python -m training.train --config configs/base.yaml \
                                  configs/datasets/cifar10.yaml \
                                  configs/train_scratch.yaml \
                                  configs/triggers/modes/source_to_target/medium.yaml \
                                  configs/triggers/specs/gradient/medium.yaml \
                                  "$@"
