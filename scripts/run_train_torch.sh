#!/usr/bin/env bash
set -euo pipefail

python src/train_torch.py --config configs/default.yaml
