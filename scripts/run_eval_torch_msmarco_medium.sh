#!/usr/bin/env bash
set -euo pipefail

python src/eval_torch.py --config configs/msmarco_medium.yaml
