#!/usr/bin/env bash
set -euo pipefail

python src/eval_textcnn_torch.py --config configs/msmarco_medium_textcnn.yaml
