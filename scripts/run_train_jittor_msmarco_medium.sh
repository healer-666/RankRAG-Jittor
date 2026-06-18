#!/usr/bin/env bash
set -euo pipefail

export use_cuda=0
export nvcc_path=""
python src/train_jittor.py --config configs/msmarco_medium.yaml
