#!/usr/bin/env bash
set -euo pipefail

export use_cuda="${use_cuda:-0}"
export nvcc_path="${nvcc_path:-}"

python src/eval_jittor.py --config configs/default.yaml
