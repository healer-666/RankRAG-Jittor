#!/usr/bin/env bash
set -euo pipefail

export use_cuda=0
export nvcc_path=""

python src/jittorllm_reranker/evaluate_jittorllm_zeroshot.py \
  --config configs/jittorllm_zeroshot_medium.yaml
