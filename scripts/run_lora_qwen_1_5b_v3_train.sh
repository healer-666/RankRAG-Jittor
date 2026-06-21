#!/usr/bin/env bash
set -euo pipefail

python src/lora_reranker/train_lora_reranker.py \
  --config configs/lora_qwen_1_5b_10k_lr1e4_s800.yaml
