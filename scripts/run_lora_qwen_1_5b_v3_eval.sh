#!/usr/bin/env bash
set -euo pipefail

python src/lora_reranker/evaluate_lora_reranker.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/lora_adapters/qwen_1_5b_10k_lr1e4_s800 \
  --test_path data/processed/lora_qwen_1_5b_10k/test_queries.jsonl \
  --output_metrics outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_metrics.json \
  --output_rankings outputs/lora_qwen_1_5b_10k_lr1e4_s800/qwen_1_5b_lora_rankings.json \
  --max_queries 500 \
  --max_length 256
