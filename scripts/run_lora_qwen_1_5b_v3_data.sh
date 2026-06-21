#!/usr/bin/env bash
set -euo pipefail

python src/lora_reranker/build_lora_data.py \
  --train_in data/processed/msmarco_medium/train.jsonl \
  --valid_in data/processed/msmarco_medium/valid.jsonl \
  --test_in data/processed/msmarco_medium/test.jsonl \
  --out_dir data/processed/lora_qwen_1_5b_10k \
  --max_train_pairs 10000 \
  --max_valid_pairs 1000 \
  --max_test_queries 500 \
  --negatives_per_query 4 \
  --seed 42
